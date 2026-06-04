from __future__ import annotations

import math
import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from go2_rough.models.terrain_targets import terrain_lite_from_scan


class TerrainSupervisedTeacherPPO(PPO):
    """PPO with explicit supervision on the teacher terrain branch."""

    def __init__(
        self,
        policy,
        *args,
        terrain_regression_coef_stage0: float = 0.5,
        terrain_regression_coef_stage1: float = 0.2,
        terrain_regression_coef_stage2: float = 0.05,
        terrain_stage0_end: int = 300,
        terrain_stage1_end: int = 800,
        **kwargs,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        self.terrain_regression_coef_stage0 = float(terrain_regression_coef_stage0)
        self.terrain_regression_coef_stage1 = float(terrain_regression_coef_stage1)
        self.terrain_regression_coef_stage2 = float(terrain_regression_coef_stage2)
        self.terrain_stage0_end = int(terrain_stage0_end)
        self.terrain_stage1_end = int(terrain_stage1_end)
        self._update_counter = 0
        self.terrain_loss_fn = nn.MSELoss(reduction="none")

    def _current_terrain_regression_coef(self) -> float:
        if self._update_counter < self.terrain_stage0_end:
            return self.terrain_regression_coef_stage0
        if self._update_counter < self.terrain_stage1_end:
            return self.terrain_regression_coef_stage1
        return self.terrain_regression_coef_stage2

    def _stabilize_policy_std_params(self) -> None:
        min_std = float(getattr(self.policy, "min_std", 1.0e-6))
        with torch.no_grad():
            if hasattr(self.policy, "std"):
                self.policy.std.data = torch.nan_to_num(
                    self.policy.std.data, nan=min_std, posinf=1.0, neginf=min_std
                )
                self.policy.std.data.clamp_(min=min_std, max=1.0)
            if hasattr(self.policy, "log_std"):
                min_log_std = math.log(min_std)
                self.policy.log_std.data = torch.nan_to_num(
                    self.policy.log_std.data, nan=min_log_std, posinf=0.0, neginf=min_log_std
                )
                self.policy.log_std.data.clamp_(min=min_log_std, max=0.0)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_terrain_regression_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    min_std = float(getattr(self.policy, "min_std", 1.0e-6))
                    sigma_batch = sigma_batch.clamp(min=min_std, max=1.0)
                    old_sigma_batch = old_sigma_batch.clamp(min=min_std, max=1.0)
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            terrain_regression_loss = torch.tensor(0.0, device=self.device)
            terrain_regression_coef = self._current_terrain_regression_coef()
            if (
                terrain_regression_coef > 0.0
                and "terrain_privileged" in obs_batch.keys()
                and hasattr(self.policy, "predict_terrain_target")
            ):
                terrain_scan = obs_batch["terrain_privileged"][:original_batch_size]
                terrain_target = terrain_lite_from_scan(terrain_scan).detach()
                terrain_pred = self.policy.predict_terrain_target(obs_batch[:original_batch_size])
                per_sample = self.terrain_loss_fn(terrain_pred, terrain_target).mean(dim=-1)
                terrain_regression_loss = per_sample.mean()
                loss = loss + terrain_regression_coef * terrain_regression_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._stabilize_policy_std_params()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_terrain_regression_loss += terrain_regression_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_terrain_regression_loss /= num_updates

        self.storage.clear()
        self._update_counter += 1

        return {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "entropy": mean_entropy,
            "terrain_regression_loss": mean_terrain_regression_loss,
        }
