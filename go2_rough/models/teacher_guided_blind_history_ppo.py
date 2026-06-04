from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from go2_rough.models.frozen_teacher_policy import FrozenTeacherPolicy


class TeacherGuidedBlindHistoryPPO(PPO):
    """Blind-history PPO with time-decayed teacher imitation and latent-target regression."""

    def __init__(
        self,
        policy,
        *args,
        teacher_expert_path: str | None = None,
        latent_command_threshold: float = 0.1,
        latent_regression_coef_stage0: float = 0.0,
        latent_regression_coef_stage1: float = 0.0,
        latent_regression_coef_stage2: float = 0.0,
        latent_stage0_end: int = 300,
        latent_stage1_end: int = 800,
        imitation_command_threshold: float = 0.1,
        imitation_coef_stage0: float = 0.2,
        imitation_coef_stage1: float = 0.05,
        imitation_stage0_end: int = 300,
        imitation_stage1_end: int = 800,
        **kwargs,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        self.latent_command_threshold = float(latent_command_threshold)
        self.latent_regression_coef_stage0 = float(latent_regression_coef_stage0)
        self.latent_regression_coef_stage1 = float(latent_regression_coef_stage1)
        self.latent_regression_coef_stage2 = float(latent_regression_coef_stage2)
        self.latent_stage0_end = int(latent_stage0_end)
        self.latent_stage1_end = int(latent_stage1_end)
        self.imitation_command_threshold = float(imitation_command_threshold)
        self.imitation_coef_stage0 = float(imitation_coef_stage0)
        self.imitation_coef_stage1 = float(imitation_coef_stage1)
        self.imitation_stage0_end = int(imitation_stage0_end)
        self.imitation_stage1_end = int(imitation_stage1_end)
        self._update_counter = 0
        self.teacher_policy = None
        self.latent_loss_fn = nn.MSELoss(reduction="none")
        self.imitation_loss_fn = nn.MSELoss(reduction="none")

        if teacher_expert_path:
            self.teacher_policy = FrozenTeacherPolicy(
                checkpoint_path=teacher_expert_path,
                device=self.device,
            ).to(self.device)

    def _current_imitation_coef(self) -> float:
        if self._update_counter < self.imitation_stage0_end:
            return self.imitation_coef_stage0
        if self._update_counter < self.imitation_stage1_end:
            return self.imitation_coef_stage1
        return 0.0

    def _current_latent_regression_coef(self) -> float:
        if self._update_counter < self.latent_stage0_end:
            return self.latent_regression_coef_stage0
        if self._update_counter < self.latent_stage1_end:
            return self.latent_regression_coef_stage1
        return self.latent_regression_coef_stage2

    def _command_mask(self, obs_batch, threshold: float) -> torch.Tensor | None:
        if self.teacher_policy is None or "policy" not in obs_batch.keys():
            return None
        command = obs_batch["policy"][:, 9:12]
        return (torch.linalg.norm(command, dim=-1) > threshold).float()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_latent_regression_loss = 0
        mean_imitation_loss = 0

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
            entropy_batch = self.policy.entropy[:original_batch_size]

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

            latent_regression_loss = torch.tensor(0.0, device=self.device)
            imitation_loss = torch.tensor(0.0, device=self.device)
            latent_regression_coef = self._current_latent_regression_coef()
            imitation_coef = self._current_imitation_coef()

            if self.teacher_policy is not None:
                teacher_obs_batch = obs_batch[:original_batch_size]
                required_groups = {"policy", "terrain_privileged", "dynamics_privileged"}
                if required_groups.issubset(set(teacher_obs_batch.keys())):
                    if latent_regression_coef > 0.0 and hasattr(self.policy, "encode_history_target"):
                        latent_mask = self._command_mask(teacher_obs_batch, self.latent_command_threshold)
                        if latent_mask is not None and torch.count_nonzero(latent_mask) > 0:
                            student_target = self.policy.encode_history_target(teacher_obs_batch)
                            teacher_target = self.teacher_policy.get_latent_target(teacher_obs_batch).detach()
                            per_sample_latent = self.latent_loss_fn(student_target, teacher_target).mean(dim=-1)
                            latent_regression_loss = (per_sample_latent * latent_mask).sum() / (
                                latent_mask.sum() + 1e-6
                            )
                            loss = loss + latent_regression_coef * latent_regression_loss

                    if imitation_coef > 0.0:
                        mask = self._command_mask(teacher_obs_batch, self.imitation_command_threshold)
                        if mask is not None and torch.count_nonzero(mask) > 0:
                            teacher_actions = self.teacher_policy(teacher_obs_batch).detach()
                            per_sample = self.imitation_loss_fn(mu_batch, teacher_actions).sum(dim=-1)
                            imitation_loss = (per_sample * mask).sum() / (mask.sum() + 1e-6)
                            loss = loss + imitation_coef * imitation_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_latent_regression_loss += latent_regression_loss.item()
            mean_imitation_loss += imitation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_latent_regression_loss /= num_updates
        mean_imitation_loss /= num_updates

        self.storage.clear()
        self._update_counter += 1

        return {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "entropy": mean_entropy,
            "latent_regression_loss": mean_latent_regression_loss,
            "imitation_loss": mean_imitation_loss,
        }
