# Policy Architecture & Training Configuration

---

## 1. Actor–Critic Architecture

The locomotion policy uses a separate actor and critic network, both implemented as fully connected networks.

| Property | Value |
|---|---|
| Hidden layers | [256, 128, 64] |
| Activation | ELU |
| Policy output | Gaussian action distribution |
| Initial exploration std (σ) | 1.0 |

**ELU activation:**

$$
\text{ELU}(x) = \begin{cases} x & x > 0 \\ e^x - 1 & x \leq 0 \end{cases}
$$

The critic additionally receives privileged observations (base linear velocity, joint efforts) not available to the policy at deployment time.

---

## 2. PPO Hyperparameters

Training uses the [RSL-RL](https://github.com/leggedrobotics/rsl_rl) on-policy PPO implementation.

| Hyperparameter | Value |
|---|---|
| Rollout horizon | 32 steps/env |
| Learning epochs | 5 |
| Mini-batches per update | 4 |
| Clipping parameter (ε) | 0.2 |
| Discount factor (γ) | 0.99 |
| GAE parameter (λ) | 0.95 |
| Initial learning rate (α) | 1e-3 |
| KL divergence target | 0.01 |
| Value loss coefficient | 0.75 |
| Entropy coefficient | 0.01 |
| Gradient clip threshold | 1.0 |

### PPO Clipped Objective

$$
L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t\right)\right]
$$

### Generalized Advantage Estimation

$$
\gamma = 0.99, \quad \lambda = 0.95
$$

---

## 3. Runner Configuration

The PPO runner is defined in `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`.

```python
# rsl_rl_ppo_cfg.py
class BasePPORunnerCfg(OnPolicyRunnerCfg):
    num_steps_per_env = 32
    save_interval = 100          # checkpoint every 100 iterations

    policy = ActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = PPOCfg(
        value_loss_coef=0.75,
        entropy_coef=0.01,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

---

## 4. Checkpoint Output

Trained checkpoints and exported policies are saved to:

```
logs/rsl_rl/[TASK-NAME]/
```

This integrates with the same export workflow used by the original `unitree_rl_lab` environments.