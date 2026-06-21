# Reward Functions

Custom reward terms are defined in `source/trakr_rl/trakr_rl/tasks/locomotion/mdp/rewards.py`. This document covers each term's motivation, formula, and tuning rationale.

---

## Standard Velocity Tracking Terms

The base reward structure follows the `unitree_rl_lab` velocity-tracking formulation.

### Velocity Tracking

$$
r_{\text{vel}} = \exp\!\left(-\frac{\left(v^{xy}_{\text{cmd}} - v^{xy}_{\text{yaw}}\right)^2}{2\sigma^2}\right)
$$

Where:
- $v^{xy}_{\text{cmd}}$ — commanded linear velocity in the horizontal plane
- $v^{xy}_{\text{yaw}}$ — measured linear velocity in the gravity-aligned yaw frame
- $\sigma$ — width of the exponential kernel

### Penalty Terms (standard)

| Term | What it discourages |
|---|---|
| Vertical motion | Unnecessary bobbing |
| Roll / pitch instability | Body tilt during locomotion |
| High joint velocities | Aggressive joint motion |
| High joint accelerations | Jerky actuation |
| Excessive torques | Mechanical stress |
| Rapid action changes | Unsmooth policy output |
| Energy consumption | Inefficient gaits |
| Joint limit violations | Hardware damage |
| Foot slipping | Loss of traction |
| Uneven foot air time | Asymmetric gait |
| Undesired non-foot contacts | Shin/body scraping |

---

## Custom Reward Terms

The following terms were added specifically for the Trakr platform to address observed failure modes during training.

---

### 1. Body Roll Penalty

**Problem:** The robot exhibited significant side-to-side oscillations during walking, leading to unstable gaits and occasional falls.

**Term:**

$$
r_{\text{roll}} = -k_{\text{roll}}\,\theta^2
$$

Where $\theta$ is the roll angle of the robot base and $k_{\text{roll}}$ is the penalty weight.

**Effect:** Encourages a level base posture during locomotion.

---

### 2. Pitch Rate Penalty

**Problem:** Large pitch rates caused the actuators and legs to lock up rigidly, making recovery impossible. This led to forward toppling on stair descents, slope exits, and minor terrain transitions.

**Term:**

$$
r_{\text{pitch rate}} = -k_{\text{pitch rate}}\,\omega_{\text{pitch}}^2
$$

Where $\omega_{\text{pitch}}$ is the pitch angular velocity of the robot base.

**Effect:** Encourages the robot to reposition its hind legs to damp pitch, resulting in smoother descents and reduced toppling.

---

### 3. Foot Height Reward

**Problem:** The robot failed to climb stairs — when encountering a stair edge, the policy generated insufficient swing-leg clearance, causing the feet to collide repeatedly with the obstacle and stall.

**Term:**

$$
r_{\text{feet height}} = \sum_{i=1}^{4} \left(h^{\text{toe}}_i - h_{\text{target}}\right)^2 \tanh\!\left(k_v \left\|v^{\text{body}}_{i,xy}\right\|\right)
$$

Where:
- $h^{\text{body}}_i$ — height of foot $i$ relative to the robot body frame
- $h_{\text{target}}$ — desired swing-foot clearance height
- $v^{\text{body}}_{i,xy}$ — foot velocity projected onto the horizontal plane in the body frame
- $k_v$ — velocity scaling parameter

The $\tanh$ term ensures the reward primarily affects swinging feet; stance feet contribute minimally. The reward is only active when the commanded velocity magnitude exceeds a threshold.

> **Note:** Foot height reward alone was insufficient — the robot cleared its feet but did not make forward progress. The velocity tracking reward weight was increased alongside this term to restore forward momentum.

---

### 4. Joint Position Penalty (tuned for wide gait suppression)

**Problem:** During stair training the robot learned to widen its stance for stability. This over-generalised to flat terrain and slopes, causing unnecessarily high adduction motor torques and energy waste.

**Term:**

$$
r_{\text{joint}} = \begin{cases}
\|q - q_{\text{default}}\|^2 & \text{if } \|c\| > 0 \text{ or } \|v^{xy}_b\| > v_{\text{th}} \\
s\,\|q - q_{\text{default}}\|^2 & \text{otherwise}
\end{cases}
$$

Where:
- $q$ — current joint configuration
- $q_{\text{default}}$ — default (standing) joint configuration
- $c$ — commanded base velocity
- $v^{xy}_b$ — base velocity in the horizontal plane
- $v_{\text{th}}$ — velocity threshold
- $s$ — stand-still scaling factor

**Effect:** Increasing this penalty's weight pushes the robot toward a compact nominal gait, while still allowing wider stances when needed for stair stability.

---

### 5. Stable Progress Reward

**Problem:** On stairs, the robot sometimes placed a single foot on a step and drove excessive force through that leg to advance. This imbalance caused it to topple sideways.

**Term:**

$$
r_{\text{stable}} = \max\!\left(0,\ v^{xy}_r \cdot \hat{v}^{xy}_{\text{cmd}}\right) \times \exp\!\left(-2\left(\omega_{\text{pitch}}^2 + \omega_{\text{roll}}^2\right)\right)
$$

Where:

$$
\hat{v}^{xy}_{\text{cmd}} = \frac{v^{xy}_{\text{cmd}}}{\|v^{xy}_{\text{cmd}}\| + \varepsilon}, \quad \varepsilon = 10^{-6}
$$

**Effect:** Forward progress is only rewarded when accompanied by stable body motion (low pitch and roll rates). This discourages exploiting aggressive single-leg pushes for short-term velocity gains.

---

## Reward Weight Summary

| Term | Sign | Notes |
|---|---|---|
| Velocity tracking | + | Exponential kernel; primary training signal |
| Body roll | − | Quadratic in roll angle |
| Pitch rate | − | Quadratic in pitch rate |
| Foot height | + | Active only during swing; velocity-gated |
| Joint position | − | Weight increased to suppress wide gait |
| Stable progress | + | Combines forward progress with stability |
| Energy / torque | − | Standard penalties from unitree_rl_lab |
| Contact / slip | − | Standard penalties from unitree_rl_lab |