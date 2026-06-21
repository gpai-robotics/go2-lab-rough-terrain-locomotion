# Task Setup

This document covers the Isaac Lab task definitions for Trakr flat and rough terrain locomotion.

---

## 1. Extension and Task Registration

Tasks are exposed to Isaac Lab via Gymnasium registration. The flat task is registered with id `Trakr-Velocity`, the rough task with `Trakr-Velocity-Rough`, and the stair traversal task with `Trakr-Velocity-Stairs`.

### Task Registration

```python
# source/trakr_rl/trakr_rl/tasks/locomotion/robots/[TASK_NAME]/__init__.py
gym.register(
    id="[TASK-NAME]",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "BasePPORunnerCfg"
        ),
    },
)
```

## 2. Scene Configuration

The scene replaces the Unitree Go2 asset with `TRAKR_CFG` and updates all prim/body name references after inspecting the USD stage in Isaac Sim.

| Property | Value |
|---|---|
| Robot prim path | `ENV_REGEX_NS/trakr` |
| Base body name | `base_link` |
| Contact sensor path | `ENV_REGEX_NS/trakr/trakr/.*` |
| Training environments | 512 |
| Evaluation environments | 32 |

### Terrain (Flat)

The flat task uses a single plane sub-terrain. This lets the policy focus on velocity tracking before progressing to rough terrain.

| Property | Value |
|---|---|
| Tile size | 8 m × 8 m |
| Friction | 1.0 |
| Visual material | Isaac Lab Nucleus marble |

---

## 3. MDP Design

### Action Space

Joint position targets for all 12 Trakr joints. Actions are scaled by **0.25** and applied as offsets from the nominal standing posture.

### Observation Space (Policy)

| Observation | Description |
|---|---|
| Base angular velocity | Body-frame ω |
| Projected gravity | Gravity vector in body frame |
| Commanded velocity | Linear x/y + yaw rate |
| Relative joint positions | `q − q_default` |
| Relative joint velocities | `dq` |
| Previous action | Action from the last timestep |

- History length: **5 steps**
- Observation noise corruption: **enabled** (robustness to sensor noise)

### Critic Observations (Privileged)

In addition to policy observations, the critic also receives:
- Base linear velocity
- Joint effort measurements

### Velocity Commands

Generated via `UniformLevelVelocityCommandCfg`, which supports curriculum expansion of the command distribution.

| Stage | Forward (m/s) | Lateral (m/s) | Yaw (rad/s) |
|---|---|---|---|
| Initial | [-0.1, 0.1] | [-0.1, 0.1] | [-1.0, 1.0] |
| Final | [-1.0, 1.0] | [-0.4, 0.4] | [-1.0, 1.0] |

---

## 4. Events

Randomization is applied during startup and resets to improve robustness.

| Event | Parameter | Range |
|---|---|---|
| Startup | Rigid body friction | [0.3, 1.2] |
| Startup | Restitution | [0.0, 0.15] |
| Startup | Base mass perturbation | [-1.0, 3.0] kg |
| Reset | Base pose | Randomized |
| Reset | Yaw angle | Randomized |
| Reset | Joint velocities | Randomized |
| Periodic | Horizontal base velocity push | Applied |

For the Rough Terrain Training Setup, the following randomization is added,
| Event | Parameter | Range |
|---|---|---|
| Startup | Center of Mass | x = [-0.05, 0.05], y = [-0.05, 0.05], z = [-0.02, 0.02] | 


---

## 5. Termination Conditions

Episodes end on any of the following:

- **Timeout** — episode length exceeded
- **Base contact** — `base_link` contacts the ground
- **Bad orientation** — body tilt exceeds **0.8 rad**

---

## 6. Rough Terrain and Stair Traversal Configuration

### Rough-Terrain Mix

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough":     HfRandomUniformTerrainCfg(proportion=0.1, noise_range=(0.05, 0.12), ...),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.5, 0.6), ...),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(proportion=0.1, slope_range=(0.5, 0.6), ...),
    },
)
```

| Sub-terrain | Proportion | Key Parameter |
|---|---|---|
| Random rough | 10% | Height noise 0.05–0.12 m |
| Pyramid slope | 10% | Slope 0.5–0.6 |
| Inverted Pyramid slope | 10% | Slope 0.5-0.6 |


### Stair Traversal Mix

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pyramid_stairs":   MeshPyramidStairsTerrainCfg(proportion=0.2, step_height_range=(0.08, 0.15), ...),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(proportion=0.2, step_height_range=(0.08, 0.15), ...),
    },
)
```

| Sub-terrain | Proportion | Key Parameter |
|---|---|---|
| Pyramid stairs | 20% | Step height 0.08–0.15 m |
| Inverted pyramid stairs | 20% | Step height 0.08–0.15 m |

### Terrain Curriculum

Both tasks use Isaac Lab's terrain curriculum mechanism. Agents are progressively moved to harder terrain patches as performance improves.

```python
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
```
