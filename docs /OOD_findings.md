# OOD Findings: 

This document covers the OOD results from the Trakr Baseline in the same OOD tasks as the Unitree Go2 Rough AsymPPO checkpoint.

## Reading Rule

For OOD results, the metrics considered are identical to the metrics used to evaluate the Rough AsymPPO checkpoint:

- `velocity_error`
- `timeout_fraction_of_terminals`
- `base_contact_fraction_of_terminals`
- `bad_orientation_fraction_of_terminals`

Higher timeout fraction is better.
Higher base-contact fraction and bad-orientation fraction is worse.

## Geometry OOD Summary

### Stairs Down

- Trakr AsymPPO:
  - `episodes = 98`
  - `vel_err = 0.1908`
  - `timeout_frac = 0.9479`
  - `base_contact_frac = 0.0000`
  - `bad_orientation_frac = 0.0521`

Visual Comparison:

- Unitree Rough AsymPPO checkpoint can only traverse low step height stairs.
- Even for low step height, the success rate is around 75%
- For greater step heights, the Rough AsymPPO checkpoint cannot handle the high momentum and topples before reaching the bottom
- Trakr AsymPPO checkpoint can handle step heights upto 0.18m with 95% success rate.
- Trakr checkpoint can traverse down the stairs with a stable gait and low momentum.

### Random Rough

- Trakr AsymPPO:
  - `episodes = 111`
  - `vel_err = 0.2030`
  - `timeout_frac = 0.7117`
  - `base_contact_frac = 0.0000`
  - `bad_orientation_frac = 0.2882`

Visual Comparison:

- Both checkpoints perform well on Random Rough terrain.
- Unitree Rough AsymPPO checkpoint is better at recovering when one of its feet gets stuck.

### Boxes

- Trakr AsymPPO:
  - `episodes = 106`
  - `vel_err = 0.2704`
  - `timeout_frac = 0.5094`
  - `base_contact_frac = 0.0188`
  - `bad_orientation_frac = 0.4716`

Visual Comparison:

- Unitree Rough AsymPPO checkpoint can only traverse through very small height obstacles.
- It cannot lift its legs to traverse over small step height obstacles.
- Trakr AsymPPO checkpoint can lift its legs to traverse through obstacles of step height <= 0.18m
- It cannot lift its legs to overcome obstacles of height > 0.18m
- Both checkpoints cannot anticipate gaps in the terain.
- Unitree Rough AsymPPO checkpoint can recover from situations where it has fallen on its base and stand up straight, but the Trakr AsymPPO checkpoint cannot.

### Stairs Up

- Trakr AsymPPO:
  - `episodes = 106`
  - `vel_err = 0.2059`
  - `timeout_frac = 0.9056`
  - `base_contact_frac = 0.0000`
  - `bad_orientation_frac = 0.0943`

Visual Comparison:

- Unitree Rough AsymPPO checkpoint cannot climb up stairs of any step height.
- However, it does recover well once it has fallen, toppled, or has its feet stuck to regain its stand position.
- Trakr AsymPPO checkpoint can climb up stairs of step height upto 0.17m with 90% success rate.
- However it cannot climb up stairs of step height more than 0.18m.
- Also, it cannot recover from positions where its feet is stuck or it has fallen over. 


## Dynamics OOD Summary

### Ultra-High Friction


### Very Heavy

- Trakr AsymPPO:
  - `episodes = 96`
  - `vel_err = 0.6648`
  - `timeout_frac = 1.0000`
  - `base_contact_frac = 0.0000`
  - `bad_orientation_frac = 0.0000`

Visual Comparison:

- Both Unitree Rough AsymPPO and Trakr AsymPPO checkpoints can handle weight additions of 15kgs
- Trakr AsymPPO can handle the weight better with more stable gait, whereas Unitree Rough AsymPPO has more base contact and comparetively unstable gait.
- The behavior might be due to the provided Trakr USD, since the base is higher compared to the legs, leading to lesser torque on the motors, and more stable gait.
- Both checkpoints have high energy usage since the base is low and their is high torque on the knee actuators.
- Velocity Tracking error is higher for both checkpoints as they have to compensate for more weight.

### Ultra-Low Friction


### Very Weak Motors


## Push Recovery OOD Summary

### Yaw Push Medium


### Forward Push Medium

### Lateral Push Medium

### Lateral Push Repeated

### Push Recovery Takeaway

## Switch OOD Summary

### Switch To Ultra-Low Friction

### Switch To Low-Friction + Heavy

### Switch To Very Heavy

### Switch To Very Weak Motors


