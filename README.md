# gym-jsbsim

`gym-jsbsim` is a JSBSim-based reinforcement learning project for fixed-wing flight control and shared-world dogfight training/evaluation.

The repository contains two layers:

- the original single-aircraft gym-jsbsim control tasks
- an extended two-aircraft dogfight stack with self-play training, scenario-based evaluation, and HTML playback dashboards

## Current Status

The active work in this repo is centered on the dogfight pipeline:

- self-play PPO training for `plane_a` and `plane_b`
- shared-world two-plane evaluation
- scenario-library evaluation episodes
- CSV telemetry export
- standalone cockpit playback HTML
- shared-world dashboard HTML

Recent changes made in this repo:

- added JSBSim simulation time to telemetry/export
- made cockpit playback use sim time
- added interpolation/smoothing to cockpit playback
- added deterministic scenario-library evaluation episodes
- added scenario labels to telemetry and dashboards
- added scenario-library sampling for training resets
- removed `target_roll` from dogfight policy observation so RL now learns bank behavior from state/reward instead of a handcrafted desired-bank input
- preserved roll-related telemetry fields for analysis
- updated cockpit/dashboard branding and styling

## Repository Structure

Core environment and task code:

- [gym_jsbsim/environment.py](/content/gym-jsbsim/gym_jsbsim/environment.py)
  Gym-facing single-aircraft environment wrapper.
- [gym_jsbsim/simulation.py](/content/gym-jsbsim/gym_jsbsim/simulation.py)
  JSBSim adapter around `jsbsim.FGFDMExec`.
- [gym_jsbsim/properties.py](/content/gym-jsbsim/gym_jsbsim/properties.py)
  Property definitions used across the project.
- [gym_jsbsim/tasks.py](/content/gym-jsbsim/gym_jsbsim/tasks.py)
  Single-aircraft task logic such as heading/turn control.
- [gym_jsbsim/rewards.py](/content/gym-jsbsim/gym_jsbsim/rewards.py)
  Reward component definitions.
- [gym_jsbsim/assessors.py](/content/gym-jsbsim/gym_jsbsim/assessors.py)
  Reward aggregation/shaping logic.

Shared-world and dogfight code:

- [gym_jsbsim/multi_env.py](/content/gym-jsbsim/gym_jsbsim/multi_env.py)
  Shared-world two-aircraft wrapper.
- [gym_jsbsim/dogfight.py](/content/gym-jsbsim/gym_jsbsim/dogfight.py)
  Core two-plane dogfight environment, reward, observations, telemetry.
- [gym_jsbsim/dogfight_sb3_env.py](/content/gym-jsbsim/gym_jsbsim/dogfight_sb3_env.py)
  Single-agent SB3 wrapper around the two-plane dogfight env.
- [gym_jsbsim/dogfight_policies.py](/content/gym-jsbsim/gym_jsbsim/dogfight_policies.py)
  Opponent policy helpers, including frozen SB3 policy loading.
- [gym_jsbsim/dogfight_scenarios.py](/content/gym-jsbsim/gym_jsbsim/dogfight_scenarios.py)
  Predefined scenario library for deterministic starts.

Realtime training/eval/dashboard tools:

- [gym_jsbsim/realtime/train_dogfight_sb3.py](/content/gym-jsbsim/gym_jsbsim/realtime/train_dogfight_sb3.py)
  Self-play training and eval entry point.
- [gym_jsbsim/realtime/build_dogfight_cockpit.py](/content/gym-jsbsim/gym_jsbsim/realtime/build_dogfight_cockpit.py)
  Builds standalone cockpit playback HTML from eval CSV.

HTML outputs/templates:

- [models/eval_outputs/dogfight_cockpit_merged.html](/content/gym-jsbsim/models/eval_outputs/dogfight_cockpit_merged.html)
  Merged cockpit template used for generated playback files.
- [notebook/eval_shared_world_dashboard.html](/content/gym-jsbsim/notebook/eval_shared_world_dashboard.html)
  Shared-world dashboard for loading combined CSVs.

## Dogfight Architecture

### Shared World

The dogfight system uses two separate JSBSim simulations, one per aircraft, but keeps them in a shared notional world.

Each reset defines:

- common geodetic world origin
- per-plane north/east offsets
- per-plane altitude offsets
- per-plane initial headings

This is managed in [gym_jsbsim/multi_env.py](/content/gym-jsbsim/gym_jsbsim/multi_env.py).

### Dogfight Task

The dogfight environment is implemented in [gym_jsbsim/dogfight.py](/content/gym-jsbsim/gym_jsbsim/dogfight.py).

Main responsibilities:

- set/reset two aircraft
- update opponent-relative target track each step
- compute relative geometry
- build per-plane observations
- compute dogfight reward
- export telemetry rows

The dogfight env uses `PursuitDogfightTask`, which inherits from `TurnHeadingControlTask` for underlying aircraft/task mechanics, but the dogfight reward and observation are built at the outer env level.

## Current Dogfight Observation, Action, Reward

### Action Space

The dogfight policy outputs a 3D continuous control vector:

1. `aileron_cmd`
2. `elevator_cmd`
3. `rudder_cmd`

These come from [gym_jsbsim/tasks.py](/content/gym-jsbsim/gym_jsbsim/tasks.py).

The policy does not output:

- roll directly
- target roll directly
- throttle

Throttle is controlled separately by task initialization.

### Observation Space

The current dogfight policy observation is:

- aircraft physical state
- turn-task state except `target_roll`
- opponent-relative geometry features

The important current design choice is:

- `target_roll_rad` is no longer part of the dogfight policy observation

That means RL must infer bank behavior from physical and relative state, rather than reading a handcrafted desired-bank signal.

Current observation contents:

Own-aircraft state:

- altitude
- pitch
- roll
- body-axis velocities `u, v, w`
- angular rates `p, q, r`
- left/right aileron positions
- elevator position
- rudder position
- altitude error
- sideslip
- track error
- steps left

Opponent-relative features:

- normalized range
- normalized bearing error
- normalized elevation error
- normalized heading difference
- normalized vertical separation
- normalized forward separation

### Reward

Current dogfight reward uses outcome-oriented combat terms:

- aim quality
- elevation quality
- range quality
- closure bonus
- fire-solution bonus
- defensive penalty if threatened by the opponent

Current reward does not include:

- explicit roll-matching reward
- explicit `target_roll` minimization reward

## Target Roll: Current Meaning

`target_roll` still exists inside the inherited turn-task machinery and in telemetry, but it is not fed to the dogfight policy anymore.

Current meaning:

- it is a mathematically computed desired bank angle derived from track error
- it is not something RL directly outputs
- it is not currently part of dogfight reward shaping

This means:

- RL still outputs low-level surface commands
- RL now learns bank direction/magnitude from state and reward instead of an explicit target-roll observation cue

Roll-related telemetry still exported for analysis:

- `target_roll_deg`
- `current_roll_deg`
- `roll_error_deg`

## Training

Training entry point:

- [gym_jsbsim/realtime/train_dogfight_sb3.py](/content/gym-jsbsim/gym_jsbsim/realtime/train_dogfight_sb3.py)

Training mode:

- alternating self-play cycle
- train `plane_b` against frozen `plane_a`
- then train `plane_a` against frozen `plane_b`

SB3 wrapper:

- [gym_jsbsim/dogfight_sb3_env.py](/content/gym-jsbsim/gym_jsbsim/dogfight_sb3_env.py)

Normalization:

- `VecNormalize` is used and saved alongside models

### Scenario-Based Training Resets

Training can now reset from the scenario library instead of only random spawn offsets.

Current training options:

- `--train-scenario-set random`
  Traditional random reset behavior.
- `--train-scenario-set all`
  Sample one predefined scenario on each episode reset.

This is useful for training on varied starting geometries.

### Important Compatibility Note

Because `target_roll` was removed from dogfight policy observations:

- old dogfight checkpoints are not observation-compatible with the new architecture
- old `VecNormalize` statistics are also not compatible

Use a fresh model directory for the new training family.

## Evaluation

Evaluation also uses:

- [gym_jsbsim/realtime/train_dogfight_sb3.py](/content/gym-jsbsim/gym_jsbsim/realtime/train_dogfight_sb3.py)

### Scenario-Library Evaluation

Evaluation can sweep a fixed scenario library.

Current default:

- `--scenario-set all`

That means:

- each scenario becomes a different evaluation episode
- `scenario_name` is written into the CSV
- dashboards display the scenario name

Current scenario count:

- `18`

If `--episodes 1` is used with `--scenario-set all`, total episode count is `18`.

If `--episodes 2` is used, total episode count is `36`.

Random eval is still possible with:

- `--scenario-set random`

## Scenario Library

Scenario definitions are in:

- [gym_jsbsim/dogfight_scenarios.py](/content/gym-jsbsim/gym_jsbsim/dogfight_scenarios.py)

Examples included:

- head-on merge
- offset head-on merge
- plane A behind plane B
- plane B behind plane A
- crossing left/right
- line abreast
- vertical stack
- oblique tail chase
- descending merge

These scenarios are used for:

- deterministic evaluation episodes
- optional training-reset sampling

## Telemetry and Outputs

The dogfight env exports telemetry rows through [gym_jsbsim/dogfight.py](/content/gym-jsbsim/gym_jsbsim/dogfight.py), built on [gym_jsbsim/multi_env.py](/content/gym-jsbsim/gym_jsbsim/multi_env.py).

Important telemetry fields include:

- `plane_id`
- `episode`
- `step`
- `reward`
- `done`
- `scenario_name`
- `simulation_sim_time_sec`
- state/attitude/velocity properties
- relative geometry values
- roll diagnostics

### Sim Time

Telemetry now includes native JSBSim simulation time:

- `simulation_sim_time_sec`

Cockpit playback uses this sim time instead of relying only on `step * dt`.

## Dashboards and HTML Playback

### Cockpit Playback

Built by:

- [gym_jsbsim/realtime/build_dogfight_cockpit.py](/content/gym-jsbsim/gym_jsbsim/realtime/build_dogfight_cockpit.py)

Output:

- standalone HTML with embedded telemetry payload
- automatic playback on open
- scenario label in status line
- interpolation between samples for smoother playback

Merged cockpit template:

- [models/eval_outputs/dogfight_cockpit_merged.html](/content/gym-jsbsim/models/eval_outputs/dogfight_cockpit_merged.html)

### Shared Dashboard

Interactive dashboard:

- [notebook/eval_shared_world_dashboard.html](/content/gym-jsbsim/notebook/eval_shared_world_dashboard.html)

Features:

- load combined CSV
- view episodes
- see scenario names
- inspect trajectory, reward, altitude, heading, radar-like panels

## Branding / Styling

The HTML assets were updated to use:

- Reflex Blue: `#263685`
- Red 485 C: `#DD140E`

Brand text:

- `TURKISH`
- `AEROSPACE`

Font preference used in headers:

- first line: `NeoSans Pro Black Italic`
- second line: `NeoSans Pro Medium Italic`

Logo usage:

- both cockpit/dashboard HTMLs assume `logo.png` exists in the same directory as the HTML file

Examples:

- `/content/gym-jsbsim/models/eval_outputs/logo.png`
- `/content/gym-jsbsim/notebook/logo.png`

## Typical Commands

### Train New Dogfight Family With Scenario-Sampled Resets

Example for `dogfight6`:

```bash
python3 -m gym_jsbsim.realtime.train_dogfight_sb3 \
  --mode train_cycle \
  --rounds 3 \
  --timesteps 200000 \
  --seed 0 \
  --train-scenario-set all \
  --model-a-dir /content/gym-jsbsim/models/dogfight6/plane_a \
  --model-b-dir /content/gym-jsbsim/models/dogfight6/plane_b \
  --model-a-path /content/gym-jsbsim/models/dogfight6/plane_a/ppo_dogfight_plane_a.zip \
  --vecnorm-a-path /content/gym-jsbsim/models/dogfight6/plane_a/vecnormalize.pkl \
  --model-b-path /content/gym-jsbsim/models/dogfight6/plane_b/ppo_dogfight_plane_b.zip \
  --vecnorm-b-path /content/gym-jsbsim/models/dogfight6/plane_b/vecnormalize.pkl
```

### Evaluate Across All Scenarios

```bash
python3 -m gym_jsbsim.realtime.train_dogfight_sb3 \
  --mode eval \
  --episodes 1 \
  --max-steps 500 \
  --model-a-path /content/gym-jsbsim/models/dogfight6/plane_a/ppo_dogfight_plane_a.zip \
  --vecnorm-a-path /content/gym-jsbsim/models/dogfight6/plane_a/vecnormalize.pkl \
  --model-b-path /content/gym-jsbsim/models/dogfight6/plane_b/ppo_dogfight_plane_b.zip \
  --vecnorm-b-path /content/gym-jsbsim/models/dogfight6/plane_b/vecnormalize.pkl \
  --csv-path /content/gym-jsbsim/models/eval_outputs/dogfight6_scenarios_eval.csv
```

### Build Merged Cockpit HTML

```bash
python3 -m gym_jsbsim.realtime.build_dogfight_cockpit \
  --csv-path /content/gym-jsbsim/models/eval_outputs/dogfight6_scenarios_eval.csv \
  --output-html /content/gym-jsbsim/models/eval_outputs/dogfight6_cockpit_merged.html \
  --template merged
```

## Validation

Targeted dogfight tests:

```bash
python -m pytest gym_jsbsim/tests/test_dogfight.py
```

This test file currently covers:

- relative geometry basics
- telemetry fieldnames
- dogfight shot-quality helpers
- scenario catalog behavior
- scenario reset control
- exclusion of `target_roll` from dogfight policy state variables

## Notes

- The original single-aircraft gym-jsbsim code is still present and usable.
- The dogfight stack is the actively customized part of this repository.
- The current dogfight architecture intentionally separates:
  - learned low-level control outputs
  - engineered reward terms
  - analysis-only telemetry signals

