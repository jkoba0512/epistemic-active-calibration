# Epistemic Active Calibration

Continuous-time active inference experiments for robot self-calibration under
observation degeneracy.

This repository studies a minimal question:

> Can a robot use uncertainty about its own body parameters to choose actions
> that make those parameters identifiable?

The current proof of concept uses a planar 2-DoF arm with unknown link lengths.
In a deliberately degenerate 1D observation setting, task/VFE-only control can
stay perfectly still because the current observation already matches the goal.
That behavior leaves the Fisher information matrix rank-deficient, so the two
link lengths cannot be identified individually. Adding a parameter epistemic
term to the A-step makes the controller move into informative configurations,
improving calibration and the later 2D reaching task.

## Current Claim

The project currently supports this bounded claim:

> In a 2-DoF link-length identification problem with 1D end-effector
> observations, feeding the E-step posterior precision back into an epistemic
> A-step can actively break an observation degeneracy, improve parameter
> identifiability, and reduce downstream task error.

This is a simulation proof of concept. It does not yet claim general validity
for high-DoF robots, real hardware, contact-rich settings, strong model
mismatch, or superiority over classical optimal experimental design.

## Core Idea

The E-step estimates parameters and an approximate posterior precision:

```text
P_theta = P_prior + J_theta.T @ R_inv @ J_theta
```

For a candidate action, a short rollout predicts future observations and their
parameter sensitivity:

```text
FIM_future(a) = J_future(a).T @ R_inv @ J_future(a)
IG_theta(a)  = 0.5 * (logdet(P_theta + FIM_future(a)) - logdet(P_theta))
```

The A-step then optimizes a task objective with an epistemic bonus:

```text
J_action(a) = F_task(a)
            + alpha * action_energy(a)
            + beta  * safety_or_limit_penalty(a)
            - lambda_eff * IG_theta(a)
```

The important causal chain in the main experiment is:

```text
degenerate 1D observation
  -> VFE-only remains at q2 = 0
  -> FIM stays rank-deficient
  -> l1 and l2 are not individually calibrated
  -> later 2D reaching task fails

parameter epistemic A-step
  -> moves q2 away from zero
  -> FIM becomes informative
  -> l1 and l2 are calibrated
  -> later 2D reaching task improves
```

## Repository Layout

```text
src/dem/
  core.py          Generalized-coordinate utilities
  model.py         DEM model definitions
  inference.py     D-step / VFE inference utilities
  estep.py         E-step parameter inference and precision tracking
  action.py        Basic ADEM action update

experiments/
  phase0_foundation.py            Differentiable rollout, FIM, IG checks
  identifiability_stress_test.py  1D observation rank-deficiency stress test
  dual_control_1d_obs.py          Main two-phase dual-control experiment
  lambda_sweep.py                 Fixed/adaptive epistemic weight sweep
  epistemic_calibration_2dof.py   Earlier 2-DoF calibration experiment
  dual_control_2dof.py            Earlier dual-control experiment
  so101/                          Optional MuJoCo/SO-101 extension scripts

results/
  lambda_sweep.json               Summary metrics for lambda sweep
  sensitivity_2dof_arm.csv        Existing SO-101/MuJoCo sensitivity sweep data
  *.png                           Generated experiment figures

plan/
  research-plan-20260417.md       Research plan and phased roadmap

tutorial/
  epistemic_active_calibration_tutorial.typ
  epistemic_active_calibration_tutorial.pdf
```

## Setup

The project uses Python 3.12 and `uv`.

```bash
uv sync --group dev
```

Run the unit tests:

```bash
uv run pytest -q
```

Expected current result:

```text
50 passed
```

If old virtual-environment entrypoints point to another project, regenerate the
environment:

```bash
uv sync --group dev --reinstall
uv run pytest -q
```

## Optional SO-101 / MuJoCo Experiments

The SO-101/MuJoCo scripts are preserved as future hardware-facing research
assets. They are not part of the core proof-of-concept reproduction path.

Install their optional dependencies with:

```bash
uv sync --group dev --group so101
```

Available commands:

```bash
uv run --group so101 python experiments/so101/dem_mujoco_2dof_arm.py
uv run --group so101 python experiments/so101/param_id_2dof_arm.py
uv run --group so101 python experiments/so101/sensitivity_2dof_arm.py
```

Outputs:

```text
results/dem_mujoco_2dof_arm.png
results/param_id_2dof_arm.png
results/sensitivity_2dof_arm.png
results/sensitivity_2dof_arm.csv
```

Use these scripts when extending the project toward higher-fidelity MuJoCo
simulation or real SO-101 calibration. Keep the central README reproduction
commands above as the default benchmark until the hardware pipeline is tested.

## Reproducing the Main Results

All commands below should be run from the repository root.

### 1. Foundation Checks

Verifies that the short-horizon rollout is JAX-differentiable, that
`dy_future/dtheta` has the expected shape, that FIM/IG are finite, and that
information gain is differentiable with respect to action.

```bash
uv run python experiments/phase0_foundation.py
```

Expected output:

```text
Phase 0: Foundation Tests
[PASS] ...
```

### 2. Identifiability Stress Test

Compares random, sinusoidal, and epistemic exploration with only 1D
end-effector x-position observations. This isolates the rank-deficiency problem:
when the arm stays near `q2 = 0`, `l1` and `l2` are locally hard to identify
separately.

```bash
uv run python experiments/identifiability_stress_test.py
```

Output:

```text
results/identifiability_stress_test.png
```

Use this figure to inspect:

- parameter RMSE
- posterior standard deviations
- condition number of `P_theta`
- mean absolute `q2` exploration

### 3. Main Dual-Control Experiment

Runs the current central demonstration.

Protocol:

- Phase 1, steps 0-49: 1D calibration using `x_ee` only.
- Phase 2, steps 50-99: 2D reaching task using the calibrated parameter estimate.
- Degenerate initial condition:
  - `q0 = [pi/3, 0]`
  - `theta_true = [0.5, 0.5]`
  - `theta_init = [0.9, 0.1]`
  - `l1_init + l2_init = l1_true + l2_true = 1.0`

At the start, the VFE-only controller sees zero task error and does not move.
The epistemic controller moves the elbow to break the degeneracy.

```bash
uv run python experiments/dual_control_1d_obs.py
```

Output:

```text
results/dual_control_1d_obs.png
results/dual_control_1d_obs_summary.json
```

Expected qualitative result:

- `vfe_only`: high parameter RMSE at step 50 and high final 2D task error.
- `random`: better than VFE-only because it excites the elbow by chance, but
  worse than epistemic dual-control in calibration and downstream task error.
- `dual_weak`, `dual_strong`, `dual_adaptive`: low parameter RMSE and much lower
  final 2D task error.

Representative medians from the current result set:

```text
vfe_only     RMSE@50 ~= 0.268   TaskErr@100 ~= 0.135 m   EnergyPh1 ~= 0.000
random       RMSE@50 ~= 0.041   TaskErr@100 ~= 0.041 m   EnergyPh1 ~= 0.415
dual_weak    RMSE@50 ~= 0.006   TaskErr@100 ~= 0.014 m   EnergyPh1 ~= 0.668
dual_strong  RMSE@50 ~= 0.007   TaskErr@100 ~= 0.015 m   EnergyPh1 ~= 0.661
adaptive     RMSE@50 ~= 0.008   TaskErr@100 ~= 0.012 m   EnergyPh1 ~= 0.350
```

### 4. Lambda Sweep

Sweeps the epistemic weight over fixed values plus an adaptive condition.

```bash
uv run python experiments/lambda_sweep.py
```

Outputs:

```text
results/lambda_sweep.png
results/lambda_sweep.json
```

Current summary in `results/lambda_sweep.json`:

| Condition | RMSE@50 median | Final 2D task error median | Phase-1 RMSE AUC median |
|---|---:|---:|---:|
| lambda = 0.0 | 0.313 | 0.139 | 0.355 |
| lambda = 0.1 | 0.007 | 0.009 | 0.109 |
| lambda = 0.5 | 0.006 | 0.013 | 0.104 |
| lambda = 1.0 | 0.006 | 0.013 | 0.103 |
| lambda = 3.0 | 0.008 | 0.013 | 0.102 |
| lambda = 10.0 | 0.008 | 0.016 | 0.100 |
| adaptive | 0.008 | 0.015 | 0.105 |

Interpretation:

- The large change is between `lambda = 0` and any positive `lambda`.
- The range `lambda = 0.1` to `5.0` is relatively insensitive.
- Very large `lambda` can slightly hurt the later task because the arm starts
  Phase 2 from a more exploratory posture.

## Suggested Reproduction Order

For a fresh run:

```bash
uv sync --group dev
uv run pytest -q
uv run python experiments/phase0_foundation.py
uv run python experiments/identifiability_stress_test.py
uv run python experiments/dual_control_1d_obs.py
uv run python experiments/lambda_sweep.py
```

## Current Development Priorities

The three-reviewer internal discussion converged on this near-term plan:

1. Keep the claim as a 2-DoF proof of concept.
2. Make the repository reproducible: README, commands, seeds, outputs, figures.
3. Add a random exploration baseline to the main dual-control comparison.
4. Add fixed/scripted exploration if it remains lightweight.
5. Strengthen tests around `EStep.compute_precision()`.
6. Report FIM/precision indicators alongside RMSE and task error.
7. Draft a short workshop-style paper section before expanding to hardware.

## Known Limitations

- The current central result is low-dimensional and simulated.
- The observation degeneracy is deliberately constructed to expose the mechanism.
- The current main comparison is strongest against VFE-only; more classical OED
  and FIM-greedy baselines are still future work.
- `P_theta` is a local Laplace / Gauss-Newton precision approximation, not a
  guaranteed globally accurate posterior covariance.
- Real robot safety constraints, latency, friction, calibration targets, and
  sensor synchronization are not yet handled.

## Related Documents

- `plan/research-plan-20260417.md`
- `tutorial/epistemic_active_calibration_tutorial.typ`
- `tutorial/epistemic_active_calibration_tutorial.pdf`
- `survey/literature_survey_codex.md`
- `survey/discussion_1_codex.md`
