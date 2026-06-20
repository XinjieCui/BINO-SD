# BINO-SD

Official source code for the ICANN 2026 paper:

**Beyond Smooth Darcy: Evaluation Changes the Conclusion in Sparse-Sensor Inverse Learning**

This repository contains the reproducible experiment code for the paper, including
Darcy benchmark generation, sparse-sensor inverse learning, evaluation scripts, and
paper-table utilities.

This project is a minimal landing implementation for the first item in `PDE.txt`:

`inverse problem + experimental design + Bayesian neural operator`

The code instantiates a small but complete pipeline:

- generate a synthetic Darcy / diffusion inverse dataset locally;
- scale to larger and more realistic benchmark families;
- learn a differentiable sparse sensor layout;
- infer a posterior mean and log-variance for the hidden coefficient field;
- regularize the inverse model with a PDE residual term.

## Project layout

- `scripts/generate_dataset.py`: builds a synthetic Darcy inverse dataset.
- `scripts/generate_benchmark_suite.py`: generates larger benchmark-suite datasets.
- `scripts/run_benchmark_preset.py`: launches named large-benchmark train/eval presets.
- `scripts/train.py`: trains the Bayesian inverse operator.
- `scripts/evaluate.py`: evaluates a checkpoint and writes plots and metrics.
- `scripts/aggregate_results.py`: aggregates multiple runs into markdown and JSON summaries.
- `scripts/make_paper_tables.py`: generates paper-style markdown and LaTeX tables.
- `scripts/make_method_figure.py`: draws a method overview figure for reports or drafts.
- `scripts/make_dataset_figure.py`: renders representative coefficient/solution/forcing examples from a dataset.
- `scripts/setup_env.ps1`: creates the conda environment and installs dependencies.
- `src/bi_operator/pde.py`: PDE sampler, solver, and physics residual.
- `src/bi_operator/model.py`: soft sensor layer and FNO-style inverse model.
- `src/bi_operator/data.py`: dataset loading helpers.

## Repro steps

```powershell
cd <repo>
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1

conda run -n pde_inverse_operator python .\scripts\generate_dataset.py --output .\data\darcy_inverse_dataset.npz
conda run -n pde_inverse_operator python .\scripts\train.py --dataset .\data\darcy_inverse_dataset.npz --output-dir .\artifacts\baseline_run
conda run -n pde_inverse_operator python .\scripts\evaluate.py --dataset .\data\darcy_inverse_dataset.npz --checkpoint .\artifacts\baseline_run\best_model.pth --output-dir .\artifacts\baseline_run
```

## GPU Setup

If you want to use the RTX 5090 on this machine, a separate GPU environment is available:

```powershell
cd <repo>
powershell -ExecutionPolicy Bypass -File .\scripts\setup_gpu_env.ps1
```

This creates `pde_inverse_operator_gpu` and installs a CUDA 12.8 PyTorch build without disturbing the CPU environment.

## Notes

- The dataset is generated locally, so there is no external download dependency.
- The model predicts a Gaussian posterior over the log-coefficient field.
- Sensor design is differentiable: the sensor maps are trainable and sharpened via entropy and diversity regularization.
- The current implementation is a research prototype meant to be runnable end to end, not a final large-scale benchmark.
- The newer `darcy_structured_realistic` benchmark family adds signed multisource forcing,
  nonzero Dirichlet boundaries, and optional noisy interior observations.
- Large generated datasets, checkpoints, plots, and run artifacts are intentionally excluded from Git.
  They can be regenerated with the scripts above, or restored locally under `data/` and `artifacts/`.

## Latest Best Runs

- Current paper-track best single run on the main 20x20 constant-forcing task:
  `artifacts/abl_sensors24`
  test NLL about `-0.4921`
  test RMSE about `0.4632`
- Current paper-track best stable 20x20 constant-forcing result:
  `artifacts/multiseed_summary_24s.md`
  RMSE mean about `0.4641`, std about `0.0027`
  NLL mean about `-0.4904`, std about `0.0080`
- Current paper-track best 32x32 constant-forcing run:
  `artifacts/task_constant32_lgr`
  test NLL about `-0.5907`
  test RMSE about `0.4322`
- Current best learned-sensor hard-task 32x32 multisource-forcing run:
  `artifacts/task_multisource32_big40`
  test NLL about `-0.3363`
  test RMSE about `0.5164`
- Current best matched-budget random-grid hard-task run:
  `artifacts/task_multisource32_random40`
  test NLL about `-0.3849`
  test RMSE about `0.5041`
- Current hard-task matched-40 layout summary:
  `artifacts/multisource32_layout_matched40_summary.md`
- Current hard-task random40 multi-seed summary:
  `artifacts/multiseed_summary_multisource32_random40.md`
  RMSE mean about `0.5247`, std about `0.0175`
  NLL mean about `-0.3286`, std about `0.0514`
- Current larger structured benchmark run:
  `artifacts/task_structured48_big40_v2`
  test NLL about `-0.1185`
  test RMSE about `0.6704`
- Current realistic scale-up dataset:
  `data/darcy_structured_realistic_64_medium.npz`
  `64x64`, `1024/160/160`, interior observation noise `0.03`
- Current realistic 64x64 best single run:
  `artifacts/task_structured_realistic64_main64s`
  test NLL about `1.2622`
  test RMSE about `1.2250`
- Current realistic 64x64 multi-seed summary:
  `artifacts/multiseed_summary_realistic64_64s.md`
  RMSE mean about `1.2251`, std about `0.0005`
- Current realistic 80x80 scale-up run:
  `artifacts/task_structured_realistic80_main`
  test NLL about `1.4239`
  test RMSE about `1.2633`
- Current realistic 80x80 multi-seed summary:
  `artifacts/multiseed_summary_realistic80_56s.md`
  RMSE mean about `1.2647`, std about `0.0012`
- Current realistic 96x96 first run:
  `artifacts/task_structured_realistic96_main`
  test NLL about `1.6389`
  test RMSE about `1.2970`
- Current realistic 96x96 multi-seed summary:
  `artifacts/multiseed_summary_realistic96_72s.md`
  RMSE mean about `1.2962`, std about `0.0006`
- Current realistic 96x96 baseline summary:
  `artifacts/realistic96_baseline_summary.md`
- Main-task large dataset file:
  `data/darcy_inverse_dataset_large.npz`
- Structured larger-scale dataset file:
  `data/darcy_structured_multisource_48_large.npz`
- Realistic benchmark scale-up summary:
  `artifacts/benchmark_realism_scale_update_2026-03-22.md`
- Benchmark suite manifest:
  `artifacts/benchmark_suite_conference_scale.md`

## Paper-Track Results

- Baseline comparison:
  `artifacts/baseline_summary.md`
- Ablation study:
  `artifacts/ablation_summary.md`
- Multi-seed summary for the 16-sensor main config:
  `artifacts/multiseed_summary.md`
- Multi-seed summary for the 24-sensor main config:
  `artifacts/multiseed_summary_24s.md`
- Multi-seed summary for the 32x32 multisource big-40 config:
  `artifacts/multiseed_summary_multisource32_big40.md`
- Matched 40-sensor layout comparison on the 32x32 multisource task:
  `artifacts/multisource32_layout_matched40_summary.md`
- Multi-seed summary for the 32x32 multisource random-40 config:
  `artifacts/multiseed_summary_multisource32_random40.md`
- Task and scale summary:
  `artifacts/taskscale_summary.md`
- Multisource 32x32 tuning summary:
  `artifacts/multisource_tuning_summary.md`
- Structured 48x48 baseline summary:
  `artifacts/structured48_baseline_summary.md`
- Realistic 64x64 baseline summary:
  `artifacts/realistic64_baseline_summary.md`
- Realistic 64x64 multi-seed summary:
  `artifacts/multiseed_summary_realistic64_64s.md`
- Realistic 80x80 baseline summary:
  `artifacts/realistic80_baseline_summary.md`
- Realistic 80x80 multi-seed summary:
  `artifacts/multiseed_summary_realistic80_56s.md`
- Realistic 96x96 baseline summary:
  `artifacts/realistic96_baseline_summary.md`
- Realistic 96x96 multi-seed summary:
  `artifacts/multiseed_summary_realistic96_72s.md`
- Paper-style markdown / LaTeX tables:
  `artifacts/paper_tables_2026-03-22.md`
  `artifacts/paper_tables_2026-03-22.tex`
- Method overview figure:
  `artifacts/method_overview.png`
- Structured 48x48 dataset example:
  `artifacts/structured48_dataset_example.png`
- Paper positioning memo:
  `artifacts/PAPER_POSITIONING_2026-03-22.md`

Current key takeaways:

- On the main large 20x20 constant-forcing task, the best trainable sensor design remains
  `learned_gaussian_random`, and increasing the sensor count from `16` to `24`
  improves the single-run RMSE from about `0.4684` to about `0.4632`.
- The 24-sensor main-task result is now backed by a three-seed summary:
  RMSE mean about `0.4641`, std about `0.0027`
- The previous weakest setting, the 32x32 multisource-forcing task, is now materially better:
  RMSE improves from about `0.5859` to about `0.5164` when moving to a larger
  learned sensor budget and a stronger operator backbone.
- The new matched-budget `40`-sensor controls show that the hard task is also
  much more layout-sensitive than the smooth main task:
  `random40` reaches about `0.5041` RMSE in the reference seed, while
  `fixed40` reaches about `0.5290`.
- The corresponding hard-task three-seed summaries are now close enough to change
  the paper story:
  `learned40` is about `0.5285 +/- 0.0107` RMSE, while `random40` is about
  `0.5247 +/- 0.0175`.
- This makes the current strongest paper angle benchmark-centered rather than
  method-centered:
  the hard benchmark exposes layout sensitivity and ranking instability,
  while the realistic scale-up line exposes calibration degradation.
- The project now also includes a larger `48x48 structured multisource` benchmark
  to move beyond purely smooth Fourier-style coefficient fields.
- The project now also includes a `64x64 structured realistic` benchmark with
  signed forcing, nonzero boundaries, and noisy observations to increase
  conference-style workload and realism.
- On the new `64x64 structured realistic` benchmark, the best Bayesian learned-sensor
  configuration uses `64` sensors and reaches about `1.2251 +/- 0.0005` RMSE over
  three seeds.
- A first `80x80 structured realistic` run is now available, so the project has
  moved beyond a single scale-up point at `64x64`.
- The `80x80 structured realistic` benchmark now also has baseline comparisons and a
  three-seed summary, which makes the realistic scale-up line much more substantial.
- The `96x96 structured realistic` benchmark now also has random-grid and fixed-grid
  Bayesian baselines, plus a three-seed summary, so it is no longer only a
  scale-showing point.
- The current `96x96` three-seed summary is about `1.2962 +/- 0.0006` RMSE, while
  the deterministic operator remains slightly best in pure RMSE at about `1.2959`.
- On that `48x48 structured` benchmark, the current Bayesian learned-sensor model
  is slightly better than both the `random_grid` Bayesian baseline and the
  deterministic operator baseline.

## Experimental Branches

- The stable main line is still the learned Gaussian-random sensor design without
  forcing-sensor fusion.
- An exploratory branch is now implemented via:
  `--use-forcing-sensor-fusion`
  This is meant for future work on harder multisource settings, but the first run
  was not yet better than the current non-fusion best model.
