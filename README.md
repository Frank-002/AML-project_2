# Pose-Agnostic Anomaly Detection with Gaussian Splatting

A project built on top of
[SplatPose](https://github.com/m-kruse98/SplatPose), the method introduced in
["SplatPose & Detect: Pose-Agnostic 3D Anomaly Detection"](https://arxiv.org/abs/2404.06832)
(Kruse et al., CVPR 2024 Workshops).

This repository is **not** the official SplatPose code. It starts from the
official repository at commit `2c07b6d` and adds what was needed to run it,
measure it and study it. The method and the original code belong to its
authors: their README is kept unchanged in
[README_SplatPose.md](README_SplatPose.md).

> **Work in progress.** Step-by-step instructions to set up the environment and
> reproduce every number from scratch will be added once they have been tested
> on a clean copy of the repository.

## Project status

| Part | Content | Status |
| --- | --- | --- |
| 1 | Reproducing SplatPose on MAD-Sim and studying its main design choices | Done |
| 2 | Editing the trained splat to create synthetic defects | Not started |
| 3 | Training a small detector on the synthetic defects | Not started |

## Results so far

Reproduction with the default settings (210 training views, 30,000 Gaussian
Splatting iterations, 175 pose refinement steps) on four MAD-Sim objects. Our
values are the mean over five runs with seeds 0–4; the standard deviation over
the five runs is shown for the image-level AUROC.

| Object | Image AUROC, paper | Image AUROC, ours | Pixel AUROC, paper | Pixel AUROC, ours |
| --- | ---: | ---: | ---: | ---: |
| Unicorn | 97.9 | 98.3 ± 0.4 | 99.6 | 99.6 |
| Mallard | 97.4 | 97.6 ± 0.5 | 99.7 | 99.7 |
| Owl | 86.8 | 87.6 ± 0.9 | 99.2 | 99.3 |
| Cat | 82.4 | 81.4 ± 1.1 | 99.3 | 99.1 |
| Mean | 91.1 | 91.2 | 99.5 | 99.4 |

The ablation study changes one setting at a time on the same four objects,
with a single seed: number of training views (42 to 210), Gaussian Splatting
iterations (3,000 to 30,000) and pose refinement steps (1 to 300).

The numbers of every run are in [`results/`](results):

- `baseline_results.csv`: the 20 reproduction runs behind the table above;
- `ablation_results_seed0.csv`: the 52 evaluations of the ablation study;
- `ablation_diagnostics_seed0.csv`: pose alignment and reconstruction quality
  for each ablation setting.

## What was added to the official code

| File | Purpose |
| --- | --- |
| `run_baseline.py` | Runs the default configuration over several objects and seeds and collects the metrics |
| `run_ablation.py` | Runs the ablation grid and collects the metrics |
| `analyze_ablation.py` | Measures pose alignment and reconstruction quality on the ablation runs |
| `prepare_view_subsets.py` | Builds the data folders with fewer training views |
| `coarse_pose.py` | Computes the LoFTR matches used for the initial pose once per object and stores them |
| `evaluation_export.py` | Saves scores, metrics, reference renderings and anomaly maps of each evaluation |
| `project_paths.py` | Resolves default paths from the repository root |
| `requirements_2.txt` | Package versions of the environment used for all the results |

`train_and_render.py` and `pose_estimation.py` were extended with the options
these scripts rely on; with their default arguments they run the same
computation as the official code. The comments in the added code are written in
Italian.

## Environment

All results were produced on Windows 11 with an NVIDIA RTX 4070 SUPER (12 GB),
Python 3.10, PyTorch 2.0.1 and CUDA 11.8. The exact package versions are listed
in [`requirements_2.txt`](requirements_2.txt). The official `requirements.txt`
and `environment.yml` are kept as they were and describe the authors' original
environment, not this one.

## Credits and license

- SplatPose: [official repository](https://github.com/m-kruse98/SplatPose) and
  [paper](https://arxiv.org/abs/2404.06832). Citation, credits and
  acknowledgments of the authors are in [README_SplatPose.md](README_SplatPose.md).
- MAD-Sim data set: [PAD repository](https://github.com/EricLee0224/PAD).
- 3D Gaussian Splatting: [official repository](https://github.com/graphdeco-inria/gaussian-splatting).

This repository is distributed under the same Gaussian-Splatting License as the
official code, for non-commercial research use only: see
[LICENSE.md](LICENSE.md).
