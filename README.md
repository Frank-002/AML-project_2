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

> **Work in progress.** The commands that produce the results are listed below.
> Step-by-step instructions to install the environment from scratch will be
> added once they have been tested on a clean copy of the repository.

## Project status

| Part | Content | Status |
| --- | --- | --- |
| 1 | Reproducing SplatPose on MAD-Sim and studying its main design choices | Done |
| 2 | Editing the trained splat to create synthetic defects | Not started |
| 3 | Training a small detector on the synthetic defects | Not started |

## Results so far

The commands below are run from the repository root. They assume that the
Python environment is ready (see [Environment](#environment)), that the MAD-Sim
data set is in `MAD-Sim/` and that the pretrained checkpoints are in
`PAD_utils/model/`. Data set and checkpoints are downloaded as explained in
[README_SplatPose.md](README_SplatPose.md).

### Reproduction

```shell
python prepare_lego_ad.py
python run_baseline.py -c 02Unicorn 03Mallard 07Owl 16Cat
```

`prepare_lego_ad.py` converts MAD-Sim into the format read by the Gaussian
Splatting training and is needed only once. `run_baseline.py` trains and
evaluates each object with the default settings (210 training views, 30,000
Gaussian Splatting iterations, 175 pose refinement steps) and seeds 0–4, writes
`results/baseline_results.csv` and prints mean and standard deviation per
object. Runs that are already complete are skipped, so the command can be
interrupted and restarted.

One run of Unicorn takes about 15 minutes on the GPU listed below; objects with
more test images take longer. In addition, the LoFTR matches used for the
initial pose are computed once per object, at about 6 seconds per test image
(roughly half an hour per object).

In the table, our values are the mean over the five runs; the standard
deviation over the five runs is shown for the image-level AUROC.

| Object | Image AUROC, paper | Image AUROC, ours | Pixel AUROC, paper | Pixel AUROC, ours |
| --- | ---: | ---: | ---: | ---: |
| Unicorn | 97.9 | 98.3 ± 0.4 | 99.6 | 99.6 |
| Mallard | 97.4 | 97.6 ± 0.5 | 99.7 | 99.7 |
| Owl | 86.8 | 87.6 ± 0.9 | 99.2 | 99.3 |
| Cat | 82.4 | 81.4 ± 1.1 | 99.3 | 99.1 |
| Mean | 91.1 | 91.2 | 99.5 | 99.4 |

These numbers cannot be reproduced digit by digit: training and pose refinement
are not deterministic on the GPU, so even the same seed gives slightly
different results on each execution. Across the five seeds the image-level
AUROC varies, from the worst to the best run, by 0.9 points for Unicorn, 1.5
for Mallard, 2.5 for Owl and 3.2 for Cat. A new execution should fall within a
similar range.

### Ablation

```shell
python run_ablation.py -c 02Unicorn 03Mallard 07Owl 16Cat
python analyze_ablation.py -c 02Unicorn 03Mallard 07Owl 16Cat -render 1
```

`run_ablation.py` changes one setting at a time on the same four objects, with
a single seed (0):

- training views: 42, 84, 126, 168 (default 210);
- Gaussian Splatting iterations: 3,000, 7,000, 15,000 (default 30,000);
- pose refinement steps: 1, 25, 50, 100, 300 (default 175).

It writes `results/ablation_results_seed0.csv`, one row for each of the 52
evaluations, and took about 9 hours with the LoFTR matches already computed.
Like the baseline, it can be interrupted and restarted.

`analyze_ablation.py` reads the files saved by the grid and writes
`results/ablation_diagnostics_seed0.csv`: for each setting, how many test
images end up aligned with the reference pose and how well the splat
reconstructs the training views (PSNR).

All CSV files are in [`results/`](results).

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
in [`requirements_2.txt`](requirements_2.txt). Besides these packages, the two
CUDA extensions in `submodules/` have to be compiled and installed in the same
environment; the tested installation steps will be added here.

The official `requirements.txt` and `environment.yml` are kept as they were and
describe the authors' original environment, not this one.

## Credits and license

- SplatPose: [official repository](https://github.com/m-kruse98/SplatPose) and
  [paper](https://arxiv.org/abs/2404.06832). Citation, credits and
  acknowledgments of the authors are in [README_SplatPose.md](README_SplatPose.md).
- MAD-Sim data set: [PAD repository](https://github.com/EricLee0224/PAD).
- 3D Gaussian Splatting: [official repository](https://github.com/graphdeco-inria/gaussian-splatting).

This repository is distributed under the same Gaussian-Splatting License as the
official code, for non-commercial research use only: see
[LICENSE.md](LICENSE.md).
