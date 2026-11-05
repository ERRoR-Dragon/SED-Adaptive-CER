# SED Adaptive CER

Group 6 · Shashank Sadashiv, Arunabh Singh, Maniar Sheil Dipenkumar, and Samarth Bhatnagar

We add a small learned budget predictor to [SED](https://github.com/xb534/SED), an open-vocabulary semantic segmenter. At each decoder stage, SED keeps the union of the top k class candidates at every spatial location. Our predictor reads the first stage's class scores and chooses k=4, 8, 16, 32, or the dense decoder for the whole image. Only the 517-parameter predictor is trained for this project. We reuse the released ConvNeXt-B weights for the encoder and segmentation decoder.

## Run on Windows

Use PowerShell with Python 3.13, Git, and a CUDA-capable NVIDIA GPU. This project was built for the 6 GB RTX 4050 Laptop GPU in an HP Victus. The default run uses one image at a time and FP16 autocast.

On the Victus with the runtime already staged:

~~~powershell
.\scripts\run.ps1
~~~

For a fresh checkout:

~~~powershell
.\scripts\setup.ps1
.\scripts\run.ps1
~~~

Setup creates a local Python environment, installs matching PyTorch and torchvision builds, and downloads the pinned SED and Detectron2 source, the two model checkpoints, and the official dataset archives. Allow roughly 12 GB of disk space. The dataset and checkpoint archives alone total about 3.9 GB.

The default experiment fits the head on **640 COCO-Stuff val2017 images**, selects its checkpoint on **160 other COCO-Stuff images**, and scores all **2,000 ADE20K-150 validation images**. It times the seven policies independently on a seeded subset of **256** validation images. Exact split IDs and per-image results are saved in the run directory. Repeating the same command resumes completed images.

~~~powershell
# Verify the installation on a few images in a separate run directory.
.\scripts\run.ps1 -Output runs/smoke -Fit 4 -Dev 4 -EvalImages 4 -TimingImages 4

# Time all 2,000 images in a separate run if you need a full latency sweep.
.\scripts\run.ps1 -Output runs/ade150-full-timing -TimingImages 2000
~~~

## Project layout

| Path | Role |
| --- | --- |
| [src/sed_adaptive_cer](src/sed_adaptive_cer) | Model loading, datasets, budget policy, and paired evaluation |
| [patches/adaptive_cer.patch](patches/adaptive_cer.patch) | Adds the optional policy to pinned SED and exposes reusable decoder inputs |
| [scripts](scripts) and [requirements-windows.txt](requirements-windows.txt) | Setup, run commands, and dependency versions |
| [tests](tests) | Tests for feature extraction, budget labels, and semantic IoU |
| [submission_docs](submission_docs) | Earlier assignment PDFs and the detailed report |
| [results](results) | Compact benchmark summaries and per-class IoU |
| `runs/` | Local logs, split IDs, saved head, and per-image measurements |

### How we measure

A-150 is ADE20K's 150-class segmentation benchmark. We compute mean intersection over union (mIoU) from all validation pixels with valid annotations, giving each class equal weight. The p50 and p95 are the median and 95th percentile of synchronized GPU inference latency. The full methodology, baseline comparisons, and interpretation are in the report.

### Git workflow

Each iteration is a commit. Use a new run directory for each changed experiment and record the compact results alongside the code. The large archives, checkpoints, environments, and detailed run logs stay in Git-ignored local storage.

~~~powershell
git status
git log --oneline
# After the next iteration is tested and its results are recorded:
git add src patches scripts tests submission_docs results README.md
git commit -m "Iteration 2 - Refine the budget policy"
~~~
