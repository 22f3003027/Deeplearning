# CS601T Assignment 4 - Group 32

Complete Python implementation of the six tasks in `CS601T_Assignment4_Details_30Sept2026.pdf`.
Uses **only Group_32.zip** and preserves its train/val/test splits. Digits: **0, 3, 5, 6, 9**.

## Setup and run

Place this code folder beside `Group_32.zip`. Python 3.11 or newer is recommended; the delivered run used Python 3.13.7 and CPU PyTorch.

Clean environment with uv, from inside this code folder:

```powershell
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe run_assignment.py --data ../Group_32.zip
.venv/Scripts/python.exe verify_results.py
.venv/Scripts/python.exe report.py --compile
```

For the environment already created in the original workspace, run from the workspace root:

```powershell
.venv/Scripts/python.exe Group32_Assignment4_code/run_assignment.py --resume
.venv/Scripts/python.exe Group32_Assignment4_code/verify_results.py
.venv/Scripts/python.exe Group32_Assignment4_code/report.py --compile
.venv/Scripts/python.exe -m unittest discover -s Group32_Assignment4_code/tests -v
.venv/Scripts/python.exe Group32_Assignment4_code/make_submission.py
```

The original workspace environment uses `--system-site-packages` to reuse installed numerical libraries; a clean installation uses the first set of commands. No GPU, external MNIST download, or pretrained model is required.

The default training budget is 80 epochs per autoencoder and 40 per FCNN, batch size 256, Adam learning rate 0.001, base seed 32, and four CPU threads. Early stopping and validation-selected checkpoint restoration apply. `--resume` skips completed training runs. For a different training budget, use a new output folder:

```powershell
python run_assignment.py --data ../Group_32.zip --output results_long --ae-epochs 150 --classifier-epochs 60
python verify_results.py --results results_long
python report.py --results results_long --compile
```

## Experimental choices

- PCA: exact eigendecomposition of the **training** sample covariance, training-mean centering for all splits; dimensions 32, 64, 128, 256.
- FCNN: hidden layers `[64]`, `[128]`, `[128,64]`, ReLU hidden activations, five logits, cross-entropy. Identical architectures in Tasks 1, 3, and 4. Train-fitted feature standardization is saved with each classifier.
- Shallow AE: `784 -> d -> 784`. Deep AE: `784 -> 400 -> d -> 400 -> 784`. Logistic sigmoid is used consistently in all hidden and reconstruction layers; pixelwise MSE loss.
- Task 4 says "2-hidden layer" in its heading, while Task 2 specifies a three-hidden-layer AE. We use the two-layer encoder (`784 -> 400 -> d`) from that explicitly specified deep AE. This interpretation is recorded in the report.
- Reconstruction error: `sum((prediction - clean_image)^2)/(N*784)`, evaluated after training on complete splits. Multiply by 784 for average per-image squared L2 reconstruction error.
- Denoising: independent **Bernoulli masking** of pixels to zero with probability 0.20 or 0.40, with clean targets. Fresh training masks; deterministic fixed masks for final reconstruction evaluation and examples.
- Task 5 bottleneck follows the brief's **best Task 3 test accuracy**, ties favoring smaller dimensions. The FCNN reuses that bottleneck's best Task 3 validation-selected architecture. Primary classification uses clean encodings; an additional noisy-input robustness evaluation is included.
- Classifier architectures and checkpoint epochs are selected on validation data; only the selected architecture is tested at each dimension. Test rankings across dimensions are descriptive, and the report notes the test-informed Task 5 design.
- Signed input weights for **every** neuron in the chosen shallow AE and both DAEs are plotted with one shared color scale. A spatial smoothness statistic supplements the visual comparison.
- One image per class per split is selected deterministically and reused across architectures. No labels enter representation training.

## Assignment 3 comparison

The subsequently supplied Assignment 3 handout is implemented in `../Group32_Assignment3_code/`: three FCNN architectures with 3, 4, and 5 hidden layers, each trained using all seven required optimizers, prescribed batch sizes, matched initial weights, and the exact loss-difference stopping criterion with no epoch limit. Its best validation-selected converged model supplies the measured Assignment 3 comparison in the updated report.

To reproduce it, run `../Group32_Assignment3_code/run_assignment3.py`, then `verify_results.py` in that folder. For a new Assignment 4 training run, pass `--assignment3-results ../Group32_Assignment3_code/results/assignment3_best.json`. The LaTeX generator also attaches this verified baseline to the completed Assignment 4 results without retraining existing models.

The earlier raw-pixel reference used a different, smaller architecture search and is retained in the saved experiment history; it is **not used as Assignment 3** in the updated comparison. These Assignment 3 values are a newly measured reproduction, not a recovered historical submission.

## LaTeX reports

`report.py` now generates **LaTeX source**, and `--compile` produces both Assignment 3 and Assignment 4 PDFs. Install MiKTeX or TeX Live with `pdflatex`, or upload the portable project ZIPs to Overleaf. The document packages are standard: geometry, graphicx, amsmath, amsfonts, booktabs, array, longtable, tabularx, xcolor, fancyhdr, hyperref.

```powershell
python report.py --compile
```

Projects: `../output/latex/assignment3/` and `../output/latex/assignment4/`. Each contains its main `.tex` and all figure assets. Compile its main file twice with `pdflatex`, or use the corresponding `Group32_Assignment*_report_latex.zip` in `output/` for Overleaf. You may edit the `.tex` directly; regeneration from Python overwrites the generated source.

The Assignment 4 code ZIP bundles the measured Assignment 3 reference and figures in `assignment3_reference/` so its report can be reproduced without the sibling Assignment 3 folder. The separate Assignment 3 code ZIP contains the actual optimizer implementations and reproduction instructions.

## Files and outputs

| File | Purpose |
| --- | --- |
| `data.py` | Archive loading, checks, dataset provenance, exact PCA |
| `models.py` | Required architectures, FCNNs, masking corruption |
| `training.py` | Seeded training, checkpoint selection, encoding, metrics |
| `run_assignment.py` | Complete experiment orchestration and resume |
| `plots.py` | Reconstructions, confusion matrices, weights, learning curves |
| `report.py`, `latex_report.py` | Editable LaTeX projects and compiled PDFs from actual saved results |
| `verify_results.py` | Independently recompute saved checkpoint metrics |
| `tests/test_pipeline.py` | Leakage, architecture, corruption, checkpoint tests |
| `make_submission.py` | Package correctly named submission artifacts |

`results/` contains:

- `metrics.json`: complete measured metrics, histories, seeds, configurations.
- `dataset_manifest.json`: original image paths, class counts, archive SHA-256, cross-split pixel-hash audit.
- `classification_validation.csv`: validation accuracy of every FCNN architecture.
- `classification_selected.csv`: train/validation/test accuracy of selected FCNNs.
- `reconstruction_errors.csv`: final reconstruction MSE of all ten AEs.
- `verification.json`: results of the checkpoint/output audit.
- `figures/`: all 14 report figures.
- `representations/`: compressed arrays, with `train`, `val`, `test` keys, for every representation and both noisy DAE evaluations. These are the **unstandardized** bottleneck outputs; FCNN standardization is stored in checkpoints.
- `checkpoints/`: all model weights and `pca.npz` (training mean, eigenvectors, eigenvalues).
- `dataset.npz`: normalized image cache.

Large caches, checkpoints, and feature arrays stay in the workspace. The code submission ZIP includes all `.py` source, requirements, README, compact measured results, figures, and verification; it excludes large reproducible caches and the source dataset. In the ZIP, the compact measured outputs are stored under `reference_results/`, keeping the default `results/` path available for a fresh run. Extract it, place `Group_32.zip` beside the extracted folder, and rerun to regenerate the large outputs.

Submission artifacts:

- `../output/Group32_Assignment4_code.zip` (top-level folder `Group32_Assignment4_code`).
- `../output/pdf/Group32_Assignment4_report.pdf`.
- `../output/Group32_Assignment4_report_latex.zip` (LaTeX source and all local figures).

No Moodle upload is performed by these scripts.
