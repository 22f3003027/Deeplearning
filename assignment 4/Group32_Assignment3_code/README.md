# Group 32 - CS601T Assignment 3 reproduction

Follows the uploaded Assignment 3 handout on **Group_32.zip**, using the same splits as Assignment 4. These are newly measured reproduction results, not a historical submission.

From the workspace:

```powershell
.venv/Scripts/python.exe Group32_Assignment3_code/run_assignment3.py
.venv/Scripts/python.exe Group32_Assignment3_code/verify_results.py
.venv/Scripts/python.exe -m unittest discover -s Group32_Assignment3_code/tests -v
.venv/Scripts/python.exe Group32_Assignment4_code/report.py --compile
```

For a clean environment inside this folder, run `uv venv .venv`, then `uv pip install --python .venv/Scripts/python.exe -r requirements.txt`. Place `Group_32.zip` beside this folder.

`--resume` reuses completed runs. There is **no maximum epoch argument**: the handout requires the loss-difference criterion, rather than an iteration budget. Slow full-batch runs may need thousands of epochs.

## Protocol

- ReLU FCNN hidden layers `[32,16,8]`, `[32,24,16,8]`, `[32,24,16,12,8]` (3, 4, 5 layers). The first hidden width is held at 32; intermediate widths and depth vary.
- Five output logits; average cross-entropy; pixels divided by 255; supplied splits retained.
- Identical He-normal weights and zero biases for every optimizer within an architecture. Initial parameter hashes and checkpoints are saved.
- SGD, momentum, NAG, Adam: **batch size 1**, exactly one update after every image. Matching epoch permutations across online optimizers.
- Batch GD, AdaGrad, RMSProp: **full batch of 11,385 training examples**, one update per epoch.
- Learning rate 0.001; momentum 0.9; RMSProp alpha 0.99, epsilon 1e-8; Adam betas (0.9,0.999), epsilon 1e-8. AdaGrad epsilon 1e-8, zero initial accumulator, no learning-rate decay.
- Stop at the **first** epoch with `abs(post_epoch_mean_training_CE - previous_post_epoch_mean_training_CE) < 1e-4`. No epoch cap or validation stopping. Small loss changes at high loss are retained and discussed.
- Final converged weights determine train/validation metrics. Highest validation accuracy selects architecture and optimizer, lower validation CE breaking a tie. Only that model is tested.
- Numba compiles exact online backpropagation and optimizer loops in float64; NumPy handles full-batch products. All seven implementations are numerically checked against PyTorch. No minibatch substitute is used.
- CPU BLAS pools are limited to one thread for these small online matrix-vector operations. Subnormal floating-point values can be flushed to zero to avoid severe CPU slowdowns; each run records that hardware setting.

## Outputs

`results/metrics.json`: all 21 runs, loss histories/deltas, epochs, step counts, accuracies, confusion matrices, seeds, hashes. `optimizer_results.csv`: compact tables. `assignment3_best.json`: measured Assignment 4 comparison. `verification.json`: artifact audit. `checkpoints/`: initial and converged weights. Four figures provide superimposed optimizer losses for each architecture and selected train/test confusion matrices.

Code ZIP compact outputs reside under `reference_results/`; regenerate caches and checkpoints into the default `results/` directory by running the code.

The LaTeX generator is `Group32_Assignment4_code/report.py`. Project: `output/latex/assignment3/`; compiled report: `output/pdf/Group32_Assignment3_report.pdf`. The portable LaTeX ZIP includes `.tex` and local figures for Overleaf.
