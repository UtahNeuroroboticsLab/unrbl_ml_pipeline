# Mamba validation record

Validation date: 2026-10-30. This is implementation evidence, not a convergence or clinical-performance claim.

## Package checks

The complete package regression suite passed 88 tests; two additional targeted optimizer/LSTM-bias regression tests also passed (90 package tests total). The suite includes existing CLSTM, Caleb, Kalman, array-based training, TFLite fallback, and import guards. Ruff and `git diff --check` passed. Keras/NumPy and TensorFlow interpreter deprecation warnings remain dependency warnings.

Frozen fixtures generated from the pinned decoder-design PyTorch implementation verify all five architectures, independent windows, nonzero recurrent state, chunk equivalence, conditional input-Jacobian penalties, and second parameter derivatives. Normal tests require neither PyTorch nor `.ref`.

## Independent acceptance

A separate agent started with no inherited conversation. It designed tests from the requirements, public API documentation, and original reference sources without intentionally reading implementation or implementation-authored tests. Its initial acceptance passed 72 cases plus all three CLI entrypoints. Its independently mapped reference predictions differed by at most 2.09e-7.

The initial report and tests were preserved before a separately labeled source-aware numerical review. That follow-up identified training details missed by forward parity: corrected-moment epsilon placement in AdamW, separate LSTM biases, and calibration probe selection. The implementation also corrects first-chunk prefill differentiation and uses explicit centered-variance LayerNorm for accurate second derivatives. The follow-up passed 67 checks on the corrected source: 49 scientific/boundary checks plus 18 acceptance reruns. Reference per-parameter Jacobian gradients differed by at most 1.79e-7; calibration coefficients differed by at most 1.33e-7. Actual AdamW updates, clipping, A_log/D decay exclusions, first-chunk prefill gradients, later detached-state gradients, and exact CPU resume were verified. No unresolved defect was found in the tested scope.

Local independent test artifacts and both review phases are preserved under `outputs/mamba_validation/independent/`. The initial blind review is not relabeled as a blind review of later changes; subsequent checks are explicitly source-aware.

## GPU and real data

The bounded GPU check passed second derivatives for all five models and regularized resume for carry/reset modes. Maximum resumed prediction difference was 1.49e-8. TensorFlow peak allocated GPU memory was 33,926,400 bytes under a 2 GiB cap. These are small synthetic validation models, not a benchmark of full-size production training.

The data audit examined all 79 local recordings. Beta filtering qualified 74; 23 also had usable timestamp progression. Mamba's default timestamp requirement records and excludes constant/nonfinite clocks. Existing models retain their prior behavior.

Three one-epoch CPU runs used the three `hook*/trainKDF_202607*.kdf` recordings, six targets, and reduced width-8 architectures:

| Mode | Objective | Seconds | Test MSE |
|---|---|---:|---:|
| Carry | MSE | 8.08 | 0.27298 |
| Reset | MSE | 4.25 | 0.26546 |
| Carry | MSE + temporal + Jacobian | 20.87 | 0.28479 |

Regularized smoke validation used warmup zero. These one-epoch scores verify execution; they do not compare converged model quality. Peak process RSS across the combined CPU validation process was about 2.4 GiB.

Reproduce package checks with `pytest`, oracle generation with `tools/generate_mamba_reference.py` in an isolated CPU PyTorch environment, bounded real-data validation with `tools/validate_mamba_data.py`, and GPU checks with `tools/validate_mamba_gpu.py`. GPU validation requires access to the actual GPU. Detailed run records are in `outputs/mamba_validation/`.

Full convergence, historical checkpoint migration, deployment latency, TFLite/Summit export, MATLAB export, and crash-injection testing are not established by these checks.
