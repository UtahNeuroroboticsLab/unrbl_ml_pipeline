# Mamba decoders

The five registered models are `mamba_decoder`, `tcn_mamba_decoder`, `mamba_tcn_decoder`, `mamba_tcl_decoder`, and `mamba_lstm_decoder`. They port decoder-design's official_v2 Mamba-1 topology to TensorFlow/Keras. Runtime use requires neither PyTorch nor the reference checkout. Author licenses are distributed in `training/licenses`.

```bash
python testfile.py --config config_mamba.jsonc
python testfile_train_api.py --model mamba_tcl_decoder --config config_mamba.jsonc
python testfile.py --model mamba_decoder --config config_mamba.jsonc --state-mode reset
unrbl-train --model mamba_decoder --state-mode carry --list-losses
```

`testfile.py` writes the model, records, plots, and performance report. The direct API example calls `train()` and saves the model plus its resumable checkpoint directory. The existing CLSTM configuration is unchanged. The twelve-output preset requires twelve real kinematic columns and five-section KDF files.

```python
from unrbl_ml_pipeline import models
from unrbl_ml_pipeline.training import supported_losses, train, run_training

print(supported_losses("mamba_decoder", state_mode="carry"))
model, results = run_training(
    "mamba_decoder", config_path="config_mamba.jsonc",
    config={"epochs": 1}, output_dir="outputs",
)
```

# Configuration contract

Configuration order remains model defaults, JSONC, Python overrides, explicit CLI arguments.

The top-level `model_name` selects the decoder; `model` remains the object containing window and prediction settings. Selection precedence is explicit CLI/API model, Python `model_name`, JSONC `model_name`, then the script's fallback (normally `clstm_decoder`). Selection occurs before model-specific overrides are merged. Both example scripts and `unrbl-train` support this. `run_training(config_path="config_mamba.jsonc")` also selects the configured model; the explicit model argument of `train(model, ...)` remains authoritative. An explicit `--model` does not discard incompatible settings in the file.

Mamba-specific settings are:

| Key | Default | Meaning |
|---|---|---|
| `state_mode` | `carry` | `carry` or independent-window `reset` |
| `streams` | 4 | Parallel independent carry streams |
| `tbptt_steps` | 16 | Samples per gradient chunk |
| `seed` | 0 | Training RNG seed; split seed is separately configured |
| `model.window_size` | 10 | Chronological input window |
| `model.prediction_lag_samples` | 1 | Target offset after aligned window end |
| `alignment.lag_min`, `lag_max` | -20, 20 | Training-only lag search |
| `alignment.lag` | absent | Optional fixed lag instead of fitted choice |
| `feature_selection.count` | 48 | Maximum selected channels |
| `feature_selection.corr_threshold` | 0.3 | Reference GS candidate threshold |
| `feature_selection.candidates` | `emg` | `all`, `emg`, inclusive `a:b`, or explicit zero-based list |
| `feature_selection.indices` | absent | Explicit selected raw columns |
| `loss.type` | `mse` | See generated compatibility table |
| `loss.warmup_epochs` | 20 | Penalty calibration/activation boundary |
| `loss.temporal_ratio`, `jacobian_ratio` | 0.1, 0.03 | Calibration gradient-norm ratios |
| `epochs`, `min_epochs` | 500, 145 | Maximum epochs and patience stopping gate |
| `early_stopping_patience` | 45 | Validation RMSE non-improving epochs |
| `learning_rate`, `weight_decay` | 0.0005, 0.0001 | AdamW settings |
| `gradient_clip` | 1 | Global gradient norm |
| `batch_size` | 64 | Reset mode batch size |
| `resume_from` | absent | Run directory or checkpoints directory |

Architecture overrides: `mamba_width=64`, `state_size=16`, `conv_kernel=4`, `expand=2`, `tcn_width=96`, `lstm_units=96`, `head_units=192`, `dropout=0.2`. State mode, topology, and learned transformations are serialized. No output deadband is added by the model.

# Preprocessing and alignment

The default ordered steps are load, quality filtering, trial/rest split, lag alignment, channel selection, and windowing. Add `normalize` immediately before `window` to embed training-only feature and target normalization. Reference defaults do not normalize raw signals.

KDF formats are `cob`, `five_section`, or `auto`; ambiguous layouts require an explicit choice. Local COB recordings have 496 features and seven kinematic columns. The supplied config selects the first six. The reference twelve-output config cannot fabricate the remaining target columns.

Lag scoring and GS selection use training segments only. Lag `L` makes aligned feature row `t` read original feature row `t-L`. With target offset `P`, a prediction corresponds to target index `input_end+L+P`. Negative effective offsets describe targets earlier than the latest input; they are recorded, not silently turned into a future prediction. Online callers do not shift incoming data again.

Selection remains inside the saved model: provide the original raw feature count and ordering. The `emg` candidate preset uses raw indices 192 through 719 when at least 720 features exist; otherwise it uses all columns. Fewer than 48 eligible channels may be selected. All mappings and fitted scores are recorded.

No alignment or windows cross split, segment, file, or acquisition-gap boundaries. Original target/input indices and timestamps appear in `provenance.json`. Mamba defaults enable `data.require_usable_timestamps`: recordings with constant/nonfinite timestamps or missing usable clock information are rejected with a reason in the file selection log. If none remain, training fails clearly. By default, fix acquisition metadata or select recordings with usable timestamps. The explicit fallback below permits proxy timing without claiming it was acquired.

## Optional training trial alignment

`config_mamba.jsonc` adds `align_trials` after `align_lag` and before `select_features` and `window`. Comment out just the `"align_trials",` entry in `preprocessing_steps` to disable it. The complete JSONC list replaces the model's default list; there is no separate enable flag. Other model defaults and presets remain unchanged.

Only training cues receive per-trial residual correction. Validation and test retain global alignment, making their targets identical with the step enabled or disabled. This is offline label preparation, not a runtime EMG transformation or an independent measurement of actual movement. A constant shift preserves cue shape and duration; it cannot repair a slower movement or early release.

The splitter's `split.trial_detection.pre_cue_samples` is 40 in this preset and zero by default. Each trial receives at most half the preceding rest interval, except the first trial can use all available leading rest. Regions never overlap. Keep this setting unchanged when comparing alignment modes; reproducing historical unpadded preprocessing also requires setting it to zero. The quality filter still operates before alignment and can reject recordings before correction.

| `trial_alignment` setting | Default | Meaning |
|---|---|---|
| `feature_indices` | `emg` | Amplitude features, using the existing candidate syntax; all columns for 496-feature recordings |
| `max_residual_lag_samples` | 10 | Search radius around the globally aligned cue onset |
| `baseline_samples` | 10 | Rest samples immediately preceding the search interval |
| `threshold_multiplier` | 3.0 | Robust baseline deviation multiplier |
| `min_active_samples` | 3 | Consecutive above-threshold samples required |

These are initial sample-based engineering settings for MAV/RMS features, not validated physiological thresholds. Do not interpret them as milliseconds without knowing the feature update interval. Configure amplitude columns explicitly for mixed feature sets.

Detection excludes nonfinite and constant channels, subtracts each channel's baseline median, scales by its baseline MAD with a numerical floor, and averages positive standardized amplitudes. A unique sustained rising threshold crossing supplies the onset. Missing or ambiguous onsets, search-bound hits, missing context, truncated movements, and insufficient remaining window samples retain global alignment and receive a recorded fallback reason. Inspection and tuning use training data only.

For residual shift `D = EMG_onset - cue_onset`, the cue moves by `D` without another feature shift. Unsupported edges are cropped rather than wrapped or filled. Original label indices move with labels; aligned timestamps remain on the destination timeline. The effective target offset is `L - D + P`, where `L` is the global lag and `P` the prediction horizon. Each continuous segment retains a separate model-state reset boundary.

Recorded runs save `trial_alignment.png` with up to 12 deterministic successful/fallback examples and per-segment records in `preprocessing.json`. Records include global lag, detected/applied residual shift, local onset positions, original source indices, status, fallback reason, and cropped samples. Counts refer to trial segments because acquisition gaps can divide one trial. `provenance.json` preserves window mappings and original cue metadata. Changing the step list or its settings invalidates an incompatible checkpoint resume.

# Carry inference and loss semantics

```python
from unrbl_ml_pipeline.inference import load_model_file

model = load_model_file("outputs/example/model.keras")
stream = model.stream()
prediction = stream.predict(raw_window)  # [window_size, raw_features], oldest first
stream.reset()                          # new recording or acquisition gap
```

A streaming adapter owns one stream. It prefills the oldest W-1 samples once, then consumes the newest observation. Broken overlap resets automatically; callers must explicitly reset after gaps invisible in feature values. `predict_windows(batch)` explicitly evaluates independent windows. Bare `model.predict()` on a carry model is not a streaming interface.

Advanced callers can use `initial_state(batch_size)`, `prefill([B,W-1,F])`, and `step([B,T,F], state)`, which returns `[B,T,D]` predictions and the next state. Convolution state has shape `[B,kernel-1,inner]`, SSM state `[B,inner,state_size]`, and history `[B,W-1,selected_features]` for TCN-first or `[B,W-1,mamba_width]` otherwise. These runtime states are not persisted in model weights.

Supported Mamba objectives are `mse`, `mse_pinn`, `mse_jacobian`, and `mse_pinn_jacobian` in both modes. These are tested training capabilities, not a mathematical claim that other losses cannot apply to Mamba. Temporal terms match D1/D2 with SmoothL1 beta 0.01 and penalize squared predicted D3. They are sample differences, not time-normalized derivatives or a physical dynamics model. Jacobian penalties differentiate raw observations, scaled by training-window feature standard deviations; carry history is detached.

Regularizers activate after training-only calibration at the end of warmup (before epoch 1 when warmup is zero). `epochs` must exceed warmup for a regularized objective. Calibration failures are errors rather than silently disabled penalties. `loss` and `val_loss` remain MSE; `objective`, `d1`, `d2`, `d3`, and `jacobian` describe training penalties separately. The best validation checkpoint can precede penalty activation and is labeled accordingly.

Training displays a Keras progress bar with valid samples completed and an estimated remaining time. Padding does not count toward progress. Each completed epoch reports `loss`, `mse`, `rmse`, `objective`, `val_loss`, `val_mse`, `val_rmse`, `learning_rate`, and `epoch_seconds`. These fields also reach epoch callbacks and the returned/saved history. Training metrics are sample-weighted across updates; validation metrics use the complete validation set at the end of the epoch. `epoch_seconds` measures training and validation wall time, including initial graph compilation but excluding calibration and checkpoint writes. Use `verbose=2` in the Python API for completed-epoch lines only, or `verbose=0` for silent training. Resuming an older checkpoint leaves unavailable historical fields as JSON `null` rather than inventing past measurements.

The model summary shows shapes from its batch-one independent-window build pass. For default `mamba_decoder`, the Mamba block returns sequence features `(1, 10, 64)` plus convolution state `(1, 3, 128)` and SSM state `(1, 128, 16)`. Both readout normalization layers output `(1, 64)`. Carry execution uses its actual stream count and chunk length; a displayed batch dimension of one is not a training constraint. Custom widths, state sizes, and window lengths are reflected in the shapes.

# Checkpoints and reports

```bash
python testfile.py --model mamba_decoder --config config_mamba.jsonc \
  --resume-from outputs/previous_run/checkpoints --epochs 600
```

Resume restores the last completed epoch, optimizer, RNGs, calibration, best weights, and stopping counters. It checks data/config fingerprints; only output location, resume location, and maximum epoch count may change. A new wrapper run retains provenance of its resume source. State restarts at epoch boundaries. Checkpoints are immutable epoch directories with an atomically updated `latest.json` pointer; incomplete directories are not selected by that pointer.

The best weights are saved as `model.keras`. Reports include `preprocessing.json`, `provenance.json`, `calibration.json`, `training_state.json`, standard metrics/history/plots, and `performance.md`. Full convergence, TFLite/Summit deployment, MATLAB export, and historical checkpoint conversion are outside this implementation.

# Verification

`tools/generate_mamba_reference.py` generates pinned PyTorch prediction/weight fixtures in an isolated development environment. Normal tests use the frozen fixtures without PyTorch. The generated loss matrix is in `docs/loss_compatibility.md`.


# Optional sample-order timing fallback

```bash
python testfile.py --model mamba_decoder --config config_mamba.jsonc --allow-missing-timestamps
```

The same flag works with `testfile_train_api.py` and `unrbl-train`. Python/JSONC equivalent:

```python
config = {"data": {"allow_missing_timestamps": True}}
```

Usable clocks, including their gaps and resets, are preserved. Missing, constant, or nonfinite clocks receive a float64 proxy starting at zero with a fixed interval. The default interval is the median of each matched, successfully loaded recording's positive-delta median. Each recording has equal weight; feature and target values are not used. Date/glob-excluded files cannot contribute. This timing estimate uses matched recordings before beta filtering; it is distinct from the training-only lag, channel selection, and normalization fits.

Across the local 26 usable clocks, per-file medians are 990 or 992 NIP ticks and the resulting estimate is 992 ticks/sample. The reference acquisition code specifies a 30,000-tick/second NIP clock and a 0.033-second loop, consistent with roughly 33 ms per row. The fallback operates in timestamp units, not an assumed universal clock frequency.

If the selected files all have unusable timestamps, supply the interval explicitly:

```bash
python testfile.py --model mamba_decoder --config config_mamba.jsonc \
  --allow-missing-timestamps --sample-interval 992
```

Equivalent: `data.sample_interval = 992`. This overrides only proxy spacing, not real clocks. Without donors or an explicit interval, fallback fails clearly. It never removes beta filtering.

`[ROW ORDER]` console entries and each selection/provenance record identify proxy timing, its interval, and why it was needed. `preprocessing.json` records the estimate and donor files; `performance.md` states how many selected files use proxy timing. Original timestamps are preserved in the preprocessing state. Each file and trial/split segment remains a separate stream, but unrecorded within-segment gaps cannot be inferred from row order. Proxy timestamps cannot be matched to absolute KEF events; use a sample-based/detected-trial split for such recordings.

Keep `align_lag` in a custom Mamba preprocessing list: it also separates recorded timestamp gaps and resets before windowing. To disable lag estimation while retaining this boundary handling, set `alignment.lag = 0`. Calling `window` directly without that segmentation does not enforce timestamp boundaries.
