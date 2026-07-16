# unrbl_ml_pipeline
A standardized python package for training ML models, replicating results, and studying Neurorobotics. This is tailor-made as a tool primarily for usage within the Utah Neurorobotics Lab, but is open-access for replicating results from prior publications as well.

## Installation

```bash
pip install unrbl-ml-pipeline           # core: inference/deploy, works on Python 3.7+
pip install "unrbl-ml-pipeline[train]"  # + training stack, needs Python 3.10+
```

The distribution is `unrbl-ml-pipeline`; the import is `unrbl_ml_pipeline`.

Neither command installs a TFLite interpreter. See below for what `.tflite` inference needs.

### Deploying to the Ripple Summit (Python 3.7, Debian 8, i686)

**This is a two-step install.** `pip install unrbl-ml-pipeline` gives you numpy and a `load_model()`
that raises `ImportError` until you also install the interpreter:

```bash
pip install unrbl-ml-pipeline
pip install tflite_runtime-2.5.0-cp37-cp37m-linux_i686.whl   # from the repo below
```

The interpreter is **not** a declared dependency, and that is forced rather than chosen:

- PyPI has no i686 `tflite-runtime` wheel for any Python version.
- The lab's own build at
  [tflite_for_rippleSummit](https://github.com/UtahNeuroroboticsLab/tflite_for_rippleSummit) is tagged
  `linux_i686`, which PyPI rejects on upload (only `manylinux*` Linux tags are accepted).

So declaring `tflite-runtime` would make the Summit install *fail outright* instead of partially
succeeding. Documenting the second step is the honest trade. Everything else resolves normally:
`numpy>=1.21` has a `manylinux_2_12_i686` cp37 wheel, and Debian 8's glibc 2.19 satisfies it.

> **TFLite compatibility ceiling.** The Summit's interpreter is **2.5.0 (2021)**. A model converted by
> a modern TF converter may use ops or a flatbuffer schema that 2.5.0 cannot load — and this fails
> **on-device after deploy**, not at conversion time. Validate converted models on the Summit (or
> against a 2.5.0 interpreter) before a run depends on them. There is no automated guard for this yet;
> add one to the export path when it is written.

### `.tflite` inference elsewhere

| Platform | What provides the interpreter |
|---|---|
| Ripple Summit (3.7, i686) | lab-built wheel, installed out-of-band (above) |
| Linux x86_64 / aarch64 | `pip install tflite-runtime` |
| Windows / macOS | no `tflite-runtime` wheels exist; use `[train]`, which brings `tf.lite` |

`load_model()` tries `tflite_runtime` first and falls back to `tensorflow.lite`, so whichever you have
works without a code change.

### `.keras` inference

Needs Python **>=3.11** (Keras 3), so **the Summit cannot load `.keras` at all** — the newest TF that
runs on 3.7 is 2.11, which predates the format. Convert to `.tflite` on a modern machine and deploy
that. `load_model()` checks the interpreter version and says so explicitly rather than failing with a
misleading "No module named keras".

### Why the split

The core depends only on `numpy`, so it installs on the lab's Python 3.7 machine. Training pulls
TensorFlow, and **TF 2.16+ requires Python 3.10+** - no packaging trick changes that, so the training
stack lives in the `[train]` extra. If you `pip install "unrbl-ml-pipeline[train]"` on 3.7 it fails
loudly with "no matching distribution" rather than quietly installing a broken environment.

Dependency versions are not pinned per-Python. `numpy>=1.21` resolves to 1.21.6 on Python 3.7 and to
current numpy on 3.11+, because pip reads numpy's own `requires-python`. Don't add environment markers
for this - it's already handled.

## Repository layout

```
src/unrbl_ml_pipeline/     # the package (src layout: it must be installed to import)
  __init__.py              # __version__ lives here - single source of truth
tests/                     # pytest; test_smoke.py must stay importable on 3.7
pyproject.toml             # deps, extras, ruff config
.github/workflows/
  qa.yml                   # ruff + pytest on 3.11-3.13, plus a python:3.7 container job
  publish.yml              # on a v* tag: build + publish to PyPI via trusted publishing
```

Where new code goes:

| Adding | Goes in | Constraint |
|---|---|---|
| Inference, data reading, file formats | core modules | **must be 3.7-compatible** |
| Model definitions, training, plotting | modules that only load under `[train]` | may use modern syntax |
| Anything importing `tensorflow` | never at import time from `__init__.py` | see below |

### The two rules that packaging imposes

1. **Core code must stay Python 3.7-syntax-clean.** No `match`, no `X | Y` annotations, no dataclass
   `slots`. `ruff` is configured with `target-version = "py37"` and runs in CI, so this fails in review
   rather than on the lab machine. Use `from __future__ import annotations` if you want modern type
   syntax in annotations on 3.7.
2. **Never import `tensorflow` at package import time.** `import unrbl_ml_pipeline` must work on a
   core-only install. Import TF inside the functions/modules that need it, so the 3.7 machine can
   import the package without the training stack present. The 3.7 CI job exists to catch violations.

## Contributing

```bash
git clone git@github.com:UtahNeuroroboticsLab/unrbl_ml_pipeline.git
cd unrbl_ml_pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"                       # use Python 3.11-3.13
```

Note: `[train]` will not install on Python 3.14+ yet - TensorFlow has no wheels for it.

Before opening a PR:

```bash
ruff check .
pytest
```

Both run in CI across 3.11/3.12/3.13. A separate job installs the built wheel into a `python:3.7-slim`
container and runs the tests there - that job is what actually holds the 3.7 support claim honest.

To check 3.7 locally before pushing:

```bash
python -m build
docker run --rm -v "$PWD:/w" -w /w python:3.7-slim \
  sh -c "pip install dist/*.whl && python -c 'import unrbl_ml_pipeline'"
```

### Releasing

1. Bump `__version__` in `src/unrbl_ml_pipeline/__init__.py`.
2. Tag it: `git tag v0.1.0 && git push --tags`.

`publish.yml` builds and uploads to PyPI over OIDC trusted publishing. There is no API token to
rotate, and releases only happen from a tag on this repo.

## Intended Features

Here are the known features it needs, feel free to add:

- Plot Training
- Plot Testing
- read format @Fredi mentioned
- implement way to have y labeled different from what the actual output of the NN produces (y labeled is kinematics but maybe we also want confidence in real time) @Marshall Trout this might already be accessible after saving a Kera's/tflite model. Or it might be too advanced for our tflite that the Summit runs. Something to be aware of but I can chat with @Jewan Chae?
- additional preprocessing options for when using raw data (eventually neural :eyes:). Including preprocessing update rate and MAV window size.
- Additional postprocessing steps like kalman filters or latching filters in training and or testing.
- add in published models: 7 layer CNN, AlexNet, LSTM, C-LSTM,
- Provide wrapper to improve calling from command line
- Provide lean config files for simple use
- Provide code documentation
- provide how-to documentation for KDFs and other file types
- Improve inference pipeline so it is more simple
- provide example datasets for testing setup

Models:
1. George et al. EMBC 2018 (https://ieeexplore.ieee.org/document/8513342)
    - Thompson et. al. 2022

2. Thomson et al. LSTM (https://pmc.ncbi.nlm.nih.gov/articles/PMC12742979/) 
3. CLSTM by Troy (Recent)

----
Automated QA
- [ ] `.github/workflows/qa.yml` - ruff + pytest on 3.11-3.13 and a Python 3.7 container job
- [ ] `.github/workflows/publish.yml` - tag-triggered PyPI release via trusted publishing
