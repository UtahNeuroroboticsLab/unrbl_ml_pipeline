# unrbl_ml_pipeline
A standardized python package for training ML models, replicating results, and studying Neurorobotics. This is tailor-made as a tool primarily for usage within the Utah Neurorobotics Lab, but is open-access for replicating results from prior publications as well.

## Installation

```bash
pip install unrbl-ml-pipeline
```

One command, everywhere. What you get is chosen by the interpreter you install into:

| Interpreter | Resolves to | Can do |
|---|---|---|
| Python 3.10–3.13 | numpy, TensorFlow, Keras, scikit-learn, matplotlib | everything: training, `.tflite` and `.keras` inference |
| Python 3.7 (Ripple Summit, i686) | numpy, `unrbl-tflite-runtime` 2.5.0 | `.tflite` inference only |

The distribution is `unrbl-ml-pipeline`; the import is `unrbl_ml_pipeline`.

<!-- There are no extras to remember - `[train]` and `[rppl]` do not exist. See
[Why markers](#why-markers) if you want the mechanism. -->

### Deploying to the Ripple Summit (Python 3.7, Debian 8, i686)

```bash
pip install unrbl-ml-pipeline
```

That is the whole install. The interpreter comes from
[`unrbl-tflite-runtime`](https://github.com/UtahNeuroroboticsLab/unrbl_tflite_runtime), the lab's
rebuild of TFLite 2.5.0 for i686/cp37, which is declared as an ordinary dependency and resolves
automatically on 3.7.

**Your pip must be >=19.3** to recognise the `manylinux2014` tag from an index - check with
`pip --version`. Older pip will not see the wheel and the install will fail to resolve.

> **TFLite compatibility ceiling.** The Summit's interpreter is **2.5.0 (2021)**. A model converted by
> a modern TF converter may use ops or a flatbuffer schema that 2.5.0 cannot load - and this fails
> **on-device after deploy**, not at conversion time. Validate converted models on the Summit (or
> against a 2.5.0 interpreter) before a run depends on them. There is no automated guard for this yet;
> add one to the export path when it is written.

### `.tflite` inference elsewhere

Nothing to install beyond the package. On 3.10+ TensorFlow is a dependency, so `tf.lite` is always
present - including on Windows and macOS, where no `tflite-runtime` wheel has ever existed.

`load_model()` tries `tflite_runtime` first and falls back to `tensorflow.lite`, so it uses whichever
backend your interpreter resolved to without a code change.

### `.keras` inference

Needs Python **>=3.11** (Keras 3), so **the Summit cannot load `.keras` at all** - the newest TF that
runs on 3.7 is 2.11, which predates the format. Convert to `.tflite` on a modern machine and deploy
that. `load_model()` checks the interpreter version and says so explicitly rather than failing with a
misleading "No module named keras".

### Why markers

The package supports everything, but the Summit's Python 3.7 cannot run most of it. That is expressed
with **environment markers** in `pyproject.toml`, not extras:

```toml
dependencies = [
    "numpy>=1.21",
    "tensorflow>=2.16;     python_version >= '3.10' and python_version < '3.14'",
    "scikit-learn;         python_version >= '3.10' and python_version < '3.14'",
    "matplotlib;           python_version >= '3.10' and python_version < '3.14'",
    "unrbl-tflite-runtime; python_version <  '3.8'",
]
```

<!-- An extra can only **add** to the core dependencies - there is no way to spell "core minus
TensorFlow" as an extra, so `[rppl]` could never have meant "the inference-only subset". A marker
whose condition is false drops its dependency silently, which is exactly the subtraction needed.
pip evaluates the condition against the interpreter it is installing into, so one command gives the
full stack on 3.11 and the 2.5.0 interpreter on 3.7. -->

The TF marker is **mandatory, not cosmetic**. TensorFlow declares `requires-python >=3.10`, so an
unmarked `tensorflow` in `dependencies` would fail the entire 3.7 resolve with "no matching
distribution" and take the Summit's install down with it.

The `< '3.14'` ceiling is **not** redundant with TF's own `requires-python`. TF declares `>=3.10` with
no upper bound, then ships no cp314 wheel - so without the ceiling, pip on 3.14 fails the *entire*
install rather than dropping TF. **Bump it when TF ships cp314.**

`requires-python` stays `>=3.7` and is load-bearing: raise it to 3.10 and pip refuses the package on
the Summit outright, making the 3.7 path impossible no matter what the markers say.

`numpy` deliberately has **no** marker - it declares its own `requires-python`, so pip resolves 1.21.6
on 3.7 and current numpy on 3.10+ by itself. Don't add one.

**Known gaps.** Both resolve to numpy only, with no interpreter, so `load_model()` raises:

| Interpreter | Why |
|---|---|
| Python 3.8/3.9 | TF needs >=3.10 and the i686 wheel is cp37-only - neither marker fires |
| Python 3.14+ | the ceiling above; TF has no cp314 wheel yet |

No lab machine runs either, and CI covers what actually ships (3.7 i686, 3.11–3.13). If one ever does,
those gaps need closing.

## Repository layout

```
src/unrbl_ml_pipeline/     # the package (src layout: it must be installed to import)
  __init__.py              # __version__ lives here - single source of truth
  inference.py             # load_model() - root module, must import on 3.7
  training/                # the 3.10+ subpackage; may import TF at module level
tests/                     # pytest; the whole suite must run on 3.7
  toy_model.tflite         # X -> 3X, one dense layer; the end-to-end inference fixture
pyproject.toml             # deps, markers, ruff config
.github/workflows/
  qa.yml                   # ruff + pytest on 3.11-3.13, plus a real i686 Python 3.7 job
  publish.yml              # on a v* tag: build + publish to PyPI via trusted publishing
```

Where new code goes:

| Adding | Goes in | Constraint |
|---|---|---|
| Inference, data reading, file formats | root modules | **must be 3.7-compatible** |
| Model definitions, training, plotting | `training/` | 3.10+ only; may use modern syntax |
| Anything importing `tensorflow` | never at import time from a root module | see below |

### The two rules that packaging imposes

Both survive the move to markers - the Summit now gets a *marker-thinned* install rather than a
core-only one, but it still has no TensorFlow and still runs Python 3.7.

1. **Root modules must stay Python 3.7-syntax-clean.** No `match`, no `X | Y` annotations, no dataclass
   `slots`. `ruff` is configured with `target-version = "py37"` and runs in CI, so this fails in review
   rather than on the lab machine. Use `from __future__ import annotations` if you want modern type
   syntax in annotations on 3.7.
2. **Never import `tensorflow` at package import time.** `import unrbl_ml_pipeline` must work on the
   Summit, where TF is absent. Import TF inside the functions/modules that need it, or put the module
   in `training/`. This matters *more* now that TF is a core dependency on 3.10+: there is no longer
   any dev environment where a stray root-level `import tensorflow` fails on its own, so
   `tests/test_core_import.py` and the 3.7 CI job are the only things that catch it.

## Contributing

```bash
git clone git@github.com:UtahNeuroroboticsLab/unrbl_ml_pipeline.git
cd unrbl_ml_pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                             # use Python 3.11-3.13; pulls TF
```

**Use 3.11–3.13, not 3.14.** TensorFlow has no cp314 wheel, so a 3.14 dev install resolves to numpy
only and `test_tflite_inference.py` skips - you would be running a suite that cannot exercise
`load_model()` at all.

Before opening a PR:

```bash
ruff check .
pytest
```

Both run in CI across 3.11/3.12/3.13. A separate job runs the built wheel on **real 32-bit i686
Python 3.7** and executes actual inference through it - that job is what holds the 3.7 support claim
honest. It cannot be done in a `python:3.7-slim` container: that image is x86_64, and the Summit's
interpreter wheel is i686-only, so it would not install there at all.

To check 3.7 locally before pushing (needs Docker):

```bash
python -m build --wheel
docker run --rm --platform linux/386 -v "$PWD:/io:ro" \
  quay.io/pypa/manylinux2014_i686@sha256:683a201f94d04ff9548ab51c9583b28a34a326a96363c3520260efdfdca3cc03 \
  sh -c 'PY=/opt/python/cp37-cp37m/bin/python
         mkdir -p /work && cp -r /io/tests /io/dist /work/ && cd /work
         $PY -m pip install dist/*.whl pytest && $PY -m pytest tests/ -v'
```

**Why it copies to `/work` instead of running in `/io`:** on Docker Desktop (Windows/macOS) the bind
mount is not a real Linux filesystem and `mmap` fails on it, so the TFLite interpreter dies with
`ValueError: Mmap of '11' failed` while loading the model - a mount artifact, not a bug in your
change. CI runs on a Linux runner where the mount is real, so `qa.yml` works straight out of `/io`.

That image digest is pinned deliberately - read the comment in `qa.yml` before changing it.

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
- [x] `.github/workflows/qa.yml` - ruff + pytest on 3.11-3.13 and a real i686 Python 3.7 job
- [x] `.github/workflows/publish.yml` - tag-triggered PyPI release via trusted publishing
