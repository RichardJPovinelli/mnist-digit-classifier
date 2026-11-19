# MNIST digit classifier

This repository trains a PyTorch convolutional neural network to classify
MNIST handwritten digits from 0 to 9. It includes commands to train, evaluate,
and predict individual digits.

## Run the Python program

These instructions use Python 3.12 on Windows and macOS. Install Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/) once, then
follow the same clone and run steps on either platform.

### One-time setup

#### Windows (PowerShell)

Open **PowerShell** from the Start menu and run these commands. Skip any tool you already have.

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
```

Close and reopen PowerShell after installing. If `winget` is unavailable, see the [Git for Windows installer](https://git-scm.com/install/win) and the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

#### macOS (Terminal)

Open **Terminal** from Spotlight. If you already use Homebrew, run:

```sh
brew install git uv
```

Otherwise, if Git is missing or macOS prompts you to install developer tools, install Apple's Command Line Tools and wait for the installation to finish:

```sh
xcode-select --install
```

Then install uv using the [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal after installing.

### Clone and set up

1. **Clone the repository.** Open PowerShell on Windows or Terminal on macOS
   and go to the folder where you want to keep the project:

   ```sh
   git clone https://github.com/RichardJPovinelli/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

   If you already cloned the repository, open that folder instead.
2. From the folder containing `mnist.py` and `pyproject.toml`, install the
   locked dependencies:

   ```sh
   uv sync --locked
   ```

   uv uses Python 3.12 and installs the versions recorded in `uv.lock`.

### Check your setup

From the repository folder, run these commands one at a time:

```sh
git --version
uv --version
uv run --locked python --version
```

The last command should print `Python 3.12.x`, where `x` is the patch version.
If the program cannot find `mnist.py` or `pyproject.toml`, check that you are
in the repository folder.

### Train

Choose a new run directory; existing directories are not overwritten.

```sh
uv run --locked python mnist.py train --run-dir runs/my-run --device cpu
```

MNIST is downloaded automatically into `data/` on the first training run,
which requires network access. The official training set is split into 55,000
training and 5,000 validation examples using a fixed seed. The 10,000 official
test examples are held out until final evaluation.

The default is three epochs, batch size 128, learning rate 0.001, and seed
42. Use `uv run --locked python mnist.py train --help` for options. Use
`--device auto` to select CUDA when available, or `--device cuda` to require
it. Training prints epoch-level loss and accuracy and saves the checkpoint
with the highest validation accuracy.

### Evaluate

After model and training choices are fixed, evaluate the selected checkpoint
once on the held-out test set:

```sh
uv run --locked python mnist.py evaluate --run-dir runs/my-run --device cpu
```

The command reports accuracy (correct predictions divided by test examples)
and a confusion matrix with true digits as rows and predicted digits as
columns. It writes `test-evaluation/report.json` and up to ten misclassified
images under the run directory. Evaluation refuses to overwrite an existing
result.

### Predict

Provide an 8-bit grayscale, 28 x 28 image with the same light-digit-on-dark
background convention as MNIST:

```sh
uv run --locked python mnist.py predict --checkpoint runs/my-run/best.pt --image path/to/digit.png --device cpu
```

The command prints the predicted digit. Incorrect image sizes, image modes,
and missing or invalid checkpoints produce errors.

### Artifacts and reproducibility

Each run contains `config.json` (configuration, dataset and split details,
environment versions, hardware and determinism settings), `history.json`
(training and validation metrics), and `best.pt` (the selected model).
The `uv.lock` file records resolved dependencies. The seed aids repeatability
but does not guarantee identical results across hardware, software versions,
or nondeterministic operations. MNIST accuracy does not establish accuracy on
other handwriting.

Run the network-independent tests with:

```sh
uv run --locked python -m pytest -q
```
