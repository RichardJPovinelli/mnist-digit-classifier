# MNIST digit classifier

Train a PyTorch convolutional neural network to classify MNIST handwritten
digits from 0 to 9. The project requires Python 3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

## Setup

From this directory, install the locked dependencies:

```sh
uv sync --locked
```

MNIST is downloaded automatically into `data/` on the first training run.
The download requires network access. The official training set is split
into 55,000 training and 5,000 validation examples using a fixed seed; the
10,000 official test examples are held out until final evaluation.

## Train

Choose a new run directory; existing directories are not overwritten.

```sh
uv run --locked python mnist.py train --run-dir runs/my-run --device cpu
```

The default is three epochs, batch size 128, learning rate 0.001, and seed
42. Use `--help` after `train` for options. Use `--device auto` to select
CUDA when available, or `--device cuda` to require it. Training prints
epoch-level loss and accuracy and saves the checkpoint with the highest
validation accuracy.

## Evaluate

After model and training choices are fixed, evaluate the selected checkpoint
once on the held-out test set:

```sh
uv run --locked python mnist.py evaluate --run-dir runs/my-run --device cpu
```

The command reports accuracy (correct predictions divided by test examples)
and a confusion matrix with true digits as rows and predicted digits as
columns. It writes `test-evaluation/report.json` and up to ten
misclassified images under the run directory. Evaluation refuses to
overwrite an existing result.

## Predict

Provide an 8-bit grayscale, 28 x 28 image with the same light-digit-on-dark
background convention as MNIST:

```sh
uv run --locked python mnist.py predict --checkpoint runs/my-run/best.pt --image path/to/digit.png --device cpu
```

The command prints the predicted digit. Incorrect image sizes, image modes,
and missing or invalid checkpoints produce errors.

## Artifacts and reproducibility

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

## Existing general Python instructions

The following pre-existing instructions describe a different program using
`main.py`, `requirements.txt`, and Python 3.15. They do not apply to the MNIST
classifier above.

# Run the Python program

These instructions use Python 3.15 on Windows and macOS. Install Git and uv once, then follow the same clone and run steps on either platform.

## One-time setup

### Windows (PowerShell)

Open **PowerShell** from the Start menu and run these commands. Skip any tool you already have.

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
```

Close and reopen PowerShell after installing. If `winget` is unavailable, see the [Git for Windows installer](https://git-scm.com/install/win) and the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### macOS (Terminal)

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

## Clone and run on either platform

1. **Copy the clone URL from GitHub.** Open the repository's main page in your browser. If you are viewing this README as a separate file, click the repository name near the top of the page to return to the file list. Click the green **Code** button above the files, select **HTTPS**, and click the copy icon next to the URL. [GitHub’s illustrated cloning guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

2. **Clone the repository.** Open PowerShell on Windows or Terminal on macOS and go to the folder where you want to keep your coursework. Paste the URL you copied in place of `https://github.com/OWNER/REPOSITORY.git` in the first command. Run the second command after cloning finishes.

   ```sh
   git clone https://github.com/OWNER/REPOSITORY.git intro-ai-assignment
   cd intro-ai-assignment
   ```

   If you already cloned the repository, open that folder in PowerShell or Terminal and go straight to step 3.

3. From the folder containing `main.py` and `requirements.txt`, run:

   ```sh
   uv run --python 3.15 --with-requirements requirements.txt python main.py
   ```

   On first use, uv downloads Python 3.15 if needed and installs the packages listed in `requirements.txt`. To run the program again, open the same folder and repeat the command.

   If the program uses only the Python standard library and has no `requirements.txt`, run:

   ```sh
   uv run --python 3.15 python main.py
   ```

## Check your setup

If you have trouble, run these commands one at a time:

```sh
git --version
uv --version
uv run --python 3.15 python --version
```

The last command should print `Python 3.15.x`, where `x` is the patch version. If the program reports that it cannot find `main.py` or `requirements.txt`, check that you are in the assignment folder and that the filenames match those in the repository.
