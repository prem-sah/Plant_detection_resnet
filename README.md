# Plant detection (ResNet) — README

This repository contains code to train a ResNet-based classifier on the PlantVillage dataset and to run predictions against a trained model.

Files of primary interest
- `main.py` — training script. It builds a ResNet50-based model, trains it using `ImageDataGenerator`, saves the model to `plant_disease_model.keras`, stores training history to `training_history.pkl`, and writes class indices to `class_indices.json`.
- `predict.py` — prediction script. Loads `plant_disease_model.keras` and `class_indices.json` and prints predictions for a single image or all images under a folder.
- `class_indices.json` — mapping from class name -> integer label (created by training).
- `training_history.pkl` — pickled training history (loss/accuracy per epoch) created after training.

Quick overview
- Training expects the dataset in a directory structure compatible with `flow_from_directory` (one sub-folder per class). The current `main.py` uses the path:

  /Users/sushilkumarpatel/Desktop/plant_detection_prem/Plant_disease_detection_resne/archive (2)/PlantVillage

  You can change `dataset_path` at the top of `main.py` to point to your local dataset location.

- The training script saves the model to `plant_disease_model.keras` in the project root. `predict.py` loads this file by default.

Environment setup (macOS, zsh)

1) Create and activate a Python virtual environment (recommended Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Upgrade pip and install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Apple Silicon (M1/M2) notes

- If you are on Apple Silicon (M1/M2) and want to use the macOS-optimized TensorFlow builds (Metal acceleration), you can install:

```bash
pip install tensorflow-macos tensorflow-metal
```

- If you install the macOS packages, either remove or comment out the `tensorflow` entry in `requirements.txt` to avoid conflicting installs, or install the macOS packages into the environment after creating it.

Running training

Open a terminal inside the project and (optionally) activate the virtual environment, then run:

```bash
# start in the project directory
cd /Users/sushilkumarpatel/Desktop/plant_detection_prem/Plant_disease_detection_resne
python main.py
```

Notes:
- `main.py` uses `ImageDataGenerator(..., validation_split=0.2)` so it will use 80% of images for training and 20% for validation.
- By default training runs for 10 epochs. Edit `train_model()` in `main.py` to change `epochs`, `BATCH_SIZE`, or other hyperparameters.
- After successful training you will find:
  - `plant_disease_model.keras` — the saved model
  - `training_history.pkl` — pickled history dictionary
  - `class_indices.json` — mapping of class name -> index used by the model

Running predictions

`predict.py` loads `plant_disease_model.keras` and `class_indices.json` and accepts a `--path` argument which can be either a single image file or a directory. Example usages:

Predict a single image:

```bash
python predict.py --path /path/to/image.jpg
```

Predict all images in a directory:

```bash
python predict.py --path /path/to/folder_with_images
```

If you omit `--path` it uses the default path defined in `predict.py`:

```python
DEFAULT_PREDICT_PATH = "/Users/sushilkumarpatel/Desktop/plant_detection_prem/Plant_disease_detection_resne/dataset_test"
```

Output:
- For each image the script prints the predicted class name, class index, and confidence (percentage).

Inspecting or using `class_indices.json`

The JSON file maps class names to integer labels (example: `{ "Potato___Early_blight": 0, "Potato___healthy": 1, ... }`). `predict.py` inverts that mapping to print the class name for a predicted index.

If you want to use the model programmatically from another script, load the model with `tf.keras.models.load_model("plant_disease_model.keras")` and use `model.predict()` on preprocessed image batches (resize to 224x224 and rescale pixel values to [0,1]).

Troubleshooting

- Model file not found: `predict.py` expects `plant_disease_model.keras` in the project root. If you haven't trained the model yet, run `python main.py` or place a compatible saved model file at that path.
- Different class mapping: If `class_indices.json` does not match the model you loaded, predictions will show incorrect labels. Always use the `class_indices.json` produced by the model's training run.
- macOS / Apple Silicon: Installing `tensorflow-macos` / `tensorflow-metal` is recommended for hardware acceleration. If TensorFlow import fails, try installing the macOS wheel or use a CPU-only install. See TensorFlow macOS instructions on the TensorFlow website.
- Large dataset / memory: If training runs out of GPU memory, lower `BATCH_SIZE` in `main.py` or enable mixed precision (advanced).

Git and pushing changes

- If you need to configure git user identity before committing, run:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

- If you see remote/merge issues (unrelated histories) when pulling/pushing, create a backup branch first and then choose to merge or reset. Example:

```bash
git checkout -b local-backup
# then either merge remote with --allow-unrelated-histories, or reset to remote
```

Development notes / where to change code

- Dataset path: edit `dataset_path` in `main.py`.
- Default predict folder: edit `DEFAULT_PREDICT_PATH` in `predict.py` or pass `--path`.
- Image size and batch size: change `IMG_SIZE` and `BATCH_SIZE` constants in `main.py` and `predict.py` to match a different input shape.

License / credits

This repository uses the PlantVillage dataset structure. See the original dataset license/source for usage terms.

If you'd like, I can:
- add a `requirements.txt` check that includes pinned versions for macOS, or
- add a small wrapper script to run training/prediction with safer paths and flags.
