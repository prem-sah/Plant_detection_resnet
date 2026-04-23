Env setup (macOS, zsh)

1) Create and activate a Python virtual environment (recommended Python 3.10+):

python3 -m venv .venv
source .venv/bin/activate

2) Upgrade pip and install requirements:

pip install --upgrade pip
pip install -r requirements.txt

3) Apple Silicon (M1/M2) notes
- If you're on macOS with Apple Silicon and want Metal GPU acceleration, install:
  pip install tensorflow-macos tensorflow-metal
- Then comment out `tensorflow` in requirements.txt or install the macOS packages into the environment after creating it.

4) Quick smoke test (build the model):

python -c "import tensorflow as tf; from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)); print('ResNet built OK')"
