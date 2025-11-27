"""
Configuration file for Brain Tumor MRI Classification Project.
Contains all global constants, paths, and hyperparameters.
"""

import os

# ============================================
# IMAGE PARAMETERS
# ============================================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# ============================================
# MODEL PARAMETERS
# ============================================
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005  # Lower learning rate for better convergence

# ============================================
# DATA PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_DIR_TRAIN = os.path.join(DATA_DIR, 'train')
DATA_DIR_VAL = os.path.join(DATA_DIR, 'val')
DATA_DIR_TEST = os.path.join(DATA_DIR, 'test')

# ============================================
# OUTPUT PATHS
# ============================================
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# MODEL SAVE PATHS
# ============================================
CUSTOM_CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'custom_cnn.h5')
TRANSFER_MODEL_PATH = os.path.join(MODELS_DIR, 'transfer_model.h5')
CUSTOM_CNN_HISTORY_PATH = os.path.join(RESULTS_DIR, 'custom_cnn_history.pkl')
TRANSFER_HISTORY_PATH = os.path.join(RESULTS_DIR, 'transfer_history.pkl')

# ============================================
# TRAINING CALLBACKS
# ============================================
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ============================================
# DATA AUGMENTATION PARAMETERS
# ============================================
ROTATION_RANGE = 15
ZOOM_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.95, 1.05]  # Less aggressive brightness adjustment

# ============================================
# TRANSFER LEARNING PARAMETERS
# ============================================
PRETRAINED_MODEL = 'EfficientNetB0'  # Options: 'EfficientNetB0', 'ResNet50', 'VGG16'
FINE_TUNE_LAYERS = 20  # Number of top layers to unfreeze for fine-tuning

# ============================================
# RANDOM SEED
# ============================================
RANDOM_SEED = 42
