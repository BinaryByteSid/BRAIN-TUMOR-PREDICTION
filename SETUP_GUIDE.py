"""
Quick Setup and Testing Guide for Brain Tumor MRI Classification Project
"""

# STEP-BY-STEP SETUP GUIDE
# ========================

# 1. CREATE VIRTUAL ENVIRONMENT (Windows PowerShell)
# ---------------------------------------------------
# python -m venv venv
# .\venv\Scripts\Activate.ps1

# 2. INSTALL DEPENDENCIES
# ------------------------
# pip install -r requirements.txt

# 3. PREPARE YOUR DATASET
# ------------------------
# Organize your MRI images in this structure:
#
# data/
# ├── train/
# │   ├── glioma/
# │   ├── meningioma/
# │   ├── pituitary/
# │   └── no_tumor/
# ├── val/
# │   ├── glioma/
# │   ├── meningioma/
# │   ├── pituitary/
# │   └── no_tumor/
# └── test/
#     ├── glioma/
#     ├── meningioma/
#     ├── pituitary/
#     └── no_tumor/

# 4. TEST DATA LOADER (OPTIONAL)
# -------------------------------
# python data_loader.py

# 5. TEST MODEL ARCHITECTURES (OPTIONAL)
# ---------------------------------------
# python models.py

# 6. TRAIN MODELS
# ---------------
# python train.py
# 
# This will train both:
# - Custom CNN model
# - Transfer Learning model (EfficientNetB0)
#
# Training outputs:
# - models/custom_cnn.h5
# - models/transfer_model.h5
# - plots/custom_cnn_training_history.png
# - plots/transfer_model_training_history.png

# 7. EVALUATE MODELS
# -------------------
# python evaluate.py
#
# This will generate:
# - Test accuracy, precision, recall, F1-scores
# - Confusion matrices
# - Model comparison plots

# 8. RUN STREAMLIT APP
# ---------------------
# streamlit run app.py
#
# This will open a web browser with the interactive app
# where you can upload MRI images and get predictions

# TROUBLESHOOTING
# ===============

# Issue: No GPU detected
# Solution: Install CUDA and cuDNN, or train on CPU (slower)

# Issue: Out of memory
# Solution: Reduce BATCH_SIZE in config.py (e.g., 16 or 8)

# Issue: Data not found
# Solution: Check that data/ directory exists with correct structure

# Issue: Streamlit won't start
# Solution: Make sure port 8501 is not in use

# CUSTOMIZATION OPTIONS
# =====================

# Edit config.py to customize:
# - IMAGE_HEIGHT, IMAGE_WIDTH (default: 224)
# - BATCH_SIZE (default: 32)
# - EPOCHS (default: 25)
# - LEARNING_RATE (default: 0.001)
# - PRETRAINED_MODEL ('EfficientNetB0', 'ResNet50', 'VGG16')
# - Data augmentation parameters

# PROJECT FILE OVERVIEW
# =====================

# config.py          - All configuration and hyperparameters
# data_loader.py     - Dataset loading and preprocessing
# models.py          - Custom CNN and Transfer Learning models
# train.py           - Complete training pipeline
# evaluate.py        - Model evaluation and metrics
# app.py             - Streamlit web application
# utils.py           - Helper functions (plotting, etc.)
# requirements.txt   - Python dependencies

# EXPECTED OUTPUTS
# ================

# After training:
# ✓ models/custom_cnn.h5
# ✓ models/transfer_model.h5
# ✓ plots/custom_cnn_training_history.png
# ✓ plots/transfer_model_training_history.png
# ✓ results/custom_cnn_history.pkl
# ✓ results/transfer_history.pkl

# After evaluation:
# ✓ plots/custom_cnn_confusion_matrix.png
# ✓ plots/transfer_model_confusion_matrix.png
# ✓ plots/model_comparison.png
# ✓ plots/per_class_comparison.png

# QUICK COMMANDS SUMMARY
# ======================

"""
# Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python evaluate.py
streamlit run app.py
"""

# PERFORMANCE TIPS
# ================

# 1. Use GPU for faster training (5-10x speedup)
# 2. Start with smaller EPOCHS (10-15) for testing
# 3. Use transfer learning for better accuracy with less data
# 4. Monitor validation loss to prevent overfitting
# 5. Try different pretrained models (EfficientNetB0 usually best)

# NEXT STEPS
# ==========

# After successful setup:
# 1. Train on your dataset
# 2. Evaluate performance
# 3. Compare models
# 4. Deploy best model with Streamlit
# 5. (Optional) Fine-tune hyperparameters for better results

print("Setup guide loaded. Follow the steps above to set up and run the project!")
print("\nQuick Start:")
print("1. pip install -r requirements.txt")
print("2. Organize your data in data/ directory")
print("3. python train.py")
print("4. python evaluate.py")
print("5. streamlit run app.py")
