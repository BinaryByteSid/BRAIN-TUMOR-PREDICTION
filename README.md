# 🧠 Brain Tumor MRI Image Classification

A comprehensive deep learning project for classifying brain tumor MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**. This project includes custom CNN and transfer learning models, complete training pipeline, evaluation metrics, and an interactive Streamlit web application for inference.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Streamlit App](#-streamlit-app)
- [Important Notice](#-important-notice)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project implements an end-to-end deep learning pipeline for brain tumor classification using MRI images. It features:

- **Two model architectures**: Custom CNN and Transfer Learning (EfficientNetB0/ResNet50/VGG16)
- **Complete training pipeline** with data augmentation and callbacks
- **Comprehensive evaluation** with confusion matrices and classification reports
- **Interactive web application** for easy inference
- **Modular, production-ready code** structure

The project is designed for educational and research purposes, demonstrating best practices in medical image classification with deep learning.

---

## 📊 Dataset Description

The dataset contains brain MRI scans organized into four classes:

### Classes:
1. **Glioma Tumor** - Most common and aggressive brain tumor in adults
2. **Meningioma Tumor** - Usually benign tumor in the meninges
3. **Pituitary Tumor** - Tumor in the pituitary gland
4. **No Tumor** - Normal brain MRI scans

### Expected Directory Structure:
```
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

> **Note**: You need to organize your MRI dataset following this structure before training.

---

## ✨ Features

### 🔬 **Data Processing**
- Automated data loading with `tf.keras.preprocessing.image_dataset_from_directory`
- Pixel normalization to [0, 1] range
- Data augmentation (rotation, zoom, flip, translation, brightness adjustment)
- Efficient data pipeline with caching and prefetching

### 🧠 **Model Architectures**

#### Custom CNN:
- 5 convolutional blocks with batch normalization
- Progressive filter sizes: 32 → 64 → 128 → 256 → 512
- Dropout layers for regularization
- Dense layers with 512 and 256 neurons
- Softmax output for 4 classes

#### Transfer Learning:
- Pre-trained backbones (EfficientNetB0, ResNet50, VGG16)
- Global average pooling
- Custom dense layers
- Fine-tuning capability

### 📈 **Training Features**
- Early stopping with patience
- Model checkpointing (save best weights)
- Learning rate reduction on plateau
- Training history visualization
- Support for both CPU and GPU

### 📊 **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- Per-class metrics
- Confusion matrices
- Model comparison plots
- Classification reports

### 🖥️ **Streamlit Web App**
- User-friendly interface
- Drag-and-drop image upload
- Real-time prediction
- Interactive probability charts
- Model selection (Custom CNN or Transfer Learning)
- Detailed interpretation guide

---

## 📁 Project Structure

```
TASK1/
│
├── config.py                  # Global configuration and hyperparameters
├── data_loader.py             # Dataset loading and preprocessing
├── models.py                  # Model architectures (Custom CNN & Transfer Learning)
├── train.py                   # Training pipeline for both models
├── evaluate.py                # Evaluation and metrics computation
├── app.py                     # Streamlit web application
├── utils.py                   # Helper functions (plotting, saving, etc.)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── data/                      # Dataset directory (you need to add your data)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                    # Saved trained models (.h5 files)
│   ├── custom_cnn.h5
│   └── transfer_model.h5
│
├── plots/                     # Training history and evaluation plots
│   ├── custom_cnn_training_history.png
│   ├── transfer_model_training_history.png
│   ├── custom_cnn_confusion_matrix.png
│   ├── transfer_model_confusion_matrix.png
│   ├── model_comparison.png
│   └── per_class_comparison.png
│
└── results/                   # Training history pickles
    ├── custom_cnn_history.pkl
    └── transfer_history.pkl
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Step 1: Clone or Download the Project
```bash
cd TASK1
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Your Dataset
Organize your brain MRI images following the directory structure shown in the [Dataset Description](#-dataset-description) section.

---

## 💻 Usage

### 1️⃣ **Configure Settings** (Optional)
Edit `config.py` to customize hyperparameters:
- Image size, batch size, epochs
- Learning rate, data augmentation parameters
- Model architecture selection (EfficientNetB0, ResNet50, VGG16)

### 2️⃣ **Train Models**
Train both Custom CNN and Transfer Learning models:
```bash
python train.py
```

This will:
- Load and preprocess datasets
- Train Custom CNN model
- Train Transfer Learning model with fine-tuning
- Save best model weights to `models/` directory
- Generate training history plots in `plots/` directory

**Training Time:** Varies based on dataset size and hardware (GPU recommended)

### 3️⃣ **Evaluate Models**
Evaluate trained models on test data:
```bash
python evaluate.py
```

This will:
- Load saved models
- Compute test metrics (accuracy, precision, recall, F1-score)
- Generate confusion matrices
- Create model comparison plots
- Display recommended model

### 4️⃣ **Run Streamlit App**
Launch the interactive web application:
```bash
streamlit run app.py
```

This will:
- Start a local web server
- Open the app in your browser (usually http://localhost:8501)
- Allow you to upload MRI images and get predictions

### 5️⃣ **Test Individual Modules** (Optional)
Test individual components:
```bash
python data_loader.py    # Test data loading
python models.py         # Test model architectures
```

---

## 🏗️ Model Architectures

### Custom CNN Architecture

```
Input (224, 224, 3)
    ↓
[Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)]
    ↓
[Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)]
    ↓
[Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.3)]
    ↓
[Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.3)]
    ↓
[Conv2D(512) → BatchNorm → Conv2D(512) → BatchNorm → MaxPool → Dropout(0.4)]
    ↓
Flatten
    ↓
[Dense(512) → BatchNorm → Dropout(0.5)]
    ↓
[Dense(256) → BatchNorm → Dropout(0.5)]
    ↓
Dense(4, softmax)
```

**Total Parameters:** ~10-15 million (depending on input size)

### Transfer Learning Architecture

```
Input (224, 224, 3)
    ↓
Pretrained Backbone (EfficientNetB0/ResNet50/VGG16)
    ↓
GlobalAveragePooling2D
    ↓
[BatchNorm → Dropout(0.5)]
    ↓
[Dense(512) → BatchNorm → Dropout(0.5)]
    ↓
[Dense(256) → BatchNorm → Dropout(0.4)]
    ↓
Dense(4, softmax)
```

**Training Strategy:**
1. **Phase 1:** Train with frozen backbone (transfer learning)
2. **Phase 2:** Fine-tune top layers of backbone

---

## 📈 Results

Results will vary based on your dataset. After training and evaluation, you'll see:

### Example Metrics:
| Model | Test Accuracy | F1-Score | Parameters |
|-------|--------------|----------|------------|
| Custom CNN | ~92-95% | ~0.92-0.95 | ~12M |
| Transfer Learning | ~95-98% | ~0.95-0.98 | ~5-25M |

### Generated Visualizations:
- ✅ Training/Validation accuracy curves
- ✅ Training/Validation loss curves
- ✅ Confusion matrices
- ✅ Per-class F1-score comparison
- ✅ Overall model comparison

All plots are saved in the `plots/` directory.

---

## 🖥️ Streamlit App

The Streamlit application provides an intuitive interface for using your trained models:

### Features:
- 📤 **Upload MRI images** (JPG, PNG)
- 🎯 **Get instant predictions** with confidence scores
- 📊 **View probability distribution** across all classes
- 🔄 **Switch between models** (Custom CNN or Transfer Learning)
- 💡 **Interpretation guide** for each tumor type

### Screenshots:
The app displays:
1. Uploaded image preview
2. Preprocessed image (224x224)
3. Predicted class with confidence
4. Interactive probability chart
5. Detailed probability breakdown

---

## ⚠️ Important Notice

**DISCLAIMER: This project is for educational and research purposes ONLY.**

- ❌ **NOT for clinical use** or medical diagnosis
- ❌ **NOT a substitute** for professional medical advice
- ❌ **NOT validated** for clinical decision-making

Always consult qualified healthcare professionals for medical diagnosis and treatment.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement:
- Add more model architectures (ResNeXt, DenseNet, Vision Transformers)
- Implement Grad-CAM visualization
- Add more data augmentation techniques
- Support for 3D MRI volumes
- Docker containerization
- Model quantization for deployment

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📚 References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- EfficientNet Paper: https://arxiv.org/abs/1905.11946
- Streamlit Documentation: https://docs.streamlit.io/

---

## 👨‍💻 Author

Created as a comprehensive deep learning project for brain tumor classification.

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Streamlit team for the web app framework
- The open-source community for various tools and libraries

---

**Happy Learning! 🚀**
