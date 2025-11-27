"""
Brain Tumor MRI Classification - Project Workflow Diagram

This file provides a visual overview of the complete project workflow.
"""

PROJECT_WORKFLOW = """

╔═══════════════════════════════════════════════════════════════════════════════╗
║                  BRAIN TUMOR MRI CLASSIFICATION WORKFLOW                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. SETUP & CONFIGURATION                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            ┌──────────────┐                   ┌──────────────┐
            │  config.py   │                   │ Install Deps │
            │              │                   │              │
            │ - Image size │                   │ pip install  │
            │ - Batch size │                   │ requirements │
            │ - Epochs     │                   │              │
            │ - Paths      │                   └──────────────┘
            └──────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. DATA PREPARATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌──────────────────┐              ┌──────────────────┐
          │  data_loader.py  │              │  Organize Data   │
          │                  │              │                  │
          │ - Load datasets  │◄─────────────┤ data/train/...   │
          │ - Normalize      │              │ data/val/...     │
          │ - Augment (train)│              │ data/test/...    │
          │ - Optimize       │              └──────────────────┘
          └──────────────────┘
                    │
                    ▼
          Train, Val, Test Datasets Ready


┌─────────────────────────────────────────────────────────────────────────────┐
│                         3. MODEL BUILDING                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌───────────────────┐              ┌───────────────────┐
          │   Custom CNN      │              │ Transfer Learning │
          │                   │              │                   │
          │ - 5 Conv Blocks   │              │ - EfficientNetB0  │
          │ - BatchNorm       │              │ - ResNet50        │
          │ - Dropout         │              │ - VGG16           │
          │ - Dense Layers    │              │ - Custom Top      │
          └───────────────────┘              └───────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                              models.py compiled


┌─────────────────────────────────────────────────────────────────────────────┐
│                           4. TRAINING                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                              ┌───────┴───────┐
                              │   train.py    │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌───────────────────┐              ┌───────────────────┐
          │  Train Custom CNN │              │ Train Transfer    │
          │                   │              │ Learning Model    │
          │ - Fit on data     │              │                   │
          │ - Callbacks       │              │ - Phase 1: Frozen │
          │   * EarlyStopping │              │ - Phase 2: Fine-  │
          │   * ModelCheckpt  │              │   tuning          │
          │   * ReduceLR      │              │ - Callbacks       │
          └───────────────────┘              └───────────────────┘
                    │                                   │
                    ▼                                   ▼
          custom_cnn.h5                     transfer_model.h5
          Training plots                    Training plots


┌─────────────────────────────────────────────────────────────────────────────┐
│                          5. EVALUATION                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                              ┌───────┴───────┐
                              │ evaluate.py   │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌───────────────────┐              ┌───────────────────┐
          │ Evaluate Custom   │              │ Evaluate Transfer │
          │ CNN               │              │ Learning          │
          │                   │              │                   │
          │ - Test accuracy   │              │ - Test accuracy   │
          │ - Precision       │              │ - Precision       │
          │ - Recall          │              │ - Recall          │
          │ - F1-score        │              │ - F1-score        │
          │ - Confusion matrix│              │ - Confusion matrix│
          └───────────────────┘              └───────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                          ┌───────────────────┐
                          │ Model Comparison  │
                          │                   │
                          │ - Accuracy chart  │
                          │ - F1-score chart  │
                          │ - Parameters      │
                          │ - Recommendation  │
                          └───────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                        6. DEPLOYMENT (WEB APP)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                              ┌───────┴───────┐
                              │    app.py     │
                              │  (Streamlit)  │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          ┌───────────────────┐              ┌───────────────────┐
          │  User Interface   │              │  Model Inference  │
          │                   │              │                   │
          │ - Upload image    │──────────────►│ - Load model      │
          │ - Select model    │              │ - Preprocess      │
          │ - View results    │◄──────────────│ - Predict         │
          │ - Charts          │              │ - Return results  │
          └───────────────────┘              └───────────────────┘
                    │
                    ▼
          ┌───────────────────────────────┐
          │     Prediction Results        │
          │                               │
          │ - Predicted class             │
          │ - Confidence score            │
          │ - Probability distribution    │
          │ - Interpretation guide        │
          └───────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT FILES                                       │
└─────────────────────────────────────────────────────────────────────────────┘

📁 models/
  ├── custom_cnn.h5                      (Trained Custom CNN model)
  └── transfer_model.h5                  (Trained Transfer Learning model)

📁 plots/
  ├── custom_cnn_training_history.png    (Training curves)
  ├── transfer_model_training_history.png (Training curves)
  ├── custom_cnn_confusion_matrix.png    (Test evaluation)
  ├── transfer_model_confusion_matrix.png (Test evaluation)
  ├── model_comparison.png               (Performance comparison)
  └── per_class_comparison.png           (Per-class metrics)

📁 results/
  ├── custom_cnn_history.pkl             (Training history)
  └── transfer_history.pkl               (Training history)


═══════════════════════════════════════════════════════════════════════════════

                            MODULE DEPENDENCIES

config.py ────────────────────► (All modules read configuration)
                │
data_loader.py ─┼───► train.py ───► evaluate.py
                │         │              │
models.py ──────┼─────────┘              │
                │                        │
utils.py ───────┼────────────────────────┘
                │
                └──────────► app.py

═══════════════════════════════════════════════════════════════════════════════

                            KEY TECHNOLOGIES

🐍 Python 3.8+
🧠 TensorFlow 2.13+ / Keras
📊 Matplotlib / Seaborn / Plotly
🔬 Scikit-learn
🖥️ Streamlit
📦 NumPy / Pandas / Pillow

═══════════════════════════════════════════════════════════════════════════════
"""

print(PROJECT_WORKFLOW)

# Additional workflow details
DETAILED_STEPS = """

═══════════════════════════════════════════════════════════════════════════════
                         DETAILED STEP-BY-STEP WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Environment Setup
─────────────────────────
Command: python -m venv venv
Command: .\venv\Scripts\Activate.ps1  (Windows)
Command: pip install -r requirements.txt
Status: ✅ Environment ready

STEP 2: Data Preparation
────────────────────────
Action: Organize MRI images into data/ directory
Structure:
  data/
    ├── train/{glioma, meningioma, pituitary, no_tumor}/
    ├── val/{glioma, meningioma, pituitary, no_tumor}/
    └── test/{glioma, meningioma, pituitary, no_tumor}/
Status: ⏳ User action required

STEP 3: Model Training
─────────────────────
Command: python train.py
Duration: 30-60 minutes (GPU) / 2-4 hours (CPU)
Output:
  - models/custom_cnn.h5
  - models/transfer_model.h5
  - plots/training_history.png (×2)
  - results/history.pkl (×2)
Status: 🔄 Automatic

STEP 4: Model Evaluation
────────────────────────
Command: python evaluate.py
Duration: 5-10 minutes
Output:
  - Test metrics (printed)
  - plots/confusion_matrix.png (×2)
  - plots/model_comparison.png
  - plots/per_class_comparison.png
Status: 🔄 Automatic

STEP 5: Web Application
───────────────────────
Command: streamlit run app.py
Access: http://localhost:8501
Features:
  - Upload MRI image
  - Select model
  - Get prediction
  - View probabilities
Status: 🚀 Interactive

═══════════════════════════════════════════════════════════════════════════════
"""

print(DETAILED_STEPS)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Brain Tumor MRI Classification - Workflow Overview")
    print("="*80)
    print("\nRefer to the ASCII diagrams above for the complete project workflow.")
    print("\nFor detailed instructions, see:")
    print("  - README.md")
    print("  - SETUP_GUIDE.py")
    print("  - QUICK_REFERENCE.md")
    print("\n" + "="*80)
