# Brain Tumor MRI Classification - Quick Reference

## 🚀 Quick Start Commands

### 1. Setup Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Place your MRI images in the `data/` directory following this structure:
- `data/train/{glioma, meningioma, pituitary, no_tumor}/`
- `data/val/{glioma, meningioma, pituitary, no_tumor}/`
- `data/test/{glioma, meningioma, pituitary, no_tumor}/`

### 3. Train Models
```powershell
python train.py
```

### 4. Evaluate Models
```powershell
python evaluate.py
```

### 5. Run Web App
```powershell
streamlit run app.py
```

---

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Configuration and hyperparameters |
| `data_loader.py` | Dataset loading and preprocessing |
| `models.py` | Model architectures (Custom CNN & Transfer Learning) |
| `train.py` | Training pipeline |
| `evaluate.py` | Model evaluation and metrics |
| `app.py` | Streamlit web application |
| `utils.py` | Helper functions |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `SETUP_GUIDE.py` | Detailed setup instructions |

---

## ⚙️ Key Configuration (config.py)

```python
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 4
PRETRAINED_MODEL = 'EfficientNetB0'  # or 'ResNet50', 'VGG16'
```

---

## 📊 Model Architectures

### Custom CNN
- 5 Conv blocks (32→64→128→256→512 filters)
- Batch normalization + Dropout
- 2 Dense layers (512, 256)
- ~12M parameters

### Transfer Learning
- Pretrained backbone (EfficientNetB0/ResNet50/VGG16)
- Custom top layers
- Fine-tuning support
- ~5-25M parameters

---

## 🎯 Expected Outputs

### After Training:
- `models/custom_cnn.h5` - Trained Custom CNN
- `models/transfer_model.h5` - Trained Transfer Learning model
- `plots/custom_cnn_training_history.png` - Training curves
- `plots/transfer_model_training_history.png` - Training curves

### After Evaluation:
- `plots/custom_cnn_confusion_matrix.png` - Confusion matrix
- `plots/transfer_model_confusion_matrix.png` - Confusion matrix
- `plots/model_comparison.png` - Model comparison
- `plots/per_class_comparison.png` - Per-class metrics

---

## 🔧 Troubleshooting

### Out of Memory
```python
# In config.py, reduce batch size:
BATCH_SIZE = 16  # or 8
```

### Slow Training
- Use GPU (10x faster)
- Reduce EPOCHS for testing
- Use transfer learning

### Data Not Found
- Check `data/` directory structure
- Ensure images are in correct folders

### Streamlit Won't Start
```powershell
# Check if port 8501 is available
streamlit run app.py --server.port 8502
```

---

## 💡 Tips for Best Results

1. **Use GPU** for training (5-10x speedup)
2. **Start with Transfer Learning** (usually better accuracy)
3. **Balance your dataset** across classes
4. **Monitor validation loss** to prevent overfitting
5. **Try different augmentation** parameters
6. **Fine-tune hyperparameters** based on results

---

## 📈 Performance Expectations

| Model | Typical Accuracy | Training Time (GPU) |
|-------|-----------------|---------------------|
| Custom CNN | 90-95% | 20-40 min |
| Transfer Learning | 95-98% | 30-60 min |

*Results vary based on dataset size and quality*

---

## 🎨 Streamlit App Features

- Upload MRI images (JPG, PNG)
- Select model (Custom CNN or Transfer Learning)
- View prediction with confidence
- Interactive probability charts
- Interpretation guide

---

## ⚠️ Important Notes

- **Educational/Research purposes only**
- **NOT for clinical diagnosis**
- Always consult medical professionals

---

## 📞 Need Help?

1. Check `README.md` for full documentation
2. Review `SETUP_GUIDE.py` for detailed steps
3. Test individual modules:
   ```powershell
   python data_loader.py  # Test data loading
   python models.py       # Test model creation
   ```

---

## 🎓 Learning Resources

- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Streamlit: https://docs.streamlit.io/
- Deep Learning: https://www.deeplearningbook.org/

---

**Happy Coding! 🚀**
