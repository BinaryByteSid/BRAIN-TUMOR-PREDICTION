# 🔧 MODEL BIAS FIX GUIDE

## Problem: Model Only Predicts 2 Classes (no_tumor & pituitary)

This is a common issue in multi-class classification. I've implemented several fixes.

---

## ✅ What Has Been Fixed

### 1. **Class Weights** (Most Important)
- Added automatic class weight calculation
- Handles imbalanced datasets
- Gives more importance to underrepresented classes

### 2. **Improved Model Architecture**
- Added L2 regularization to prevent overfitting to majority classes
- Added extra dense layer (128 neurons) for better feature learning
- Adjusted dropout rates for better generalization (0.2 → 0.3 → 0.4 → 0.5)

### 3. **Better Training Strategy**
- **Lower learning rate**: 0.001 → 0.0005 (more stable convergence)
- **More epochs**: 25 → 30 (better learning)
- **Less aggressive augmentation**: Prevents model confusion

### 4. **Enhanced Callbacks**
- Class weights used in both training phases
- Better monitoring of validation loss

---

## 🚀 How to Fix Your Model

### Step 1: Run Diagnostic (Check Current Issue)
```powershell
python diagnose_model.py
```

This will show:
- Which classes are being predicted
- Data balance issues
- Per-class accuracy
- Confidence levels

### Step 2: Retrain Models (with fixes)
```powershell
python train.py
```

**Important Changes:**
- Class weights are now automatically calculated and used
- Training will balance learning across all 4 classes
- Better regularization prevents overfitting

### Step 3: Re-evaluate
```powershell
python evaluate.py
```

Check if all 4 classes are now being predicted correctly.

---

## 📊 Common Causes & Solutions

### Cause 1: **Imbalanced Dataset**
**Symptoms:** Some classes have many more samples than others

**Solution:** ✅ **FIXED** - Class weights automatically calculated
```python
# Now automatically done in train.py
class_weights = create_class_weight_dict(train_dataset)
```

### Cause 2: **Model Overfitting to Majority Classes**
**Symptoms:** High training accuracy, but only predicts common classes

**Solutions:** ✅ **FIXED**
- Added L2 regularization
- Adjusted dropout rates
- Added extra dense layer

### Cause 3: **Learning Rate Too High**
**Symptoms:** Model converges to local minima (only 2 classes)

**Solution:** ✅ **FIXED** - Reduced from 0.001 to 0.0005

### Cause 4: **Insufficient Training**
**Symptoms:** Model hasn't learned to distinguish all classes

**Solution:** ✅ **FIXED** - Increased epochs to 30

### Cause 5: **Too Much Augmentation**
**Symptoms:** Augmentation makes images too different from originals

**Solution:** ✅ **FIXED** - Reduced augmentation intensity

---

## 🔍 Verify Your Data First

Before retraining, check your dataset:

```powershell
python diagnose_model.py
```

Look for:

### ✓ **Balanced Classes:**
```
Train Dataset:
  glioma         :  800 samples (25.00%)  ✓
  meningioma     :  780 samples (24.37%)  ✓
  no_tumor       :  820 samples (25.62%)  ✓
  pituitary      :  800 samples (25.00%)  ✓
```

### ⚠ **Imbalanced Classes:**
```
Train Dataset:
  glioma         :  200 samples (10.00%)  ⚠
  meningioma     :  150 samples (7.50%)   ⚠
  no_tumor       : 1000 samples (50.00%)  ⚠ TOO MANY
  pituitary      :  650 samples (32.50%)  ⚠ TOO MANY
```

**If imbalanced:** The class weights will automatically fix this during training.

---

## 📈 Expected Results After Fix

After retraining with the fixes:

### Before (Problem):
```
Predicted Labels Distribution:
  glioma         :    0 samples (0.00%)   ✗
  meningioma     :    0 samples (0.00%)   ✗
  no_tumor       :  150 samples (50.00%)  
  pituitary      :  150 samples (50.00%)  
```

### After (Fixed):
```
Predicted Labels Distribution:
  glioma         :   72 samples (24.00%)  ✓
  meningioma     :   68 samples (22.67%)  ✓
  no_tumor       :   78 samples (26.00%)  ✓
  pituitary      :   82 samples (27.33%)  ✓
```

---

## 🎯 Step-by-Step Fix Process

### Complete Fix Workflow:

```powershell
# 1. Diagnose current issue
python diagnose_model.py

# 2. Backup old models (optional)
# Move models/custom_cnn.h5 and models/transfer_model.h5 to backup folder

# 3. Retrain with fixes
python train.py
# This will now use:
# - Class weights
# - Better architecture
# - Lower learning rate
# - More epochs

# 4. Evaluate new models
python evaluate.py

# 5. Test in web app
streamlit run app.py
```

---

## 🔧 Additional Tips

### If Still Having Issues:

1. **Check Your Data Distribution:**
   ```powershell
   python diagnose_model.py
   ```
   Ensure each class has at least 100+ samples

2. **Increase Training Epochs:**
   Edit `config.py`:
   ```python
   EPOCHS = 40  # Increase from 30
   ```

3. **Adjust Learning Rate:**
   Edit `config.py`:
   ```python
   LEARNING_RATE = 0.0003  # Even lower
   ```

4. **Use Only Transfer Learning:**
   Transfer learning often works better for medical images
   Focus on the `transfer_model.h5`

5. **Balance Your Dataset Manually:**
   - Collect more samples for underrepresented classes
   - Or remove samples from overrepresented classes

---

## 📊 Monitoring During Training

Watch for these signs in training output:

### ✓ Good Signs:
```
Epoch 10/30
loss: 0.8234 - accuracy: 0.6800 - val_loss: 0.7156 - val_accuracy: 0.7200
Class weights: {0: 1.2, 1: 1.3, 2: 0.8, 3: 0.9}  ← Being used!
```

### ⚠ Warning Signs:
```
Epoch 10/30
loss: 0.3234 - accuracy: 0.9500 - val_loss: 1.8156 - val_accuracy: 0.5000
                                           ^^^^^ Much higher - overfitting!
```

---

## 🎓 Understanding the Fixes

### What Class Weights Do:
```python
# Without class weights (problem):
# Model learns: "90% accuracy by just predicting 'no_tumor'!"

# With class weights (fix):
# Model learns: "Must predict ALL classes correctly!"
```

### What L2 Regularization Does:
```python
# Prevents: Dense(512) from having extreme weights
# Result: Better generalization to all classes
```

### Why Lower Learning Rate Helps:
```python
# High LR (0.001): Takes big steps → misses optimal solution
# Low LR (0.0005): Takes small steps → finds better solution
```

---

## ✅ Checklist Before Retraining

- [ ] Backup current models (if you want to keep them)
- [ ] Run `diagnose_model.py` to see current issue
- [ ] Verify data directory has all 4 classes
- [ ] Check that each class has at least 50+ images
- [ ] Ensure enough disk space for new models
- [ ] Close any programs using the GPU

---

## 🎉 After Successful Retaining

You should see in `evaluate.py` output:

```
Per-Class Metrics:
  glioma         : F1-Score: 0.88+ ✓
  meningioma     : F1-Score: 0.85+ ✓
  no_tumor       : F1-Score: 0.92+ ✓
  pituitary      : F1-Score: 0.89+ ✓
```

All classes with good F1-scores means the fix worked! 🎉

---

## 📞 Still Having Issues?

If after retraining you still see bias:

1. Share the output of `python diagnose_model.py`
2. Check your dataset - you may need more balanced data
3. Try training with only transfer learning model
4. Consider collecting more data for underrepresented classes

---

**The fixes are already in the code. Just run `python train.py` to retrain!** 🚀
