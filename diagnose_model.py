"""
Diagnostic script to identify model prediction issues.
Run this to check if your model has bias toward certain classes.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
from data_loader import prepare_datasets

def analyze_predictions(model_path, dataset, class_names):
    """
    Analyze model predictions to detect bias.
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {os.path.basename(model_path)}")
    print(f"{'='*70}\n")
    
    # Load model
    try:
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Collect predictions
    all_true = []
    all_pred = []
    all_probs = []
    
    print("Collecting predictions...")
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        all_probs.extend(predictions)
        all_pred.extend(np.argmax(predictions, axis=1))
        all_true.extend(labels.numpy())
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)
    
    # Analyze class distribution
    print("\n" + "-"*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("-"*70)
    
    print("\nTrue Labels Distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(all_true == i)
        percentage = (count / len(all_true)) * 100
        print(f"  {class_name:<15} : {count:4d} samples ({percentage:5.2f}%)")
    
    print("\nPredicted Labels Distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(all_pred == i)
        percentage = (count / len(all_pred)) * 100
        print(f"  {class_name:<15} : {count:4d} samples ({percentage:5.2f}%)")
    
    # Check if model is biased
    print("\n" + "-"*70)
    print("BIAS DETECTION")
    print("-"*70 + "\n")
    
    unique_predictions = np.unique(all_pred)
    if len(unique_predictions) < config.NUM_CLASSES:
        missing_classes = [i for i in range(config.NUM_CLASSES) if i not in unique_predictions]
        print(f"⚠ WARNING: Model never predicts {len(missing_classes)} class(es)!")
        print(f"   Missing classes: {[class_names[i] for i in missing_classes]}")
        print(f"   This indicates model bias!")
    else:
        print("✓ Model predicts all classes")
    
    # Analyze prediction confidence
    print("\n" + "-"*70)
    print("PREDICTION CONFIDENCE")
    print("-"*70 + "\n")
    
    max_probs = np.max(all_probs, axis=1)
    print(f"Average confidence: {np.mean(max_probs):.4f}")
    print(f"Min confidence: {np.min(max_probs):.4f}")
    print(f"Max confidence: {np.max(max_probs):.4f}")
    
    # Check per-class confidence
    print("\nPer-Class Average Confidence:")
    for i, class_name in enumerate(class_names):
        class_mask = all_pred == i
        if np.sum(class_mask) > 0:
            avg_conf = np.mean(max_probs[class_mask])
            print(f"  {class_name:<15} : {avg_conf:.4f}")
        else:
            print(f"  {class_name:<15} : N/A (never predicted)")
    
    # Analyze per-class accuracy
    print("\n" + "-"*70)
    print("PER-CLASS ACCURACY")
    print("-"*70 + "\n")
    
    for i, class_name in enumerate(class_names):
        class_mask = all_true == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum((all_true == i) & (all_pred == i))
            class_total = np.sum(class_mask)
            accuracy = class_correct / class_total
            print(f"  {class_name:<15} : {accuracy:.4f} ({class_correct}/{class_total})")
        else:
            print(f"  {class_name:<15} : N/A (no samples)")
    
    # Overall accuracy
    overall_accuracy = np.sum(all_true == all_pred) / len(all_true)
    print(f"\n  Overall Accuracy: {overall_accuracy:.4f}")
    
    return {
        'predictions': all_pred,
        'true_labels': all_true,
        'probabilities': all_probs
    }


def check_data_balance():
    """
    Check if training data is balanced.
    """
    print(f"\n{'='*70}")
    print("CHECKING DATA BALANCE")
    print(f"{'='*70}\n")
    
    try:
        train_ds, val_ds, test_ds = prepare_datasets()
        
        for dataset_name, dataset in [('Train', train_ds), ('Validation', val_ds), ('Test', test_ds)]:
            print(f"\n{dataset_name} Dataset:")
            labels = []
            for _, batch_labels in dataset:
                labels.extend(batch_labels.numpy())
            labels = np.array(labels)
            
            for i, class_name in enumerate(config.CLASS_NAMES):
                count = np.sum(labels == i)
                percentage = (count / len(labels)) * 100
                print(f"  {class_name:<15} : {count:4d} samples ({percentage:5.2f}%)")
            
            # Calculate imbalance ratio
            counts = [np.sum(labels == i) for i in range(config.NUM_CLASSES)]
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 2.0:
                print(f"  ⚠ WARNING: Imbalanced data! Ratio: {imbalance_ratio:.2f}:1")
            else:
                print(f"  ✓ Data is reasonably balanced (ratio: {imbalance_ratio:.2f}:1)")
    
    except Exception as e:
        print(f"✗ Error checking data balance: {e}")


def main():
    """
    Main diagnostic function.
    """
    print("\n" + "="*70)
    print("BRAIN TUMOR MODEL DIAGNOSTIC TOOL")
    print("="*70)
    
    # Check data balance
    check_data_balance()
    
    # Analyze models
    print("\n" + "="*70)
    print("ANALYZING TRAINED MODELS")
    print("="*70)
    
    # Load test dataset
    try:
        _, _, test_dataset = prepare_datasets()
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        return
    
    # Analyze Custom CNN
    if os.path.exists(config.CUSTOM_CNN_MODEL_PATH):
        analyze_predictions(config.CUSTOM_CNN_MODEL_PATH, test_dataset, config.CLASS_NAMES)
    else:
        print(f"\n⚠ Custom CNN model not found at: {config.CUSTOM_CNN_MODEL_PATH}")
    
    # Analyze Transfer Learning Model
    if os.path.exists(config.TRANSFER_MODEL_PATH):
        analyze_predictions(config.TRANSFER_MODEL_PATH, test_dataset, config.CLASS_NAMES)
    else:
        print(f"\n⚠ Transfer model not found at: {config.TRANSFER_MODEL_PATH}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\n")
    
    print("If your model shows bias (only predicts 2 classes):")
    print("1. ✓ RETRAIN with the updated code (class weights now enabled)")
    print("2. Check if your dataset is balanced across all 4 classes")
    print("3. Ensure each class has enough training samples (100+ recommended)")
    print("4. Try training for more epochs (30-40)")
    print("5. Use lower learning rate (now set to 0.0005)")
    print("6. Consider data augmentation balance")
    print("\nThe code has been updated with:")
    print("  - Class weights to handle imbalanced data")
    print("  - Better regularization (L2 + varied dropout)")
    print("  - Lower learning rate for better convergence")
    print("  - Additional dense layer for better feature learning")
    print("  - Less aggressive data augmentation")
    print("\nRun: python train.py  (to retrain with improvements)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
