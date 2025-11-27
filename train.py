"""
Training script for Brain Tumor MRI Classification models.
Trains both custom CNN and transfer learning models with callbacks and visualization.
"""

import os
import time
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import config
from data_loader import prepare_datasets, get_dataset_info
from models import (
    build_custom_cnn, 
    build_transfer_learning_model, 
    get_callbacks,
    unfreeze_top_layers
)
from utils import (
    plot_training_history, 
    save_history, 
    set_seeds,
    get_model_summary_info
)


def train_custom_cnn(train_dataset, val_dataset, class_weights=None):
    """
    Train the custom CNN model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        class_weights: Dictionary of class weights (optional)
    
    Returns:
        tuple: (trained_model, training_history)
    """
    print("\n" + "="*70)
    print("TRAINING CUSTOM CNN MODEL")
    print("="*70 + "\n")
    
    # Build model
    model = build_custom_cnn()
    
    # Get callbacks
    callbacks = get_callbacks('Custom_CNN', monitor='val_loss')
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("-" * 70)
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
    
    # Save training history
    save_history(history, config.CUSTOM_CNN_HISTORY_PATH)
    
    # Plot training history
    plot_path = os.path.join(config.PLOTS_DIR, 'custom_cnn_training_history.png')
    plot_training_history(history, 'Custom CNN', save_path=plot_path)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    # Load best model
    print(f"\nLoading best model from: {config.CUSTOM_CNN_MODEL_PATH}")
    best_model = keras.models.load_model(config.CUSTOM_CNN_MODEL_PATH)
    
    return best_model, history


def train_transfer_learning_model(train_dataset, val_dataset, fine_tune=True, class_weights=None):
    """
    Train the transfer learning model with optional fine-tuning.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        fine_tune: Whether to perform fine-tuning after initial training
        class_weights: Dictionary of class weights (optional)
    
    Returns:
        tuple: (trained_model, training_history)
    """
    print("\n" + "="*70)
    print("TRAINING TRANSFER LEARNING MODEL")
    print("="*70 + "\n")
    
    # Build model
    model = build_transfer_learning_model(
        model_name=config.PRETRAINED_MODEL,
        trainable=False
    )
    
    # Get callbacks
    callbacks = get_callbacks('Transfer_Learning', monitor='val_loss')
    
    # Phase 1: Train with frozen base
    print("\n" + "-"*70)
    print("PHASE 1: Training with frozen base model")
    print("-"*70 + "\n")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    
    start_time = time.time()
    
    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    phase1_time = time.time() - start_time
    print(f"\n✓ Phase 1 completed in {phase1_time/60:.2f} minutes")
    
    # Phase 2: Fine-tuning (optional)
    history_phase2 = None
    if fine_tune:
        print("\n" + "-"*70)
        print("PHASE 2: Fine-tuning with unfrozen top layers")
        print("-"*70 + "\n")
        
        # Load best model from phase 1
        model = keras.models.load_model(config.TRANSFER_MODEL_PATH)
        
        # Unfreeze top layers
        model = unfreeze_top_layers(model, num_layers=config.FINE_TUNE_LAYERS)
        
        # Update callbacks for fine-tuning
        callbacks = get_callbacks('Transfer_Learning_FineTune', monitor='val_loss')
        
        # Fine-tune with fewer epochs
        fine_tune_epochs = max(10, config.EPOCHS // 2)
        print(f"Fine-tuning epochs: {fine_tune_epochs}")
        print(f"Learning rate: {config.LEARNING_RATE * 0.1}")
        
        start_time = time.time()
        
        history_phase2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        phase2_time = time.time() - start_time
        print(f"\n✓ Phase 2 completed in {phase2_time/60:.2f} minutes")
        
        total_time = phase1_time + phase2_time
    else:
        total_time = phase1_time
    
    print(f"\n✓ Total training completed in {total_time/60:.2f} minutes")
    
    # Combine histories if fine-tuning was performed
    if history_phase2:
        combined_history = {}
        for key in history_phase1.history.keys():
            combined_history[key] = (
                history_phase1.history[key] + history_phase2.history[key]
            )
        
        # Create a pseudo-history object
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        history = CombinedHistory(combined_history)
    else:
        history = history_phase1
    
    # Save training history
    save_history(history, config.TRANSFER_HISTORY_PATH)
    
    # Plot training history
    plot_path = os.path.join(config.PLOTS_DIR, 'transfer_model_training_history.png')
    plot_training_history(history, f'Transfer Learning ({config.PRETRAINED_MODEL})', save_path=plot_path)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    if isinstance(history, CombinedHistory):
        print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"  Validation Loss: {history.history['val_loss'][-1]:.4f}")
    else:
        print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"  Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    # Load best model
    print(f"\nLoading best model from: {config.TRANSFER_MODEL_PATH}")
    best_model = keras.models.load_model(config.TRANSFER_MODEL_PATH)
    
    return best_model, history


def main():
    """
    Main training pipeline.
    Trains both custom CNN and transfer learning models.
    """
    print("\n" + "="*70)
    print("BRAIN TUMOR MRI CLASSIFICATION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Set random seeds for reproducibility
    set_seeds(config.RANDOM_SEED)
    
    # Check GPU availability
    print("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("⚠ No GPU found. Training will use CPU (slower).")
    
    # Load and prepare datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70 + "\n")
    
    try:
        train_dataset, val_dataset, test_dataset = prepare_datasets()
        
        # Display dataset information
        get_dataset_info(train_dataset, "Training Dataset")
        get_dataset_info(val_dataset, "Validation Dataset")
        
    except Exception as e:
        print(f"\n✗ Error loading datasets: {e}")
        print("\nPlease ensure the data directory structure exists:")
        print(f"  {config.DATA_DIR_TRAIN}")
        print(f"  {config.DATA_DIR_VAL}")
        print(f"  {config.DATA_DIR_TEST}")
        return
    
    # Calculate class weights to handle imbalanced data
    print("\n" + "="*70)
    print("CALCULATING CLASS WEIGHTS")
    print("="*70 + "\n")
    
    from data_loader import create_class_weight_dict
    try:
        class_weights = create_class_weight_dict(train_dataset)
        print("✓ Class weights calculated successfully!\n")
    except Exception as e:
        print(f"⚠ Warning: Could not calculate class weights: {e}")
        class_weights = None
    
    # Train Custom CNN
    print("\n" + "="*70)
    print("STEP 1: TRAINING CUSTOM CNN")
    print("="*70)
    
    try:
        custom_model, custom_history = train_custom_cnn(train_dataset, val_dataset, class_weights)
        print("\n✓ Custom CNN training completed successfully!")
    except Exception as e:
        print(f"\n✗ Error training custom CNN: {e}")
        custom_model = None
    
    # Train Transfer Learning Model
    print("\n" + "="*70)
    print("STEP 2: TRAINING TRANSFER LEARNING MODEL")
    print("="*70)
    
    try:
        transfer_model, transfer_history = train_transfer_learning_model(
            train_dataset, 
            val_dataset, 
            fine_tune=True,
            class_weights=class_weights
        )
        print("\n✓ Transfer learning model training completed successfully!")
    except Exception as e:
        print(f"\n✗ Error training transfer learning model: {e}")
        transfer_model = None
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70 + "\n")
    
    if custom_model:
        print("✓ Custom CNN trained and saved to:")
        print(f"    {config.CUSTOM_CNN_MODEL_PATH}")
        custom_info = get_model_summary_info(custom_model)
        print(f"  Parameters: {custom_info['total_params']:,}")
    
    if transfer_model:
        print("\n✓ Transfer Learning Model trained and saved to:")
        print(f"    {config.TRANSFER_MODEL_PATH}")
        transfer_info = get_model_summary_info(transfer_model)
        print(f"  Parameters: {transfer_info['total_params']:,}")
    
    print("\n✓ Training plots saved to:")
    print(f"    {config.PLOTS_DIR}")
    
    print("\n" + "="*70)
    print("Next step: Run 'python evaluate.py' to evaluate the models on test data")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
