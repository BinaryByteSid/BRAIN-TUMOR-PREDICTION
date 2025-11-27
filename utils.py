"""
Utility functions for the Brain Tumor MRI Classification Project.
Includes plotting, model saving, and other helper functions.
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import config


def plot_training_history(history, model_name, save_path=None):
    """
    Plot training and validation accuracy and loss curves.
    
    Args:
        history: Training history object or dictionary
        model_name: Name of the model (for title)
        save_path: Path to save the plot (optional)
    """
    # Extract history data
    if hasattr(history, 'history'):
        hist = history.history
    else:
        hist = history
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(hist['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(hist['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(hist['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(hist['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Name of the model (for title)
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names, model_name):
    """
    Print detailed classification report with precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Name of the model
    """
    print(f"\n{'='*70}")
    print(f"Classification Report - {model_name}")
    print(f"{'='*70}\n")
    
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names, 
                                   digits=4)
    print(report)


def save_history(history, save_path):
    """
    Save training history to a pickle file.
    
    Args:
        history: Training history object
        save_path: Path to save the history
    """
    with open(save_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {save_path}")


def load_history(load_path):
    """
    Load training history from a pickle file.
    
    Args:
        load_path: Path to load the history from
        
    Returns:
        Dictionary containing training history
    """
    with open(load_path, 'rb') as f:
        history = pickle.load(f)
    return history


def display_sample_images(dataset, class_names, num_images=9):
    """
    Display sample images from the dataset with their labels.
    
    Args:
        dataset: TensorFlow dataset
        class_names: List of class names
        num_images: Number of images to display
    """
    plt.figure(figsize=(12, 12))
    
    # Get a batch of images
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"Class: {class_names[labels[i]]}", fontsize=12, fontweight='bold')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def get_model_summary_info(model):
    """
    Get model summary information including total parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }


def compare_models(results_dict):
    """
    Compare multiple models and display results.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
                     Example: {'Custom CNN': {'accuracy': 0.95, 'f1_score': 0.94, 'params': 1000000}}
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Test Accuracy':<15} {'F1-Score':<15} {'Parameters':<15}")
    print(f"{'-'*80}")
    
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in results_dict.items():
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1_score', 0)
        params = metrics.get('params', 0)
        
        print(f"{model_name:<20} {accuracy:<15.4f} {f1_score:<15.4f} {params:<15,}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print(f"{'-'*80}")
    print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")
    print(f"\n{'='*80}\n")


def preprocess_image_for_prediction(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image array
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array


def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to: {seed}")
