"""
Evaluation module for Brain Tumor MRI Classification models.
Evaluates trained models on test data with metrics and visualizations.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

import config
from data_loader import prepare_datasets
from utils import (
    plot_confusion_matrix, 
    print_classification_report,
    compare_models,
    get_model_summary_info
)


def evaluate_model_on_test(model, test_dataset, model_name):
    """
    Evaluate a model on the test dataset and compute metrics.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        model_name: Name of the model for display
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n" + "="*70)
    print(f"EVALUATING {model_name.upper()}")
    print("="*70 + "\n")
    
    # Get predictions
    print("Computing predictions on test set...")
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    print("Computing evaluation metrics...")
    
    # Accuracy
    test_accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Model parameters
    model_info = get_model_summary_info(model)
    
    # Store results
    results = {
        'model_name': model_name,
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'y_true': y_true,
        'y_pred': y_pred,
        'params': model_info['total_params']
    }
    
    # Print results
    print("\n" + "-"*70)
    print("TEST SET PERFORMANCE")
    print("-"*70 + "\n")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1_score:.4f}")
    print(f"Total Params:   {model_info['total_params']:,}")
    
    # Print per-class metrics
    print("\n" + "-"*70)
    print("PER-CLASS METRICS")
    print("-"*70 + "\n")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*70)
    
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"{class_name:<15} {precision_per_class[i]:<12.4f} "
              f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} "
              f"{support_per_class[i]:<10.0f}")
    
    # Print classification report
    print_classification_report(y_true, y_pred, config.CLASS_NAMES, model_name)
    
    # Plot confusion matrix
    cm_save_path = os.path.join(
        config.PLOTS_DIR, 
        f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    )
    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, model_name, save_path=cm_save_path)
    
    return results


def plot_metrics_comparison(results_list):
    """
    Plot comparison of metrics between models.
    
    Args:
        results_list: List of result dictionaries from evaluate_model_on_test
    """
    if len(results_list) < 2:
        print("Need at least 2 models for comparison.")
        return
    
    model_names = [r['model_name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    f1_scores = [r['f1_score'] for r in results_list]
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    axes[0, 0].bar(model_names, accuracies, color=['#3498db', '#e74c3c'])
    axes[0, 0].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # F1-Score comparison
    axes[0, 1].bar(model_names, f1_scores, color=['#3498db', '#e74c3c'])
    axes[0, 1].set_title('F1-Score (Weighted)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Precision comparison
    axes[1, 0].bar(model_names, precisions, color=['#3498db', '#e74c3c'])
    axes[1, 0].set_title('Precision (Weighted)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Recall comparison
    axes[1, 1].bar(model_names, recalls, color=['#3498db', '#e74c3c'])
    axes[1, 1].set_title('Recall (Weighted)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(recalls):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    comparison_path = os.path.join(config.PLOTS_DIR, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Model comparison plot saved to: {comparison_path}")
    
    plt.show()


def plot_per_class_comparison(results_list):
    """
    Plot per-class F1-score comparison between models.
    
    Args:
        results_list: List of result dictionaries from evaluate_model_on_test
    """
    if len(results_list) < 2:
        print("Need at least 2 models for comparison.")
        return
    
    num_classes = len(config.CLASS_NAMES)
    x = np.arange(num_classes)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each model
    for i, results in enumerate(results_list):
        offset = width * (i - len(results_list)/2 + 0.5)
        bars = ax.bar(x + offset, results['f1_per_class'], width, 
                     label=results['model_name'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    per_class_path = os.path.join(config.PLOTS_DIR, 'per_class_comparison.png')
    plt.savefig(per_class_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class comparison plot saved to: {per_class_path}")
    
    plt.show()


def main():
    """
    Main evaluation pipeline.
    Evaluates both trained models and compares their performance.
    """
    print("\n" + "="*70)
    print("BRAIN TUMOR MRI CLASSIFICATION - EVALUATION PIPELINE")
    print("="*70 + "\n")
    
    # Load test dataset
    print("Loading test dataset...")
    try:
        _, _, test_dataset = prepare_datasets()
        print("✓ Test dataset loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading test dataset: {e}")
        print(f"\nPlease ensure the test data directory exists: {config.DATA_DIR_TEST}")
        return
    
    results_list = []
    
    # Evaluate Custom CNN
    if os.path.exists(config.CUSTOM_CNN_MODEL_PATH):
        print("\n" + "="*70)
        print("STEP 1: EVALUATING CUSTOM CNN")
        print("="*70)
        
        try:
            custom_model = keras.models.load_model(config.CUSTOM_CNN_MODEL_PATH)
            custom_results = evaluate_model_on_test(
                custom_model, 
                test_dataset, 
                'Custom CNN'
            )
            results_list.append(custom_results)
            print("\n✓ Custom CNN evaluation completed!")
        except Exception as e:
            print(f"\n✗ Error evaluating custom CNN: {e}")
    else:
        print(f"\n⚠ Custom CNN model not found at: {config.CUSTOM_CNN_MODEL_PATH}")
        print("   Please train the model first using: python train.py")
    
    # Evaluate Transfer Learning Model
    if os.path.exists(config.TRANSFER_MODEL_PATH):
        print("\n" + "="*70)
        print("STEP 2: EVALUATING TRANSFER LEARNING MODEL")
        print("="*70)
        
        try:
            transfer_model = keras.models.load_model(config.TRANSFER_MODEL_PATH)
            transfer_results = evaluate_model_on_test(
                transfer_model, 
                test_dataset, 
                f'Transfer Learning ({config.PRETRAINED_MODEL})'
            )
            results_list.append(transfer_results)
            print("\n✓ Transfer learning model evaluation completed!")
        except Exception as e:
            print(f"\n✗ Error evaluating transfer learning model: {e}")
    else:
        print(f"\n⚠ Transfer learning model not found at: {config.TRANSFER_MODEL_PATH}")
        print("   Please train the model first using: python train.py")
    
    # Compare models
    if len(results_list) >= 2:
        print("\n" + "="*70)
        print("STEP 3: MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dictionary
        comparison_dict = {}
        for results in results_list:
            comparison_dict[results['model_name']] = {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'params': results['params']
            }
        
        # Print comparison
        compare_models(comparison_dict)
        
        # Plot comparisons
        print("\nGenerating comparison plots...")
        plot_metrics_comparison(results_list)
        plot_per_class_comparison(results_list)
        
    elif len(results_list) == 1:
        print("\n⚠ Only one model was evaluated. Train both models for comparison.")
    else:
        print("\n✗ No models were evaluated. Please train the models first.")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70 + "\n")
    
    if results_list:
        print("✓ Evaluation completed successfully!")
        print(f"\n  Confusion matrices saved to: {config.PLOTS_DIR}")
        print(f"  Comparison plots saved to: {config.PLOTS_DIR}")
        
        # Determine best model
        best_model = max(results_list, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Model: {best_model['model_name']}")
        print(f"   Test Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
        print(f"   F1-Score: {best_model['f1_score']:.4f}")
        
        print("\n" + "="*70)
        print("Next step: Run 'streamlit run app.py' to launch the inference app")
        print("="*70 + "\n")
    else:
        print("✗ No models were successfully evaluated.")
        print("   Please train the models first using: python train.py")


if __name__ == "__main__":
    main()
