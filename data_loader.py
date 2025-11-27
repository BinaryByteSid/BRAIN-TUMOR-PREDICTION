"""
Data loading and preprocessing module for Brain Tumor MRI Classification.
Handles dataset loading, augmentation, and preprocessing.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import config


def create_data_generators():
    """
    Create ImageDataGenerator objects for training, validation, and testing.
    Training data includes augmentation, while validation and test data only normalized.
    
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
        rotation_range=config.ROTATION_RANGE,
        zoom_range=config.ZOOM_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        brightness_range=config.BRIGHTNESS_RANGE,
        fill_mode='nearest'
    )
    
    # Validation and test data generators (no augmentation, only normalization)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    print("Data generators created successfully!")
    return train_datagen, val_test_datagen, val_test_datagen


def load_datasets_from_directory():
    """
    Load train, validation, and test datasets from directory structure.
    Uses tf.keras.preprocessing.image_dataset_from_directory for efficient loading.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70 + "\n")
    
    # Load training dataset
    print("Loading training dataset...")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        config.DATA_DIR_TRAIN,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='int',  # Integer labels for sparse_categorical_crossentropy
        seed=config.RANDOM_SEED,
        shuffle=True
    )
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        config.DATA_DIR_VAL,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='int',
        seed=config.RANDOM_SEED,
        shuffle=False
    )
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        config.DATA_DIR_TEST,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='int',
        seed=config.RANDOM_SEED,
        shuffle=False
    )
    
    print(f"\n✓ Datasets loaded successfully!")
    print(f"  - Training batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"  - Validation batches: {tf.data.experimental.cardinality(val_dataset).numpy()}")
    print(f"  - Test batches: {tf.data.experimental.cardinality(test_dataset).numpy()}")
    
    return train_dataset, val_dataset, test_dataset


def normalize_dataset(dataset):
    """
    Normalize pixel values to [0, 1] range.
    
    Args:
        dataset: TensorFlow dataset
        
    Returns:
        Normalized dataset
    """""
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))


def augment_dataset(dataset):
    """
    Apply data augmentation to the dataset.
    
    Args:
        dataset: TensorFlow dataset
        
    Returns:
        Augmented dataset
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config.ROTATION_RANGE / 360.0),  # Convert to fraction
        tf.keras.layers.RandomZoom(config.ZOOM_RANGE),
        tf.keras.layers.RandomTranslation(
            height_factor=config.HEIGHT_SHIFT_RANGE,
            width_factor=config.WIDTH_SHIFT_RANGE
        ),
    ])
    
    def augment(image, label):
        return data_augmentation(image, training=True), label
    
    return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)


def prepare_datasets():
    """
    Load and prepare all datasets with proper preprocessing and augmentation.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets_from_directory()
    
    # Normalize all datasets
    print("\nApplying normalization...")
    train_dataset = normalize_dataset(train_dataset)
    val_dataset = normalize_dataset(val_dataset)
    test_dataset = normalize_dataset(test_dataset)
    
    # Apply augmentation only to training dataset
    print("Applying data augmentation to training dataset...")
    train_dataset = augment_dataset(train_dataset)
    
    # Optimize datasets for performance
    print("Optimizing datasets for performance...")
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("✓ All datasets prepared successfully!\n")
    print("="*70 + "\n")
    
    return train_dataset, val_dataset, test_dataset


def show_sample_images(dataset, num_images=9, class_names=None):
    """
    Display sample images from the dataset with their labels.
    
    Args:
        dataset: TensorFlow dataset
        num_images: Number of images to display (default: 9)
        class_names: List of class names (default: None, uses config.CLASS_NAMES)
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    plt.figure(figsize=(12, 12))
    
    # Get a batch of images
    for images, labels in dataset.take(1):
        for i in range(min(num_images, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            
            # Display image
            plt.imshow(images[i].numpy())
            
            # Get label
            label_idx = labels[i].numpy()
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
            
            plt.title(f"{class_name}", fontsize=12, fontweight='bold')
            plt.axis('off')
    
    plt.suptitle('Sample Brain MRI Images', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def get_dataset_info(dataset, dataset_name="Dataset"):
    """
    Print information about the dataset.
    
    Args:
        dataset: TensorFlow dataset
        dataset_name: Name of the dataset for display
    """
    print(f"\n{dataset_name} Information:")
    print(f"{'='*50}")
    
    # Get element spec
    element_spec = dataset.element_spec
    image_spec, label_spec = element_spec
    
    print(f"Image shape: {image_spec.shape}")
    print(f"Image dtype: {image_spec.dtype}")
    print(f"Label dtype: {label_spec.dtype}")
    print(f"Number of batches: {tf.data.experimental.cardinality(dataset).numpy()}")
    
    # Count total samples
    total_samples = 0
    for _, labels in dataset:
        total_samples += len(labels)
    
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"{'='*50}\n")


def create_class_weight_dict(train_dataset):
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        train_dataset: Training dataset
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Collect all labels
    labels = []
    for _, batch_labels in train_dataset:
        labels.extend(batch_labels.numpy())
    
    labels = np.array(labels)
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\nClass weights (for handling imbalanced data):")
    for class_idx, weight in class_weight_dict.items():
        class_name = config.CLASS_NAMES[class_idx]
        print(f"  {class_name}: {weight:.4f}")
    
    return class_weight_dict


# Main execution for testing
if __name__ == "__main__":
    print("Testing data_loader.py...")
    
    # Load and prepare datasets
    try:
        train_ds, val_ds, test_ds = prepare_datasets()
        
        # Show dataset info
        get_dataset_info(train_ds, "Training Dataset")
        get_dataset_info(val_ds, "Validation Dataset")
        get_dataset_info(test_ds, "Test Dataset")
        
        # Show sample images
        print("Displaying sample images...")
        show_sample_images(train_ds, num_images=9)
        
        print("\n✓ data_loader.py test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the data directory structure exists:")
        print(f"  {config.DATA_DIR_TRAIN}")
        print(f"  {config.DATA_DIR_VAL}")
        print(f"  {config.DATA_DIR_TEST}")
