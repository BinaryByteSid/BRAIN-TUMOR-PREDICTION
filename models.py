"""
Model architecture definitions for Brain Tumor MRI Classification.
Includes custom CNN and transfer learning model builders.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
import config


def build_custom_cnn():
    """
    Build a custom Convolutional Neural Network for brain tumor classification.
    
    Architecture:
        - Multiple Conv2D blocks with BatchNormalization and MaxPooling
        - Dropout layers for regularization
        - Dense layers for classification
        - Softmax output for 4 classes
    
    Returns:
        Compiled Keras model
    """
    print("\n" + "="*70)
    print("BUILDING CUSTOM CNN MODEL")
    print("="*70 + "\n")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),
        
        # Fifth Conv Block
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Custom CNN model built successfully!")
    print(f"\nModel Summary:")
    model.summary()
    
    return model


def build_transfer_learning_model(model_name='EfficientNetB0', trainable=False):
    """
    Build a transfer learning model using a pretrained backbone.
    
    Args:
        model_name: Name of the pretrained model ('EfficientNetB0', 'ResNet50', 'VGG16')
        trainable: Whether to make the base model trainable initially
    
    Returns:
        Compiled Keras model
    """
    print("\n" + "="*70)
    print(f"BUILDING TRANSFER LEARNING MODEL ({model_name})")
    print("="*70 + "\n")
    
    # Select pretrained model
    if model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)
        )
    elif model_name == 'ResNet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)
        )
    elif model_name == 'VGG16':
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Freeze base model layers initially
    base_model.trainable = trainable
    
    # Build the complete model
    inputs = layers.Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
    
    # Preprocessing for transfer learning models
    x = inputs
    
    # Base model
    x = base_model(x, training=False)
    
    # Add custom top layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Transfer learning model ({model_name}) built successfully!")
    print(f"   Base model trainable: {trainable}")
    print(f"   Base model layers: {len(base_model.layers)}")
    print(f"\nModel Summary:")
    model.summary()
    
    return model


def unfreeze_top_layers(model, num_layers=20):
    """
    Unfreeze the top N layers of the base model for fine-tuning.
    
    Args:
        model: Transfer learning model
        num_layers: Number of top layers to unfreeze
    
    Returns:
        Model with unfrozen top layers
    """
    print(f"\nUnfreezing top {num_layers} layers for fine-tuning...")
    
    # Find the base model (first layer that's a Model)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, models.Model):
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find base model. Skipping fine-tuning.")
        return model
    
    # Freeze all layers first
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE * 0.1),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Count trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    
    print(f"✓ Top {num_layers} layers unfrozen successfully!")
    print(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model


def get_callbacks(model_name, monitor='val_loss'):
    """
    Create training callbacks for model training.
    
    Args:
        model_name: Name of the model (for saving)
        monitor: Metric to monitor for callbacks
    
    Returns:
        List of Keras callbacks
    """
    # Determine save path based on model name
    if 'custom' in model_name.lower() or 'cnn' in model_name.lower():
        save_path = config.CUSTOM_CNN_MODEL_PATH
    else:
        save_path = config.TRANSFER_MODEL_PATH
    
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard (optional)
        # keras.callbacks.TensorBoard(
        #     log_dir=os.path.join(config.RESULTS_DIR, f'logs/{model_name}'),
        #     histogram_freq=1
        # )
    ]
    
    return callbacks


def load_trained_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    return model


# Main execution for testing
if __name__ == "__main__":
    print("Testing models.py...")
    
    print("\n" + "="*70)
    print("TEST 1: Building Custom CNN")
    print("="*70)
    custom_model = build_custom_cnn()
    
    print("\n" + "="*70)
    print("TEST 2: Building Transfer Learning Model")
    print("="*70)
    transfer_model = build_transfer_learning_model(config.PRETRAINED_MODEL)
    
    print("\n" + "="*70)
    print("TEST 3: Model Parameters Comparison")
    print("="*70)
    
    from utils import get_model_summary_info
    
    custom_info = get_model_summary_info(custom_model)
    transfer_info = get_model_summary_info(transfer_model)
    
    print(f"\nCustom CNN:")
    print(f"  Total parameters: {custom_info['total_params']:,}")
    print(f"  Trainable parameters: {custom_info['trainable_params']:,}")
    
    print(f"\nTransfer Learning Model:")
    print(f"  Total parameters: {transfer_info['total_params']:,}")
    print(f"  Trainable parameters: {transfer_info['trainable_params']:,}")
    
    print("\n✓ models.py test completed successfully!")
