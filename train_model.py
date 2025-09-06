
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. Enhanced Data Pipeline ---
IMG_SIZE = 48
BATCH_SIZE = 64
EMOTIONS = ['angry', 'happy', 'sad', 'neutral']
# --- IMPORTANT: Update this path to where your 'train' folder is located ---
DATA_PATH = 'data/archive/train' 

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# --- Data Loading ---
# Load datasets with validation split from the training directory
print("Loading and splitting dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    label_mode='categorical',
    class_names=EMOTIONS,
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    label_mode='categorical',
    class_names=EMOTIONS,
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation'
)

# --- Data Augmentation and Preprocessing ---
print("Setting up data augmentation and preprocessing...")
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# Preprocessing function for training data (includes augmentation)
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    image = data_augmentation(image)
    image = image / 255.0  # Rescale after augmentation
    return image, label

# Preprocessing function for validation data (no augmentation)
def preprocess_val(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing and optimize the data pipeline
train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

# --- 2. Optimized Model Architecture ---
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(EMOTIONS)):
    print("Building model architecture...")
    inputs = tf.keras.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 1
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

model = build_model()
model.summary()

# --- 3. Improved Training Strategy ---
print("Configuring training callbacks...")
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint('emotion_model_prod.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'),
]

# Class weighting to handle imbalanced data
class_weight = {
    0: 1.2,  # angry
    1: 0.8,  # happy
    2: 1.5,  # sad
    3: 1.0   # neutral
}

# --- 4. Model Training ---
print("\n--- Starting Model Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks,
    class_weight=class_weight
)
print("--- Model Training Finished ---")

# --- 5. Evaluation ---
def plot_enhanced_metrics(history):
    print("Plotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

plot_enhanced_metrics(history)

# Load best model and evaluate on validation set for final metrics
print("Loading best model for final evaluation...")
best_model = tf.keras.models.load_model('emotion_model_prod.h5')
test_loss, test_acc, test_prec, test_rec = best_model.evaluate(val_ds)

f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec)

print(f"\n Final Validation Metrics ")
print(f"Accuracy: {test_acc:.2%}")
print(f"Precision: {test_prec:.2%}")
print(f"Recall: {test_rec:.2%}")
print(f"F1 Score: {f1_score:.2%}")

print("\n✅ Training complete and model saved as 'emotion_model_prod.h5'")
