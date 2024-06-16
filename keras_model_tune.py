import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import os
import keras_tuner as kt

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

train_dir = config['train_dir']
validation_dir = config['validation_dir']
image_size = tuple(config['image_size'])
batch_size = config['batch_size']
epochs = config['epochs']
learning_rate = config['learning_rate']
dropout_rate = config['dropout_rate']
model_save_path = config['model_save_path']

# Image data generators for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Build model using Keras Tuner
def build_model(hp):
    model = Sequential([
        Input(shape=(image_size[0], image_size[1], 3)),
        Conv2D(hp.Int('conv_1', min_value=32, max_value=128, step=32), (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(hp.Int('conv_2', min_value=32, max_value=128, step=32), (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(hp.Int('conv_3', min_value=64, max_value=256, step=64), (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(hp.Int('conv_4', min_value=64, max_value=256, step=64), (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(hp.Int('conv_5', min_value=64, max_value=256, step=64), (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
        Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model

# Create directories and prepare for multiple iterations
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Define Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # number of different models to try
    directory=results_dir,
    project_name='image_classification_tuning'
)

# Tuner search
tuner.search(train_generator,
             steps_per_epoch=steps_per_epoch,
             epochs=epochs,
             validation_data=validation_generator,
             validation_steps=validation_steps,
             callbacks=[
                 EarlyStopping(patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-7)
             ])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
best_model.evaluate(validation_generator, steps=validation_steps)

# Plotting and saving histories
history_records = []

for i, trial in enumerate(tuner.oracle.get_best_trials(num_trials=10)):
    trial_dir = os.path.join(results_dir, f"trial_{i + 1}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save the best model for this trial
    model_path = os.path.join(trial_dir, "best_model.keras")
    best_model.save(model_path)

    # Load trial history
    trial_history = trial.metrics.get_history()
    history_records.append(trial_history)

    # Plot training results for this trial
    acc = trial_history['accuracy']
    val_acc = trial_history['val_accuracy']
    loss = trial_history['loss']
    val_loss = trial_history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo', label='Training acc')
    plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
    plt.title(f'Training and validation accuracy - Trial {i + 1}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo', label='Training loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss - Trial {i + 1}')
    plt.legend()

    plot_path = os.path.join(trial_dir, "training_plot.png")
    plt.savefig(plot_path)
    plt.close()

# Plot comparison across trials
plt.figure(figsize=(12, 4))

# Accuracy comparison
plt.subplot(1, 2, 1)
for i, history in enumerate(history_records):
    plt.plot(range(len(history['val_accuracy'])), history['val_accuracy'], label=f'Trial {i + 1}')
plt.title('Validation accuracy comparison across trials')
plt.legend()

# Loss comparison
plt.subplot(1, 2, 2)
for i, history in enumerate(history_records):
    plt.plot(range(len(history['val_loss'])), history['val_loss'], label=f'Trial {i + 1}')
plt.title('Validation loss comparison across trials')
plt.legend()

comparison_plot_path = os.path.join(results_dir, "comparison_plot.png")
plt.savefig(comparison_plot_path)
plt.close()
