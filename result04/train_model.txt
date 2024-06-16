import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import os


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

conv_1 = 32
conv_2 = 96
conv_3 = 256
conv_4 = 256
conv_5 = 256
dense_units_1 = 512
dense_units_2 = 256
dropout_rate = 0.3


# Ensure the model save path ends with .keras
if not model_save_path.endswith('.keras'):
    model_save_path += '.keras'

# Image data generators for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
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

# Building the CNN model
model = Sequential([
    Input(shape=(image_size[0], image_size[1], 3)),
    Conv2D(conv_1, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(conv_2, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(conv_3, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(conv_4, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(conv_5, (3, 3), activation='relu'),  # Added layer
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(dense_units_1, activation='relu'),
    Dropout(dropout_rate),
    Dense(dense_units_2, activation='relu'),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])

# Update optimizer argument
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(model_save_path, save_best_only=True),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-7)
]

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Optionally, plot training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
