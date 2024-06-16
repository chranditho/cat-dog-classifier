import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load your Keras model
model = load_model('results_old/trial_1/best_model.keras')

# Function to compute saliency map
def compute_saliency(model, img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_orig = img.copy()  # Make a copy of the original image for display later
    img = image.img_to_array(img)
    img /= 255.
    img_tensor = tf.convert_to_tensor(img)  # Convert NumPy array to tf.Tensor

    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, tf.argmax(predictions[0])]

    gradients = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_mean(tf.abs(gradients), axis=-1)

    return saliency[0], img_orig


# List of image paths
image_paths = ['image06.jpg']

# Generate saliency map for each image
for img_path in image_paths:
    saliency_map, original_image = compute_saliency(model, img_path)

    # Plotting both the original image and the saliency map side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    axs[0].imshow(original_image)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Plot saliency map
    axs[1].imshow(saliency_map, cmap='hot')
    axs[1].axis('off')
    axs[1].set_title(f'Saliency Map for {img_path}')

    plt.tight_layout()
    plt.savefig('saliency_map_'+img_path)
    plt.show()
