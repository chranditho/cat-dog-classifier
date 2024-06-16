import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import cv2
import matplotlib.pyplot as plt

# Define your image size
image_size = (150, 150)

# Load your trained Keras model
model = load_model('results/trial_1/best_model.keras')  # Replace with your model path

class GradCAM:
    def __init__(self, model, class_index, layer_name):
        self.model = model
        self.class_index = class_index
        self.layer_name = layer_name
        self.grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

    def compute_heatmap(self, image, epsilon=1e-8):
        # Preprocess image for prediction
        image = cv2.resize(image, (image_size[1], image_size[0]))
        image = image / 255.0  # Rescale to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.asarray(image, dtype=np.float32)  # Ensure float32 for TensorFlow

        # Start GradientTape
        with tf.GradientTape() as tape:
            # Get intermediate layer output and model predictions
            layer_output, predictions = self.grad_model(image)
            class_activation = layer_output[0]

            # Determine class index
            predicted_class = np.argmax(predictions[0])
            if self.class_index is None:
                self.class_index = predicted_class

            # Compute gradients for the predicted class
            output = predictions[:, self.class_index]
            grads = tape.gradient(output, layer_output)[0]

        # Compute pooled gradients and generate heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, class_activation), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + epsilon

        heatmap = heatmap.numpy()  # Convert to numpy array
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap, predicted_class, np.max(predictions[0])

# List of layer names you want to visualize (replace with actual layer names from your model)
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']

# Example usage after training
# Assuming you have a list of image paths for visualization
image_paths = ['image05.jpg']

# Initialize Grad-CAM for each layer
for layer_name in layer_names:
    grad_cam = GradCAM(model, None, layer_name)

    # Process each image
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path)

        # Generate class activation heatmap
        heatmap, predicted_class, _ = grad_cam.compute_heatmap(image)

        # Overlay heatmap on image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # Resize heatmap to match image size
        output_image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.5, heatmap, 0.5, 0)

        # Display the result
        plt.figure(figsize=(8, 8))
        plt.imshow(output_image)
        plt.axis('off')
        plt.savefig('gradcam_'+image_path)
        plt.title(f'Grad-CAM: Layer {layer_name}, Class {predicted_class}')
        plt.show()
