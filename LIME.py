import numpy as np
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from tensorflow.keras.models import load_model

# Load your trained Keras model
model = load_model('results/trial_1/best_model.keras')  # Replace with your model path

class LimeExplanation:
    def __init__(self, model):
        self.model = model

    def preprocess_image(self, image):
        # Resize image to model's expected input shape
        image = cv2.resize(image, (150, 150))
        return image

    def generate_lime_explanation(self, image_path, num_samples=1000):
        # Load image
        image = cv2.imread(image_path)

        # Preprocess image
        image = self.preprocess_image(image)

        # Convert image to RGB (if not already)
        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Initialize LimeImageExplainer
        explainer = lime_image.LimeImageExplainer()

        # Explain instance (image) and get explanation
        explanation = explainer.explain_instance(image, self.predict_function, top_labels=1, hide_color=0,
                                                 num_samples=num_samples)

        # Get image and mask with positive and negative colors
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5,
                                                    hide_rest=False)

        # Ensure mask is grayscale
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_BGR2GRAY)

        # Normalize mask to range [0, 1]
        mask_normalized = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-10)

        # Create heatmap with green (positive) and red (negative) colors
        heatmap = np.zeros_like(image)
        heatmap[:, :, 1] = (1 - mask_normalized) * 255  # Green channel
        heatmap[:, :, 2] = mask_normalized * 255  # Red channel

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Overlay heatmap on image
        alpha = 0.7  # Weight of the original image
        beta = 0.3   # Weight of the heatmap
        overlay = cv2.addWeighted(image, alpha, heatmap_resized.astype(np.uint8), beta, 0)

        # Find contours around highlighted areas
        contours, _ = cv2.findContours(np.uint8(mask_normalized * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw yellow bounding boxes around each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow color (BGR format)

        return overlay

    def predict_function(self, images):
        return self.model.predict(images)

# Example usage after training
# Assuming you have a list of image paths for visualization
image_paths = ['image09.jpg']

lime_explainer = LimeExplanation(model)

# Process each image
for image_path in image_paths:
    # Generate LIME explanation overlay
    lime_overlay = lime_explainer.generate_lime_explanation(image_path)

    # Display LIME explanation overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(lime_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'LIME Explanation Overlay with Yellow Borders for {image_path}')
    plt.savefig(f'lime_overlay_{image_path}')
    plt.show()
