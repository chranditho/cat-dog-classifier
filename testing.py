from tensorflow.keras.models import load_model

# Load your trained Keras model
model = load_model('results/trial_1/best_model.keras')  # Replace with your model path

# Print model summary to see layer names
model.summary()