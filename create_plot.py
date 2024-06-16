import json
import matplotlib.pyplot as plt

# Load data from JSON file
rootpath = 'results/'
with open(rootpath+'keras_tune_output.json') as f:
    data = json.load(f)

# Number of iterations
num_iterations = len(data)

# Create the plots with a larger figure size
plt.figure(figsize=(24, 32))

# Lists to store aggregated validation accuracy data
val_accuracy_combined = []

for i in range(num_iterations):
    # Extract parameters and metrics
    parameters = data[i]["Parameters"]
    metrics = data[i]["MetricsList"]

    # Extract metrics for plotting
    epochs = range(1, len(metrics) + 1)

    loss = [m["Loss"] for m in metrics]
    val_loss = [m["ValLoss"] for m in metrics]
    accuracy = [m["Accuracy"] for m in metrics]
    val_accuracy = [m["ValAccuracy"] for m in metrics]

    # Aggregate validation accuracy for combined plot
    val_accuracy_combined.append(val_accuracy)

    # Convert parameters to strings for annotations
    params_str = "\n".join([f"{p['Name']}: {p['Value']}" for p in parameters])

    # Plot loss
    plt.subplot(num_iterations, 2, 2*i + 1)
    plt.plot(epochs, loss, label=f'Train Loss Set {i+1}')
    plt.plot(epochs, val_loss, label=f'Validation Loss Set {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss for Set {i+1}')

    # Plot accuracy
    plt.subplot(num_iterations, 2, 2*i + 2)
    plt.plot(epochs, accuracy, label=f'Train Accuracy Set {i+1}')
    plt.plot(epochs, val_accuracy, label=f'Validation Accuracy Set {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training and Validation Accuracy for Set {i+1}')

    # Annotate parameters with adjusted spacing
    plt.figtext(0.03, 1 - (i+0.5)/num_iterations, f"Parameters Set {i+1}:\n{params_str}", fontsize=12, verticalalignment='center')

# Adjust layout to give less space on the left
plt.tight_layout(rect=[0.1, 0.03, 1, 0.95])

# Save the plot with individual sets as a PNG file
plt.savefig(rootpath+'plot.png')

# Show the plot
plt.show()

# Create a separate plot for combined Validation Accuracy
plt.figure(figsize=(12, 8))

# Plot combined Validation Accuracy
for i, acc in enumerate(val_accuracy_combined):
    plt.plot(epochs, acc, label=f'Set {i+1}')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Combined Validation Accuracy across Sets')
plt.legend()
plt.grid(True)

# Save the combined Validation Accuracy plot as a PNG file
plt.savefig(rootpath+'combined_accuracy_plot.png')

# Show the combined Validation Accuracy plot
plt.show()
