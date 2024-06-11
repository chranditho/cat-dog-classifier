# Cat vs Dog Classifier

This project is a Cat vs Dog image classifier built using TensorFlow and Keras, and deployed via a Flask web
application.

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/your-username/cat-dog-classifier.git
cd cat-dog-classifier
```

## Set Up the Virtual Environment

``` bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### Download the Dataset

Access the shared "data" folder on Google Drive.

Download and extract the dataset to data/cats-v-dogs:

``` bash
data/
    cats-v-dogs/
        training/
            cats/
            dogs/
        validation/
            cats/
            dogs/
```

### Train the Model

Use the provided `setup.sh` script to train the model. The script takes several parameters to configure the training
process:

``` bash

./setup.sh -l <learning_rate> -e <epochs> -b <batch_size> -d <dropout_rate>

```

### Parameters:

* -l <learning_rate>: The learning rate for the optimizer. This controls how much to change the model in response to the
  estimated error each time the model weights are updated. A smaller value can result in more stable training but may
  require more epochs to converge. For example, 0.00001.

* -e <epochs>: The number of epochs to train the model. One epoch means that every sample in the training dataset has
  had
  an opportunity to update the internal model parameters once. For example, 50.

* -b <batch_size>: The number of samples per batch of computation. The batch size defines the number of samples that
  will
  be propagated through the network. For example, 32.

* -d <dropout_rate>: The dropout rate for the dropout layer. Dropout is a regularization technique to prevent
  overfitting
  in the model. The dropout rate is the fraction of input units to drop. For example, 0.3.

#### Example:

To train the model with a learning rate of 0.00001, for 50 epochs, a batch size of 32, and a dropout rate of 0.3, you
would run:

```bash
./train_model.sh -l 0.00001 -e 50 -b 32 -d 0.3
```

**Default Parameters:**
If you do not provide any parameters, the script will use the following default values:

* learning_rate: 0.0001
* epochs: 30
* batch_size: 20
* dropout_rate: 0.5

  #### Example Without Parameters:
  If you run the script without any parameters:

``` bash
./setup.sh
```

The model will be trained with the default parameters:

* Learning rate: 0.0001
* Epochs: 30
* Batch size: 20
* Dropout rate: 0.5

## Running the Trained Model

After training the model, you can use the `run_model.sh` script to make predictions on new images.

1. Make the script executable:

``` bash
chmod +x run_model.sh
```

2. Run the run_model.sh script with the path to the image:

``` bash
./run_model.sh <path_to_image>
```

This will output whether the image is predicted to be a cat or a dog.
