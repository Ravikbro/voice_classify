# Audio Classifier Project

This project is an audio classifier that uses MFCC (Mel-frequency cepstral coefficients) features to classify audio data into different classes. It uses a neural network model implemented with Keras to perform the classification.

## Feature Extraction

The feature extraction process involves converting audio files into MFCC features. The `feature_extractor` function takes an audio file as input, loads the audio using the librosa library, and computes the MFCC features. The MFCC features are then scaled by taking the mean along the axis and returned as the extracted feature.

## Dataset

The dataset used for training the audio classifier is stored in a Pandas DataFrame called `df`. Each row in the DataFrame represents an audio sample and contains the file name, class label, and other relevant information.

## Model Architecture

The neural network model is implemented using Keras. It consists of multiple dense layers with dropout regularization to prevent overfitting. The model architecture is as follows:

1. Input layer: Dense layer with 100 units and ReLU activation function.
2. Dropout layer: Dropout regularization with a rate of 0.5.
3. Hidden layer: Dense layer with 200 units and ReLU activation function.
4. Dropout layer: Dropout regularization with a rate of 0.5.
5. Hidden layer: Dense layer with 100 units and ReLU activation function.
6. Dropout layer: Dropout regularization with a rate of 0.5.
7. Output layer: Dense layer with the number of units equal to the number of class labels and softmax activation function.

## Training and Evaluation

The training process involves iterating over each row in the DataFrame, extracting the features from the corresponding audio file, and appending the extracted features along with the class label to the `extracted_features` list.

The model is then compiled with the Adam optimizer and trained using the extracted features. Evaluation metrics such as accuracy can be calculated using the sklearn library.

## Usage

To use this audio classifier project, follow these steps:

1. Install the required libraries: librosa, keras, sklearn.
2. Prepare the dataset: Ensure that the audio dataset is properly formatted and available in the specified directory.
3. Run the feature extraction code: Execute the feature extraction code to extract MFCC features from the audio files and store them along with the class labels.
4. Run the model training code: Train the neural network model using the extracted features and class labels.
5. Evaluate the model: Calculate evaluation metrics such as accuracy to assess the performance of the trained model.

Please note that this is a basic readme file outlining the key components of the audio classifier project. For more detailed instructions and code implementation, refer to the project's source code and documentation.
