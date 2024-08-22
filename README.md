
# Speech-emotion-Recognition

This project aims to develop a Speech Emotion Recognition (SER) system that can identify the emotional state of a speaker based on their speech. The system uses machine learning techniques to classify emotions such as happiness, sadness, anger, and neutrality from audio input.




## Dateset

The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset is
used for this task, which contains emotional speech recordings.

## Emotions Considered:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

**For this project, only a subset of these emotions is considered:**
- Calm
- Happy
- Fearful
- Disgust





## Dataset and Preprocessing

- Dataset:
    - RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):
    - Contains recordings of actors speaking with different emotions.
- Audio files are labeled with specific emotions, allowing for supervised learning.
- Emotion Labels:
    - Complete Set: Neutral, calm, happy, sad, angry, fearful, disgust, surprised.
    - Selected Emotions: Focused on calm, happy, fearful, and disgust for this project.
- Data Loading Process:
    - File Iteration: Iterate through audio files, extract emotion labels from filenames.
    - Filter Emotions: Only include samples with the specified observed_emotions.
    - Feature Extraction: Extract features for each file and store them for model training.

## Model Training

- Algorithm Used:
    - Multi-Layer Perceptron (MLP) Classifier:
- Why MLP: Suitable for complex, non-linear relationships in data.
- Architecture: Consists of an input layer, a single hidden layer, and an output layer.
- Hidden Layer: Configured with 300 neurons for capturing complex patterns.
- Hyperparameters:
    - Alpha (Regularization): Helps prevent overfitting by adding a penalty for larger weights.
    - Batch Size: Determines the number of samples per gradient update.
    - Learning Rate: Adaptive learning rate adjusts during training for optimal convergence.
    - Max Iterations: Limits the number of epochs for training, ensuring timely convergence.
- Training Process:
    - Data Split: 75% training data and 25% testing data for model evaluation.
    - Training: Fit the model on the training data using the extracted features.
    - Validation: Evaluate the model on unseen test data to measure performance.
    
## Model Evaluation
- Metrics Used:
    - Accuracy: Measures the percentage of correct predictions.
    - F1 Score: Provides a balance between precision and recall, useful for imbalanced datasets.
- Performance Evaluation:
    - Predictions: Use the trained model to predict emotions on the test dataset.
    - Comparison: Compare actual and predicted emotions to assess the model's effectiveness.
    - Accuracy Calculation: Determine the percentage of correctly classified samples.



## Model Deployment and Usage

- Saving and Loading the Model:
    - Purpose: Save the trained model for future predictions without retraining.
    - Pickle Module: Used for serializing the model into a file
    (modelForPrediction1.sav).
- Making Predictions:
    - Feature Extraction for New Samples: Extract features from a new audio file for prediction.
    - Model Prediction: Use the loaded model to predict the emotion of the new audio sample.


## Implementation Summary

1. Data Preparation:
    - Load audio files and extract features (MFCC, chroma, mel).
    - Split data into training and testing sets, focusing on specific emotions.
2. Model Training:
    - Train an MLP classifier using extracted features.
    - Configure hyperparameters for optimal performance.
3. Model Evaluation:
    - Evaluate model performance using accuracy and F1 score.
    - Analyze predictions to identify areas of improvement.
4. Model Deployment:
    - Save the trained model for future use.
    - Load the model and make predictions on new audio samples.


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

