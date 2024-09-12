# NanoPhasePredictor
NanoPhasePrediction is an AI-driven framework designed to predict material phases using a hybrid CNN-SVM approach. This project focuses on optimizing phase discovery in nano-scale phase prediction of PVDF through advanced machine learning techniques.

Certainly! Below is the revised `README.md` file with the hybrid CNN-SVM model approach emphasized.

---

# CNN-SVM Hybrid Model for Phase Prediction

This project implements a **hybrid machine learning model** combining a Convolutional Neural Network (CNN) for feature extraction and a Support Vector Machine (SVM) for classification. This hybrid approach leverages the CNN's ability to extract complex patterns from image data, while the SVM efficiently classifies the extracted features.

## Overview
The CNN-SVM hybrid model is designed for tasks involving image classification. The CNN model processes input images, extracting high-level features. These features are then used as input for an SVM classifier to make the final predictions.

## Requirements
To run the project, the following libraries are required:
- `numpy`
- `tensorflow`
- `scikit-learn`

Install the dependencies using:
```bash
pip install numpy tensorflow scikit-learn
```

## Workflow

1. **CNN for Feature Extraction**:
   - The CNN architecture is defined in the `create_cnn_model()` function. It includes:
     - Two convolutional layers with ReLU activation and max-pooling layers.
     - A fully connected layer to reduce the feature dimensions.
     - A softmax output layer for initial classification.

2. **Hybrid CNN-SVM Model**:
   - After training the CNN using `train_cnn_model()`, the extracted features are passed to an SVM classifier. The SVM handles the final classification task using these extracted features.
   
3. **Feature Extraction**:
   - The CNN model is used to generate feature vectors from the input images using the `extract_cnn_features()` function.

4. **SVM Training and Classification**:
   - The SVM classifier is trained using the feature vectors extracted by the CNN. The classification performance is evaluated on the test data using accuracy as the metric.

## Code Structure

- **`create_cnn_model(input_shape, num_classes)`**: Defines the CNN architecture for feature extraction.
- **`train_cnn_model(X_train, y_train, input_shape, num_classes)`**: Trains the CNN on the training dataset.
- **`extract_cnn_features(model, X)`**: Extracts features from the CNN for classification.
- **`train_svm_model(X_train_features, y_train)`**: Trains the SVM classifier using CNN-extracted features.
- **`main()`**: Main function to execute the entire hybrid pipeline from data generation, model training, feature extraction, SVM training, and evaluation.

## How to Run

1. Download or clone this repository.
2. Ensure all dependencies are installed by running:
   ```bash
   pip install numpy tensorflow scikit-learn
   ```
3. Run the code:
   ```bash
   python Phase_prediction_hybrid_210324.ipynb
   ```

## Model Training and Evaluation

- **Feature Extraction**: After training, the CNN is used to extract high-level features from the images, which serve as input to the SVM.
- **SVM Training**: The SVM classifier is trained on the features extracted from the CNN. The SVM uses a linear kernel.
- **Model Evaluation**: The accuracy of the hybrid CNN-SVM model is computed and displayed for the test dataset.

## Author
https://github.com/ABnano


