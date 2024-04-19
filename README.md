# CNN Model Report

## Link For Testing the Project http://bsusheelkumar01.pythonanywhere.com/

## Overview

This repository contains the code for training a Convolutional Neural Network (CNN) model on the MNIST dataset. The trained model achieved an accuracy of 98% on the validation set.

## Dataset Information

- Length of Training Dataset: 60,000 samples
- Length of Validation Dataset: 10,000 samples

## Model Architecture

The CNN model architecture used in this project consists of three convolutional layers followed by max-pooling layers, and three fully connected layers. Batch normalization and dropout techniques were applied for regularization.

```python
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        
        # Convolutional layers with batch normalization and ReLU activation
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected layers with dropout and ReLU activation
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.classifier(x)
        return x
```

## Training Details

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Cross Entropy Loss
- Epochs: 10

### Training and Validation Metrics

- Training Loss: 0.0202
- Training Accuracy: 0.9942
- Validation Loss: 0.0371
- Validation Accuracy: 0.9897

## Analysis

The trained model achieved an impressive accuracy of 98% on the validation set. This indicates that the model has learned the features of the MNIST dataset effectively. 

To further improve the model performance, the following strategies can be explored:

1. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizer configurations to find the optimal hyperparameters.
2. **Model Architecture**: Try deeper or wider networks, explore different activation functions, or incorporate techniques such as residual connections.
3. **Data Augmentation**: Apply augmentation techniques such as rotation, translation, and scaling to increase the diversity of training data and improve generalization.
4. **Ensemble Learning**: Train multiple models with different initializations or architectures and combine their predictions to improve performance.

## Deployment

The trained model has been deployed using Flask framework. Frontend for the application was built with the help of ChatGPT. The deployment was done on PythonAnywhere platform.

## Conclusion

In conclusion, the trained CNN model achieved a high accuracy on the MNIST dataset. By implementing the mentioned strategies for improvement, further enhancements in model performance can be achieved.
