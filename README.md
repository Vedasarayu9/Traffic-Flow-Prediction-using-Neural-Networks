# Traffic-Flow-Prediction-using-Neural-Networks
# Introduction

# Project Overview
The objective of this project is to:

* Implement an autoencoder to preprocess and compress traffic data.

* Build a neural network model for predicting traffic congestion, incorporating batch normalization.

* Train the model using different gradient descent methods: full batch, mini-batch, and stochastic gradient descent.

* Compare the training efficiency, convergence speeds, and prediction accuracy of these methods using Adam and Adagrad optimizers.
  
We explore how batch normalization and different gradient descent methods affect the prediction of traffic congestion. An autoencoder is employed for feature reduction, followed by a neural network model to predict traffic flow. The project investigates the efficiency, convergence speed, and prediction accuracy of various gradient descent methods and optimizers.

# Requirements
- **Python 3.8 or higher**
- **Required Python libraries**
  - **TensorFlow/PyTorch**
  - **NumPy**
  - **Pandas**
  - **Scikit-learn**
  - **Matplotlib**
  - **Seaborn**

# Key Features
**Feature Reduction:** An autoencoder is used to reduce the dimensionality of input features, capturing the most relevant information for traffic flow prediction.

**Neural Network Model:** A deep learning model is employed to predict traffic congestion, with batch normalization layers to enhance training stability.

**Gradient Descent Methods:** Various gradient descent methods and optimizers are explored, including:
- **Full batch gradient descent**
- **Mini batch gradient descent**
- **Stochastic gradient descent**
- **Adam optimizer**
- **Adagrad optimizer**
  
**Evaluation Metrics:** Models are evaluated based on efficiency, convergence speed, and prediction accuracy.

# Dataset
Dataset: [Dataset link](URL)

The features in the dataset are:
- **Time**
- **Traffic Flow**
- **Average Speed**
- **Occupancy**
- **Temperature**
- **Humidity**
- **Wind Speed**
- **Precipitation**
- **Visibility**
- **Road Condition**

# Project Structure
- **Autoencoder:** Encodes the input data to a compressed representation using a neural network with an encoder-decoder architecture.
- **Neural Network Classifier:** A neural network with batch normalization layers for predicting traffic congestion.
- **Gradient Descent Optimization:**
Full Batch Gradient Descent
Mini-Batch Gradient Descent
Stochastic Gradient Descent
- **Optimizers:** Training is performed using Adam and Adagrad optimizers for comparison.

# Evaluation
The following methods are compared:

- **Full Batch Gradient Descent**
- **Mini-Batch Gradient Descent**
- **Stochastic Gradient Descent**

Each method is evaluated on:

- **Training time**
- **Convergence speed**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

The results are visualized using loss curves and classification metrics like accuracy and F1-score.

# Customization
You can customize the model architecture, training parameters, and evaluation metrics to suit your specific needs. Experiment with different hyperparameters to optimize the model's performance.


