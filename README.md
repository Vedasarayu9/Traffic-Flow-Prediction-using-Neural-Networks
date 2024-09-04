# Traffic-Flow-Prediction-using-Neural-Networks
# Introduction
This project aims to predict traffic flow using a combination of input features such as:

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
  
We explore how batch normalization and different gradient descent methods affect the prediction of traffic congestion. An autoencoder is employed for feature reduction, followed by a neural network model to predict traffic flow. The project investigates the efficiency, convergence speed, and prediction accuracy of various gradient descent methods and optimizers.

# Features
**Feature Reduction:** An autoencoder is used to reduce the dimensionality of input features, capturing the most relevant information for traffic flow prediction.

**Neural Network Model:** A deep learning model is employed to predict traffic congestion, with batch normalization layers to enhance training stability.

**Gradient Descent Methods:** Various gradient descent methods and optimizers are explored, including:
- **Full batch gradient descent**
- **Mini batch gradient descent**
- **Stochastic gradient descent**
- **Adam optimizer**
- **Adagrad optimizer**
  
**Evaluation Metrics:** Models are evaluated based on efficiency, convergence speed, and prediction accuracy.

# Requirements
- **Python 3.8 or higher**
- **Required Python libraries**
  - **TensorFlow/PyTorch**
  - **NumPy**
  - **Pandas**
  - **Scikit-learn**
  - **Matplotlib**
  - **Seaborn**

