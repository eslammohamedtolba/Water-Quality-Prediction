# Water-Quality-Prediction
This project aims to predict the potability of water using various machine learning models. 
The dataset used contains different water quality metrics, and the objective is to classify whether the water is potable (safe for human consumption) or not.

![Image about the final project](<Water Quality Prediction.png>)

## Prerequisites
To run this project, you will need the following dependencies installed:
- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Flask
- Pickle

## Overview of the Code
1- Data Loading and Preprocessing:
- Load the dataset from a CSV file.
- Perform exploratory data analysis (EDA) to understand the dataset.
- Handle missing values by filling them with the mean.
- Visualize data distributions and correlations.

2- Modeling with Classic and Ensemble Algorithms:
- Split the data into training and testing sets.
- Train and evaluate multiple models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, and Gradient Boosting.
- Evaluate the models using accuracy, confusion matrix, and classification report.

3- Modeling with Deep Neural Networks (DNN):
- Build, compile, and train a DNN model using TensorFlow.
- Implement early stopping to prevent overfitting.
- Evaluate the DNN model using accuracy and loss metrics.

4- Hyperparameters Tuning:
- Use RandomizedSearchCV and GridSearchCV for hyperparameter tuning of the Random Forest model.
- Evaluate the best model from hyperparameter tuning.

5- Saving the Model:
- Save the best model using pickle if it does not already exist.


## Model Accuracy
The best model achieved an accuracy of 70% on the test data. This model is the Random Forest classifier obtained after hyperparameter tuning.


## Flask App Structure
The Flask app is designed to serve predictions based on user inputs. Below is the structure of the Flask app:
1- Home Route:
- Renders the home page with a form for user input.

2- Predict Route:
- Accepts POST requests with water quality parameters.
- Loads the saved model.
- Makes predictions based on user inputs.
- Renders the result on the home page.

## Contribution
Contributions to this project are welcome. You can help improve the model's accuracy, explore different DNN architectures, or enhance the data preprocessing and visualization steps. 
Feel free to make any contributions and submit pull requests.
