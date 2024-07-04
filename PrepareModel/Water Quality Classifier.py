# Import necessary dependencies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint
import pickle as pk
import os
import warnings
warnings.filterwarnings('ignore')


# Load dataset
df = pd.read_csv('PrepareModel\water_potability.csv')
df.head()


# ---------------------------------------------------------------Exploratory Data Analysis

# Describe dataset 
df.describe()

# Show dataset shape
df.shape

# Show some statistical info about the dataset
df.info()

# Check about missing values to decide if I will make data cleaning or not
df.isnull().sum()

# Clean ph, Sulfate and Trihalomethanes columns 
df = df.fillna(df.mean())

df.isnull().sum()

# Group features by the Potability column 
df.groupby('Potability').mean()

# Plot distribution for all columns except Potability
plt.figure(figsize=(15, 15))
colors = sns.color_palette('husl', n_colors=len(df.columns) - 1)
for i, (col, color) in enumerate(zip(df.columns.drop('Potability'), colors)):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, color=color)
    plt.title(col)
plt.tight_layout()
plt.show()

# Count values of Potability column
plt.figure(figsize = (5,5))
sns.countplot(x = 'Potability' ,data = df)
plt.show()
count_values = df['Potability'].value_counts()
print(count_values)

# Create correlation between all columns 
correlation = df.corr()
print(correlation)
# Visualize correlation
plt.figure(figsize = (10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.3f', annot=True, annot_kws={'size':8}, cmap='terrain')
plt.show()
correlation['Potability'].drop('Potability').sort_values(ascending = False).plot(kind = 'bar')

# Box plot dataset
df.boxplot(figsize = (15,5))
plt.show()

# Build function to find the percentage of column outliers
def outlier_percentage(data):
    # Find standard deviation and mean of the data
    std = np.std(data)
    mean = np.mean(data)

    # Define cut-off
    outlier_cut_off = 3 * std

    # Find lower and upper limits of the data
    lower_limit = mean - outlier_cut_off
    upper_limit = mean + outlier_cut_off

    # Find outliers
    outliers = data[(data < lower_limit) | (data > upper_limit)]
    # Calculate percentage of outliers
    outliers_percentage = (len(outliers) / len(data)) * 100

    return outliers_percentage

# Show percentage
Solids_outliers_percentage = outlier_percentage(df['Solids'])
print(f"Percentage of outliers: {Solids_outliers_percentage:.2f}%\n")
# describe Solids column
print(df['Solids'].describe(), end = '\n\n')
# Show skew of the column
print(f"the skew of Solids column is {df['Solids'].skew()}")

# Transform column by take the log for the values
df['Solids'] = np.log(df['Solids'])

# Show percentage
Solids_outliers_percentage = outlier_percentage(df['Solids'])
print(f"Percentage of outliers: {Solids_outliers_percentage:.2f}%\n")
# describe Solids column
print(df['Solids'].describe(), end = '\n\n')
# Show skew of the column
print(f"the skew of Solids column is {df['Solids'].skew()}")


# Plot the distribution after transformation
df['Solids'].hist(figsize = (5,5))
plt.show()

# Plot Kernel Density Estimate (KDE) when Potability is 1 and 0
plt.figure(figsize = (10,10))
for i, col in enumerate(df.columns.drop('Potability')):
    plt.subplot(3, 3, i+1)
    sns.kdeplot(df[col][df['Potability']==1], shade = True, label = col, color = 'Red')
    sns.kdeplot(df[col][df['Potability']==0], shade = True, label = col, color = 'Blue')
plt.show()


# ---------------------------------------------------Modeling with classic and ensemble algorithms

# Split data into input and label data
X = df.drop(columns = ['Potability'], axis = 1)
Y = df['Potability']
print(f'x shape {X.shape}, y shape {Y.shape}')

# Split data into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

# Create function to plot the predicted values vs actual values
def model_functionality(model):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # Create confusion matrix and classification report
    cm_result = confusion_matrix(Y_test, y_pred)
    cr_result = classification_report(Y_test, y_pred)

    return model, accuracy_score(Y_test, y_pred), cm_result, cr_result

# Create models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

trained_models = {}
for name, model in models.items():
    trained_model, score, cm, cr = model_functionality(model)
    print(f'{name} with accuracy: {score * 100}% and confusion matrix:')
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{name} Confusion Matrix')
    plt.show()
    print(f'{name} Classification Report:')
    print(cr, end='\n\n\n')
    trained_models[name] = trained_model


# ------------------------------------------------Modeling with DNN

# Create DNN model
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Show model's summary
model.summary()

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model with a validation split and early stopping
history = model.fit(X_train, Y_train, epochs=20, validation_split=0.1, batch_size=32, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

plt.figure(figsize=(6, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(6, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()


# -------------------------------------------------------------Hyperparameters Tuning

# Define the parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 200),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', 'auto']
}
# Create the model
model = RandomForestClassifier(random_state=42)
# Perform Randomized Search
random_search = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, Y_train)

# Get the best estimator from Randomized Search
best_estimator = random_search.best_estimator_
best_params = random_search.best_params_
best_score = random_search.best_score_
print(f'Best score from RandomizedSearchCV: {best_score}')
print(f'Best parameters from RandomizedSearchCV: {best_params}')


# Define a more refined parameter grid for GridSearchCV based on the results of RandomizedSearchCV
param_grid = {
    'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
    'max_depth': [best_params['max_depth'] - 10, best_params['max_depth'], best_params['max_depth'] + 10],
    'min_samples_split': [best_params['min_samples_split'] - 1, best_params['min_samples_split'], best_params['min_samples_split'] + 1],
    'min_samples_leaf': [best_params['min_samples_leaf'] - 1, best_params['min_samples_leaf'], best_params['min_samples_leaf'] + 1],
    'max_features': [best_params['max_features']]
}
# Perform Grid Search
grid_search = GridSearchCV(best_estimator, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Get the best estimator from Grid Search
best_grid_estimator = grid_search.best_estimator_
best_grid_params = grid_search.best_params_
best_grid_score = grid_search.best_score_
print(f'Best score from GridSearchCV: {best_grid_score}')
print(f'Best parameters from GridSearchCV: {best_grid_params}')

# Train the best model on the full training data and evaluate it
best_model = best_grid_estimator.fit(X_train, Y_train)
final_score = best_model.score(X_test, Y_test)
print(f'Final model accuracy on test data: {final_score}')


# Find confusion matrix and classification report
y_pred_grid = best_model.predict(X_test)
cm_grid = confusion_matrix(y_pred_grid, Y_test)
cr_grid = classification_report(y_pred_grid, Y_test)

# Plot confusion matrix 
sns.heatmap(cm_grid, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()
print('Classification Report:')
print(cr_grid)


# --------------------------------------------------------------------------------------------------------------
# The best model so far is the Random Forest model that resulted from hyperparameter tuning with accuracy 70%. |
# --------------------------------------------------------------------------------------------------------------

# Save best model
# File path for the model
model_path = 'rf_model.sav'

# Check if the file exists before saving
if not os.path.exists(model_path):
    with open(model_path, 'wb') as model_file:
        pk.dump(best_estimator, model_file)
    print(f"Model saved as {model_path}")

