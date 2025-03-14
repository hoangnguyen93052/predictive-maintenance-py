import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load healthcare dataset
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")

# Explore the dataset
def explore_data(data):
    print("Dataset Overview:")
    print(data.info())
    print("\nBasic Statistics:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    
# Visualize target variable distribution
def visualize_target(data, target_variable):
    sns.countplot(x=target_variable, data=data)
    plt.title(f'Distribution of {target_variable}')
    plt.show()

# Preprocess the data
def preprocess_data(data, target_variable):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data.drop(columns=[target_variable]))
    target = data[target_variable].values
    
    # Standard scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    return pd.DataFrame(data_scaled, columns=data.columns[:-1]), target

# Split the dataset into training and test sets
def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Visualize feature importances
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()

# Main function to run the analysis
def main():
    # Load the dataset
    filepath = 'healthcare_data.csv'  # Sample file path
    data = load_data(filepath)

    # Explore the dataset
    explore_data(data)

    # Visualize target variable
    target_variable = 'disease_status'
    visualize_target(data, target_variable)

    # Preprocess the data
    features, target = preprocess_data(data, target_variable)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(features, target)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Plot feature importances
    plot_feature_importances(model, features.columns)

if __name__ == "__main__":
    main()