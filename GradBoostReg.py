import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Global variables for dataset and model
dataset = None
model = None
features = []
target = None


def load_data(file_path):
    """Load the dataset from a local file."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print("Data Preview:\n", data.head())
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
def preprocess_data(data, selected_features, target):
    """Preprocess the data."""
    X = data[selected_features]
    y = data[target]
    
    # Splitting into train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    pipeline.feature_names = selected_features

    return pipeline, X_train, X_test, y_train, y_test


def train_model(data, features, target):
    """Train the regression model and save it locally."""
    pipeline, X_train, X_test, y_train, y_test = preprocess_data(data, features, target)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model trained successfully! R² Score: {r2:.2f}")

    # Save model
    joblib.dump(pipeline, "trained_model.pkl")
    print("Model saved as 'trained_model.pkl'.")


def load_model():
    """Load the saved model."""
    try:
        pipeline = joblib.load("trained_model.pkl")
        print("Model loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict(model, input_values):
    """Make predictions using the trained model."""
    try:
        input_df = pd.DataFrame([input_values], columns=model.feature_names)
        prediction = model.predict(input_df)
        print(f"Predicted Value: {prediction[0]:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")


def main():
    print("Welcome to the Local Regression Training and Prediction App!")

    # Step 1: Load dataset
    file_path = input("Enter the path to your dataset (CSV file): ").strip()
    global dataset
    dataset = load_data(file_path)
    if dataset is None:
        return

    # Step 2: Select target variable
    print("\nAvailable columns in the dataset:")
    print(list(dataset.columns))
    global target
    target = input("Enter the target variable: ").strip()

    # Step 3: Select features
    global features
    features_input = input("Enter the features to use (comma-separated): ").strip()
    features = [f.strip() for f in features_input.split(",")]

    # Step 4: Train Model
    train_choice = input("Do you want to train the model? (yes/no): ").strip().lower()
    if train_choice == "yes":
        train_model(dataset, features, target)

    # Step 5: Predict
    predict_choice = input("Do you want to predict with the trained model? (yes/no): ").strip().lower()
    if predict_choice == "yes":
        model = load_model()
        if model:
            input_text = input("Enter feature values (comma-separated): ").strip()
            input_values = [float(x.strip()) if x.replace('.', '', 1).isdigit() else x for x in input_text.split(",")]
            predict(model, input_values)


if __name__ == "__main__":
    main()
