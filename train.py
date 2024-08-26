import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

# Load the dataset
df = pd.read_csv('data.csv')

# Drop the 'PROV' and 'RA' columns if they exist
df = df.drop(['PROV', 'RA'], axis=1, errors='ignore')

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Preprocess the data
le = LabelEncoder()
for column in categorical_features:
    df[column] = le.fit_transform(df[column].astype(str))

# Assume all columns except 'Classification Score' are features
X = df.drop('Classification Score', axis=1)

# Create a binary target: 1 if SLE is present (Classification Score >= 6), 0 otherwise
y = (df['Classification Score'] >= 6).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', SimpleImputer(strategy='constant', fill_value='missing'), X.select_dtypes(include=['object']).columns)
    ])

# Create a pipeline with preprocessor and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter search space
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = X.columns.tolist()
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# SHAP values
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
X_test_preprocessed = best_model.named_steps['preprocessor'].transform(X_test)
shap_values = explainer.shap_values(X_test_preprocessed)

plt.close()

# Save the model and required info
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(output_dir, 'model.joblib'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))

# Save model weights
model_weights = best_model.named_steps['classifier'].feature_importances_
np.save(os.path.join(output_dir, 'model_weights.npy'), model_weights)

# Save unique values for each column
unique_values = {col: df[col].unique().tolist() for col in categorical_features}
joblib.dump(unique_values, os.path.join(output_dir, 'unique_values.joblib'))

print("Model training and evaluation completed. Results saved in the 'output' directory.")
