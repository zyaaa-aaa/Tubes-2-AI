import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load datasets
additional_train = pd.read_csv('dataset/train/additional_features_train.csv')
basic_train = pd.read_csv('dataset/train/basic_features_train.csv')
content_train = pd.read_csv('dataset/train/content_features_train.csv')
flow_train = pd.read_csv('dataset/train/flow_features_train.csv')
time_train = pd.read_csv('dataset/train/time_features_train.csv')
labels_train = pd.read_csv('dataset/train/labels_train.csv')

train_data = (additional_train
              .merge(basic_train, on='id', how='inner')
              .merge(content_train, on='id', how='inner')
              .merge(flow_train, on='id', how='inner')
              .merge(time_train, on='id', how='inner')
              .merge(labels_train, on='id', how='inner'))

additional_test = pd.read_csv('dataset/test/additional_features_test.csv')
basic_test = pd.read_csv('dataset/test/basic_features_test.csv')
content_test = pd.read_csv('dataset/test/content_features_test.csv')
flow_test = pd.read_csv('dataset/test/flow_features_test.csv')
time_test = pd.read_csv('dataset/test/time_features_test.csv')

test_data = (additional_test
             .merge(basic_test, on='id', how='inner')
             .merge(content_test, on='id', how='inner')
             .merge(flow_test, on='id', how='inner')
             .merge(time_test, on='id', how='inner'))

# Separate features and target
X = train_data.drop(columns=['attack_cat', 'label', 'id'])
y = train_data['attack_cat']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=5))  # Optional PCA
])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import f1_score

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}

    def fit(self, X, y):
        self.classes = np.unique(y)  # Identify unique classes
        for c in self.classes:
            X_c = X[y == c]  # Subset of X for class c
            self.mean[c] = X_c.mean(axis=0)  # Mean of features for class c
            self.var[c] = X_c.var(axis=0)   # Variance of features for class c
            self.prior[c] = X_c.shape[0] / X.shape[0]  # Prior probability P(c)

    def calculate_likelihood(self, mean, var, x):
        # Gaussian likelihood
        eps = 1e-6  # To prevent division by zero
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-(x - mean) ** 2 / (2 * (var + eps)))
        return coeff * exponent

    def calculate_posterior(self, x):
        # Calculate posterior probability for each class
        posteriors = []
        for c in self.classes:
            prior = np.log(self.prior[c])  # Log of prior probability
            likelihood = np.sum(
                np.log(self.calculate_likelihood(self.mean[c], self.var[c], x))
            )  # Log of likelihood
            posterior = prior + likelihood  # Combine log prior and log likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]  # Return class with highest posterior

    def predict(self, X):
        return np.array([self.calculate_posterior(x) for x in X])

# Train-Test Split and Preprocessing
X_train_transformed = pipeline.fit_transform(X_train)
X_val_transformed = pipeline.transform(X_val)

# SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

# Train Naive Bayes
nb = NaiveBayes()
nb.fit(X_train_balanced, y_train_balanced)

# Validate Model
y_val_pred = nb.predict(X_val_transformed)

# Compute Validation Metrics
val_accuracy = np.mean(y_val_pred == y_val)
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')  # Macro-averaged F1 score

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Macro F1 Score: {val_f1_macro:.4f}")

# Preprocess and Predict Test Data
X_test = test_data.drop(columns=['id', 'label'], errors='ignore')
X_test_transformed = pipeline.transform(X_test)
y_test_pred = nb.predict(X_test_transformed)

# Create Submission File
submission = pd.DataFrame({
    'id': test_data['id'],
    'attack_cat': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")
