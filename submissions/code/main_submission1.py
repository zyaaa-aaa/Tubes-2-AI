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
additional_train = pd.read_csv('../../dataset/train/additional_features_train.csv')
basic_train = pd.read_csv('../../dataset/train/basic_features_train.csv')
content_train = pd.read_csv('../../dataset/train/content_features_train.csv')
flow_train = pd.read_csv('../../dataset/train/flow_features_train.csv')
time_train = pd.read_csv('../../dataset/train/time_features_train.csv')
labels_train = pd.read_csv('../../dataset/train/labels_train.csv')

train_data = (additional_train
              .merge(basic_train, on='id', how='inner')
              .merge(content_train, on='id', how='inner')
              .merge(flow_train, on='id', how='inner')
              .merge(time_train, on='id', how='inner')
              .merge(labels_train, on='id', how='inner'))

additional_test = pd.read_csv('../../dataset/test/additional_features_test.csv')
basic_test = pd.read_csv('../../dataset/test/basic_features_test.csv')
content_test = pd.read_csv('../../dataset/test/content_features_test.csv')
flow_test = pd.read_csv('../../dataset/test/flow_features_test.csv')
time_test = pd.read_csv('../../dataset/test/time_features_test.csv')

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

# Preprocess training data
X_train_transformed = pipeline.fit_transform(X_train)
X_val_transformed = pipeline.transform(X_val)

print("selesai pipeline")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

print("selesai SMOTE")

# Implement KNN from scratch
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

# Train and validate KNN
knn = KNN(k=5)
knn.fit(X_train_balanced, y_train_balanced)
y_val_pred = knn.predict(X_val_transformed)
print("selesai KNN")

# Print validation accuracy
val_accuracy = np.mean(y_val_pred == y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Preprocess test data
X_test = test_data.drop(columns=['id', 'label'], errors='ignore')
X_test_transformed = pipeline.transform(X_test)
print("selesai PREPROCESS TEST")


# Predict test data
y_test_pred = knn.predict(X_test_transformed)
print("selesai PREDICT")


# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'attack_cat': y_test_pred
})

submission.to_csv('../csv/submission1.csv', index=False)