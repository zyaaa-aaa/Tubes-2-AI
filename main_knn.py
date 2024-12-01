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

# Preprocess training data
X_train_transformed = pipeline.fit_transform(X_train)
X_val_transformed = pipeline.transform(X_val)

print("selesai pipeline")

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Adjust SMOTE sampling strategy to balance classes better
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

print("SMOTE complete. New class distribution:", Counter(y_train_balanced))

# Weighted KNN implementation
class WeightedKNN:
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
            k_weights = [1 / (distances[i] + 1e-5) for i in k_indices]  # Weighted by distance
            weighted_votes = {}
            for label, weight in zip(k_labels, k_weights):
                if label not in weighted_votes:
                    weighted_votes[label] = 0
                weighted_votes[label] += weight
            predictions.append(max(weighted_votes, key=weighted_votes.get))
        return predictions

# Hyperparameter tuning for K
best_k = None
best_f1_macro = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in range(3, 11):  # Test k values from 3 to 10
    fold_f1_scores = []
    for train_index, val_index in kf.split(X_train_balanced, y_train_balanced):
        X_train_fold = X_train_balanced[train_index]
        y_train_fold = y_train_balanced[train_index]
        X_val_fold = X_train_balanced[val_index]
        y_val_fold = y_train_balanced[val_index]
        
        knn = WeightedKNN(k=k)
        knn.fit(X_train_fold, y_train_fold)
        y_val_pred = knn.predict(X_val_fold)
        fold_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro'))
    
    avg_f1 = np.mean(fold_f1_scores)
    print(f"Average F1 Macro for k={k}: {avg_f1:.4f}")
    
    if avg_f1 > best_f1_macro:
        best_f1_macro = avg_f1
        best_k = k

print("hyperparameter tuning kelar")
print(f"Best k: {best_k}, Best F1 Macro: {best_f1_macro:.4f}")

# Train final KNN with best k
knn = WeightedKNN(k=best_k)
knn.fit(X_train_balanced, y_train_balanced)
y_val_pred = knn.predict(X_val_transformed)
print("KNN kelar")

# Evaluate on validation set
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation Macro F1 Score with k={best_k}: {val_f1_macro:.4f}")

# Preprocess test data and predict
X_test = test_data.drop(columns=['id', 'label'], errors='ignore')
X_test_transformed = pipeline.transform(X_test)
y_test_pred = knn.predict(X_test_transformed)

# Save predictions
submission = pd.DataFrame({
    'id': test_data['id'],
    'attack_cat': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")
