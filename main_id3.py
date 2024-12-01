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

import numpy as np
import pandas as pd
from collections import Counter

import numpy as np
import pandas as pd
from collections import Counter

class ID3DecisionTree:
    def __init__(self):
        self.tree = None
        self.default_class = None  # Majority class from training set

    def entropy(self, y):
        """Calculate entropy of target labels."""
        class_counts = Counter(y)
        total = len(y)
        return -sum(
            (count / total) * np.log2(count / total)
            for count in class_counts.values()
            if count > 0
        )

    def information_gain(self, X_column, y):
        """Calculate information gain for a feature."""
        total_entropy = self.entropy(y)
        values, counts = np.unique(X_column, return_counts=True)
        weighted_entropy = sum(
            (counts[i] / len(y)) * self.entropy(y[X_column == value])
            for i, value in enumerate(values)
        )
        return total_entropy - weighted_entropy

    def best_split(self, X, y):
        """Identify the feature with the highest information gain."""
        best_gain = -1
        best_feature = None
        for feature in range(X.shape[1]):
            gain = self.information_gain(X[:, feature], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    def build_tree(self, X, y, features):
        """Recursively build the ID3 decision tree."""
        if len(set(y)) == 1:
            return y[0]
        if len(features) == 0 or X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]

        # Select the best feature to split
        best_feature = self.best_split(X, y)
        tree = {features[best_feature]: {}}

        # Split the dataset and recurse
        feature_values = np.unique(X[:, best_feature])
        for value in feature_values:
            subset_indices = X[:, best_feature] == value
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]
            # Remove the best_feature from features
            new_features = np.delete(features, best_feature)
            new_X = np.delete(subset_X, best_feature, axis=1)
            tree[features[best_feature]][value] = self.build_tree(
                new_X, subset_y, new_features
            )
        return tree

    def fit(self, X, y):
        """Fit the ID3 tree to the dataset."""
        # Ensure X and y are NumPy arrays
        X = np.array(X)
        y = np.array(y)
        self.default_class = Counter(y).most_common(1)[0][0]
        features = np.array([f"Feature {i}" for i in range(X.shape[1])])
        self.tree = self.build_tree(X, y, features)

    def predict_one(self, x, tree):
        """Predict a single example."""
        if not isinstance(tree, dict):
            return tree
        root_feature = next(iter(tree))
        feature_idx = int(root_feature.split()[-1])  # Extract feature index
        if feature_idx >= len(x):
            return self.default_class
        feature_value = x[feature_idx]
        subtree = tree[root_feature].get(feature_value, self.default_class)
        return self.predict_one(x, subtree)

    def predict(self, X):
        """Predict for multiple examples."""
        X = np.array(X)
        return np.array([self.predict_one(x, self.tree) for x in X])

# Preprocess and Train the Model
X_train_transformed = pipeline.fit_transform(X_train)
X_val_transformed = pipeline.transform(X_val)

# SMOTE for class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_transformed, y_train
)

# Convert y_train_balanced to NumPy array
y_train_balanced = np.array(y_train_balanced)

# Train the ID3 Decision Tree
id3 = ID3DecisionTree()
id3.fit(X_train_balanced, y_train_balanced)

# Validate the Model
y_val_pred = id3.predict(X_val_transformed)

# Compute Validation Metrics
from sklearn.metrics import accuracy_score, f1_score

val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Macro F1 Score: {val_f1_macro:.4f}")

# Preprocess and Predict Test Data
X_test = test_data.drop(columns=['id', 'label'], errors='ignore')
X_test_transformed = pipeline.transform(X_test)
y_test_pred = id3.predict(X_test_transformed)

# Create Submission File
submission = pd.DataFrame({
    'id': test_data['id'],
    'attack_cat': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")
