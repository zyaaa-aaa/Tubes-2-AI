import numpy as np
import pandas as pd
from collections import Counter

class ID3DecisionTree:
    def __init__(self):
        self.tree = None
        self.default_class = None 

    def entropy(self, y):
        class_counts = Counter(y)
        total = len(y)
        return -sum(
            (count / total) * np.log2(count / total)
            for count in class_counts.values()
            if count > 0
        )

    def information_gain(self, X_column, y):
        total_entropy = self.entropy(y)
        values, counts = np.unique(X_column, return_counts=True)
        weighted_entropy = sum(
            (counts[i] / len(y)) * self.entropy(y[X_column == value])
            for i, value in enumerate(values)
        )
        return total_entropy - weighted_entropy

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        for feature in range(X.shape[1]):
            gain = self.information_gain(X[:, feature], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    def build_tree(self, X, y, features):
        if len(set(y)) == 1:
            return y[0]
        if len(features) == 0 or X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]

        best_feature = self.best_split(X, y)
        tree = {features[best_feature]: {}}

        feature_values = np.unique(X[:, best_feature])
        for value in feature_values:
            subset_indices = X[:, best_feature] == value
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]
            new_features = np.delete(features, best_feature)
            new_X = np.delete(subset_X, best_feature, axis=1)
            tree[features[best_feature]][value] = self.build_tree(
                new_X, subset_y, new_features
            )
        return tree

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.default_class = Counter(y).most_common(1)[0][0]
        features = np.array([f"Feature {i}" for i in range(X.shape[1])])
        self.tree = self.build_tree(X, y, features)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        root_feature = next(iter(tree))
        feature_idx = int(root_feature.split()[-1]) 
        if feature_idx >= len(x):
            return self.default_class
        feature_value = x[feature_idx]
        subtree = tree[root_feature].get(feature_value, self.default_class)
        return self.predict_one(x, subtree)

    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_one(x, self.tree) for x in X])