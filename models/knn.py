import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2):
        self.k = k
        self.metric = metric

        if self.metric == 'manhattan':
            self.p = 1
        elif self.metric == 'euclidean':
            self.p = 2
        else:
            self.p = p

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    def _get_nearest_neighbours(self, test):
        distances = np.linalg.norm(self.X_train - test, ord=self.p, axis=1)
        indices = np.argsort(distances)[:self.k]
        
        return indices
    
    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            if X_train.columns.empty:
                self.X_train = X_train.values.astype(float)
            else:
                self.X_train = X_train.iloc[:, :-1].values.astype(float)
        else:
            self.X_train = X_train.astype(float)

        self.y_train = y_train
        
    def _predict_instance(self, row):
        indices = self._get_nearest_neighbours(row)
        labels = [self.y_train.iloc[neighbour] for neighbour in indices]
        
        prediction = max(set(labels), key=labels.count)
        
        return prediction

    def predict(self, X_test):
        if isinstance(X_test, pd.DataFrame):
            if X_test.columns.empty:
                X_test = X_test.values.astype(float)
            else:
                X_test = X_test.iloc[:, :-1].values.astype(float)
        else:
            X_test = X_test.astype(float)

        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(tqdm(executor.map(self._predict_instance, X_test), total=len(X_test)))

        elapsed_time = time.time() - start_time

        print(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(results)
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))