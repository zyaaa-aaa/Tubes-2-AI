import numpy as np
import pickle

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}

    def fit(self, X, y):
        self.classes = np.unique(y)  
        for c in self.classes:
            X_c = X[y == c]  
            self.mean[c] = X_c.mean(axis=0)  
            self.var[c] = X_c.var(axis=0)  
            self.prior[c] = X_c.shape[0] / X.shape[0] 

    def calculate_likelihood(self, mean, var, x):
        eps = 1e-6  
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-(x - mean) ** 2 / (2 * (var + eps)))
        return coeff * exponent

    def calculate_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.prior[c]) 
            likelihood = np.sum(
                np.log(self.calculate_likelihood(self.mean[c], self.var[c], x))
            ) 
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self.calculate_posterior(x) for x in X])
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))