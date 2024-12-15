import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

if __name__ == "__main__":
    TARGET = "attack_cat"
    RANDOM_STATE = 42

    NUMERICALFEATURES = ['ackdat', 'ct_dst_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_src_dport_ltm', 'ct_src_ltm', 'ct_srv_dst', 'ct_srv_src', 'ct_state_ttl', 'dbytes', 'dinpkt', 'djit', 'dload', 'dloss', 'dmean', 'dpkts', 'dtcpb', 'dttl', 'dur', 'dwin', 'response_body_len', 'sbytes', 'sinpkt', 'sjit', 'sload', 'sloss', 'smean', 'spkts', 'stcpb', 'sttl', 'swin', 'synack', 'tcprtt', 'trans_depth']
    CATEGORICALFEATURES = ['is_ftp_login', 'is_sm_ips_ports', 'proto', 'service', 'state']

    train_set = pd.read_csv("train_set.csv")
    train_set.head()

    val_set = pd.read_csv("val_set.csv")
    val_set.head()

    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    def merge_and_cluster(X_train, X_test, n_clusters=5, random_state=None):
        # Merge X_train and X_test
        combined_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        combined_data['cluster_label'] = kmeans.fit_predict(combined_data)

        # Split the data back into train and test sets with cluster labels
        X_train_clustered, X_test_clustered = train_test_split(
            combined_data, test_size=len(X_test), random_state=random_state, shuffle=False
        )

        return X_train_clustered, X_test_clustered

    from sklearn.base import BaseEstimator, TransformerMixin

    # Create class to create new features
    class FeatureCreator(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.fitted_cols = X.columns
            return self
        
        def transform(self, X):
            # Create new columns
            X['total_bytes'] = X['sbytes'] + X['dbytes']
            # X['packet_rate'] = X['total_bytes'] / (X['dur'] + 1e-9)
            X['byte_ratio'] = X['sbytes'] / (X['dbytes'] + 1e-9)
            X['src_to_dst_ratio'] = X['ct_srv_src'] / (X['ct_srv_dst'] + 1e-9)
            # X['port_usage_ratio'] = X['ct_dst_sport_ltm'] / (X['ct_src_dport_ltm'] + 1e-9)

            # Get the newly created columns
            new_columns = list(filter(lambda x: x not in self.fitted_cols, X.columns))

            # Compute correlation with TARGET feature for the newly created columns
            if TARGET in X.columns:
                target_corr = X[new_columns].apply(lambda col: col.corr(X[TARGET]))

                # Print correlation results for the newly created columns
                for col, corr in target_corr.items():
                    print(f"Feature {col} created with {corr} correlation to target")
            
            return X
        
    # Create feature dropper class
    class FeatureDropper(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return X.drop(columns=CATEGORICALFEATURES)

    from sklearn.preprocessing import StandardScaler

    class FeatureScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = None
            self.numerical_cols = None
        
        def fit(self, X, y=None):
            self.numerical_cols = NUMERICALFEATURES
            self.scaler = StandardScaler().fit(X[self.numerical_cols])
            return self
        
        def transform(self, X):
            X_scaled = X.copy()
            X_scaled[self.numerical_cols] = self.scaler.transform(X[self.numerical_cols])

            return X_scaled
        
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    class CustomPCA(BaseEstimator, TransformerMixin):
        def __init__(self, n_components=2):
            self.pca = None
            self.numerical_cols = None
            self.n_components = n_components

        def fit(self, X, y=None):
            self.numerical_cols = NUMERICALFEATURES
            self.pca = PCA(n_components=self.n_components).fit(X[self.numerical_cols])
            return self
        
        def transform(self, X):
            X_pca = X.copy()
            principal_components = self.pca.transform(X[self.numerical_cols])
            
            for i in range(self.n_components):
                X_pca[f'PC{i + 1}'] = principal_components[:, i]

            X_pca = X_pca.drop(columns=[col for col in self.numerical_cols if col in X_pca.columns])

            return X_pca
        
    from sklearn.preprocessing import LabelEncoder

    class FeatureImputer(BaseEstimator, TransformerMixin):
        def __init__(self, num_strategy="mean", num_fill_value=0, cat_strategy="constant", cat_fill_value="missing", encode_attack_cat=True):
            self.num_strategy = num_strategy
            self.num_fill_value = num_fill_value
            self.cat_strategy = cat_strategy
            self.cat_fill_value = cat_fill_value
            self.encode_attack_cat = encode_attack_cat
            self.imputer = None
            self.label_encoder = None

        def fit(self, X, y=None):
            # Drop 'id' and 'label' columns
            self.columns_to_drop = ['id', 'label']
            X = X.drop(columns=self.columns_to_drop, errors='ignore')

            # Define imputers for numerical and categorical data
            self.imputer = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy=self.num_strategy, fill_value=self.num_fill_value), NUMERICALFEATURES),
                    ("cat", SimpleImputer(strategy=self.cat_strategy, fill_value=self.cat_fill_value), CATEGORICALFEATURES),
                ],
                remainder="passthrough"  # Preserve other columns
            )

            # Fit the imputer
            self.imputer.fit(X)

            # Fit the LabelEncoder for 'attack_cat' if enabled
            if self.encode_attack_cat and "attack_cat" in X.columns:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(X["attack_cat"])
                print("Labels in LabelEncoder:", self.label_encoder.classes_)

            return self

        def transform(self, X):
            # Drop 'id' and 'label' columns
            X = X.drop(columns=self.columns_to_drop, errors='ignore')

            # Apply imputations
            X_transformed = pd.DataFrame(self.imputer.transform(X), columns=self.imputer.get_feature_names_out())

            # Rename columns to remove prefixes added by ColumnTransformer
            X_transformed.columns = [col.split("__")[-1] for col in X_transformed.columns]

            # Encode 'attack_cat' using LabelEncoder if it exists
            if self.encode_attack_cat and "attack_cat" in X.columns:
                X_transformed["attack_cat"] = self.label_encoder.transform(X["attack_cat"])

            return X_transformed



        
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("imputer", FeatureImputer(num_strategy="constant", num_fill_value=0, cat_strategy="constant", cat_fill_value="missing")),  # Handle missing values
        ("featurecreator", FeatureCreator()),
        ("featurescaler", FeatureScaler()),
        ("dropper", FeatureDropper()),
        ("pca", CustomPCA(n_components=5))
    ])


    train_set = pipeline.fit_transform(train_set, train_set[TARGET])
    val_set = pipeline.transform(val_set)

    def match_columns(train, test):
        # Get list of columns in training set
        train_cols = train.columns.tolist()
        
        # Get list of columns in test set
        test_cols = test.columns.tolist()
        
        # Remove any columns in test set that aren't in training set
        for col in test_cols:
            if col not in train_cols:
                test = test.drop(col, axis=1)
        
        # Add any missing columns to test set and fill with 0
        for col in train_cols:
            if col not in test_cols:
                test[col] = 0
        
        # Reorder columns in test set to match training set
        test = test[train_cols]
        
        # Return modified test set
        return test

    # Match the columns
    val_set = match_columns(train_set, val_set)

    # print(train_set.columns.tolist())
    train_set.corr()[TARGET].sort_values(ascending=False)

    from models.knn import KNN
    from models.bayes import NaiveBayes
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    knn = KNN(k=5, n_jobs=-1, p=2, metric='manhattan')
    # nb = NaiveBayes()
    # knn_lib = KNeighborsClassifier(n_neighbors=3)
    # nb_lib = GaussianNB()

    # clfs = [knn, nb, knn_lib, nb_lib]
    clfs = [knn]

    # Split predictor and target variable
    X_train = train_set.drop(TARGET, axis=1)
    y_train = train_set[TARGET]

    X_val = val_set.drop(TARGET, axis=1)
    y_val = val_set[TARGET]

    X_train, X_val = merge_and_cluster(X_train, X_val, n_clusters=9, random_state=RANDOM_STATE)

    for clf in clfs:
        clf.fit(X_train, y_train)

    train = pd.read_csv("train_data.csv")
    test = pd.read_csv("test_data.csv")

    # Save id
    test_id = test["id"]

    # Drop columns
    test.drop(columns=["id"], inplace=True)

    class FeatureImputerTest(BaseEstimator, TransformerMixin):
        def __init__(self, num_strategy="constant", num_fill_value=0, cat_strategy="constant", cat_fill_value="missing"):
            self.num_strategy = num_strategy
            self.num_fill_value = num_fill_value
            self.cat_strategy = cat_strategy
            self.cat_fill_value = cat_fill_value
            self.imputer = None

        def fit(self, X, y=None):
            # Separate numerical and categorical columns
            self.num_cols = NUMERICALFEATURES
            self.cat_cols = CATEGORICALFEATURES

            # Define imputers for numerical and categorical data
            self.imputer = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy=self.num_strategy, fill_value=self.num_fill_value), self.num_cols),
                    ("cat", SimpleImputer(strategy=self.cat_strategy, fill_value=self.cat_fill_value), self.cat_cols),
                ]
            )

            self.imputer.fit(X)
            return self

        def transform(self, X):
            # Apply imputations
            X_transformed = pd.DataFrame(self.imputer.transform(X), columns=self.num_cols + self.cat_cols)
            return X_transformed

    pipeline2 = Pipeline([
        ("imputer", FeatureImputerTest(num_strategy="constant", num_fill_value=0, cat_strategy="constant", cat_fill_value="missing")),  # Handle missing values
        ("featurecreator", FeatureCreator()),
        ("featurescaler", FeatureScaler()),
        ("dropper", FeatureDropper()),
        ("pca", CustomPCA(n_components=5))
    ])

    pipeline2.fit(train, train[TARGET])
    train = pipeline.fit_transform(train, train[TARGET])
    test = pipeline2.transform(test)

    test = match_columns(train, test)
    test.drop(columns={TARGET}, inplace=True)

    # Split predictor and target variable
    X_train = train.drop([TARGET], axis=1)
    y_train = train[TARGET]

    X_test = test
    X_train, X_test = merge_and_cluster(X_train, X_test, n_clusters=9, random_state=RANDOM_STATE)

    # nb = NaiveBayes()
    final_model = KNN(k=10, n_jobs=-1)

    # nb.fit(X_train, y_train)
    final_model.fit(X_train, y_train)

    # y_pred_nb = pd.Series(nb.predict(X_test))
    y_pred_final = final_model.predict(X_test)

    df_submission = pd.DataFrame(data={'id': test_id, 'attack_cat': y_pred_final})
    df_submission

    # y_pred_nb.value_counts()

    df_submission.to_csv("submissionbaru.csv", index=False)

    from sklearn.metrics import f1_score

    # Validation Predictions
    y_val_pred = final_model.predict(X_val)

    # Compute F1 Score on Validation Set
    f1_val = f1_score(y_val, y_val_pred, average='macro')
    print(f"F1 Score on Validation Set: {f1_val:.4f}")