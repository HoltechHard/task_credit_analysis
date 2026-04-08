"""
classifiers.py
==============
Classification models for Credit Score Classification project.
Includes: KNN, SVM, Random Forest, XGBoost, LightGBM.
Each model uses GridSearchCV for hyperparameter optimization.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm


class KNNModel:
    """K-Nearest Neighbors Classifier with hyperparameter optimization."""

    def __init__(self):
        self.params = {
            "n_neighbors": [5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree"],
            "metric": ["euclidean", "manhattan"]
        }

    def train(self, x_train, y_train):
        print("Training KNN Classifier...")
        model = KNeighborsClassifier(n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"  Best KNN Parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
        return best_model


class SVMModel:
    """Support Vector Machine Classifier with hyperparameter optimization."""

    def __init__(self):
        self.params = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced"],
            "max_iter": [10000]
        }

    def train(self, x_train, y_train):
        print("Training SVM Classifier...")
        model = SVC()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"  Best SVM Parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
        return best_model


class RandomForestModel:
    """Random Forest Classifier with hyperparameter optimization."""

    def __init__(self):
        self.params = {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, 15],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced"]
        }

    def train(self, x_train, y_train):
        print("Training Random Forest Classifier...")
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"  Best Random Forest Parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
        return best_model


class XGBoostModel:
    """XGBoost Classifier with hyperparameter optimization."""

    def __init__(self):
        self.params = {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 10],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.8, 0.9],
            "use_label_encoder": [False],
            "eval_metric": ["mlogloss"]
        }

    def train(self, x_train, y_train):
        print("Training XGBoost Classifier...")

        # Encode labels to integers for XGBoost
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()
        y_train_enc = self.le.fit_transform(y_train)

        model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train_enc)

        best_model = grid_search.best_estimator_
        print(f"  Best XGBoost Parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
        return best_model


class LightGBMModel:
    """LightGBM Classifier with hyperparameter optimization."""

    def __init__(self):
        self.params = {
            "num_leaves": [15, 31, 50],
            "max_depth": [5, 10, 15],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100],
            "boosting_type": ["gbdt"],
            "class_weight": ["balanced"],
            "verbosity": [-1]
        }

    def train(self, x_train, y_train):
        print("Training LightGBM Classifier...")

        # Encode labels to integers for LightGBM
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()
        y_train_enc = self.le.fit_transform(y_train)

        model = lgbm.LGBMClassifier(random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train_enc)

        best_model = grid_search.best_estimator_
        print(f"  Best LightGBM Parameters: {grid_search.best_params_}")
        print(f"  Best CV Accuracy: {grid_search.best_score_:.4f}")
        return best_model
