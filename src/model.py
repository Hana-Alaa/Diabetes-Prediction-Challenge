import pandas as pd
import numpy as np
import optuna
import logging
import joblib
import os
from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A unified interface for training and tuning various machine learning models.
    Supports: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost.
    """
    
    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.params = params if params else {}
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on the name."""
        if self.model_name == 'logistic_regression':
            self.model = LogisticRegression(random_state=self.random_state, **self.params)
        elif self.model_name == 'random_forest':
            self.model = RandomForestClassifier(random_state=self.random_state, **self.params)
        elif self.model_name == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=self.random_state, **self.params)
        elif self.model_name == 'lightgbm':
            self.model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, **self.params)
        elif self.model_name == 'catboost':
            self.model = cb.CatBoostClassifier(random_state=self.random_state, verbose=0, **self.params)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds: int = 50):
        """Train the model. Supports early stopping for boosting models."""
        logger.info(f"Training {self.model_name}...")
        if self.model_name == 'xgboost' and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        elif self.model_name == 'lightgbm' and X_val is not None:
            # LightGBM 4.0+ requires callbacks for early stopping
            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=callbacks
            )
        elif self.model_name == 'catboost' and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities (for AUC)."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_val, y_val):
        """Evaluate the model with multiple metrics."""
        y_pred_proba = self.predict_proba(X_val)
        y_pred = self.predict(X_val)
        auc = roc_auc_score(y_val, y_pred_proba)
        ll = log_loss(y_val, y_pred_proba)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        logger.info(f"{self.model_name} | AUC: {auc:.4f}, LogLoss: {ll:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        return {'AUC': auc, 'LogLoss': ll, 'Precision': prec, 'Recall': rec}

    def feature_importance(self, feature_names):
        """Return feature importance if supported."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return None
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def optimize_hyperparameters(self, X, y, n_trials=10, cv_folds=3):
        """
        Optimize hyperparameters using Optuna.
        NOTE: Modified to use n_jobs=1 in CV to prevent Memory Errors.
        """
        logger.info(f"Starting Optuna optimization for {self.model_name} with {n_trials} trials...")
        
        # Suppress Optuna logs to avoid clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            if self.model_name == 'xgboost':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500), # Reduced max estimators for speed
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'random_state': self.random_state,
                    'n_jobs': 1 # internal parallelism
                }
                clf = xgb.XGBClassifier(**params)

            elif self.model_name == 'lightgbm':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_state,
                    'verbose': -1,
                    'n_jobs': 1
                }
                clf = lgb.LGBMClassifier(**params)

            elif self.model_name == 'catboost':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbose': 0,
                    'thread_count': 1
                }
                clf = cb.CatBoostClassifier(**params)

            elif self.model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': self.random_state,
                    'n_jobs': 1
                }
                clf = RandomForestClassifier(**params)
            else:
                return 0.5 # Baseline

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # CRITICAL FIX: n_jobs=1 prevents creating multiple heavy processes that consume all RAM
            scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials) # Optuna runs sequentially by default

        logger.info(f"Optimization finished. Best params: {study.best_params}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")

        self.params = study.best_params
        self._initialize_model()
        return study.best_params

    @staticmethod
    def optuna_tune_model(X, y, n_trials=5, cv_folds=3):
        """Try all supported models with Optuna and return best model name, params, and score."""
        models_to_try = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
        best_score = 0
        best_model_name = None
        best_params = None

        for model_name in models_to_try:
            try:
                logger.info(f"--- Tuning {model_name} ---")
                trainer = ModelTrainer(model_name=model_name)
                
                # Run optimization
                params = trainer.optimize_hyperparameters(X, y, n_trials=n_trials, cv_folds=cv_folds)
                
                # Validation check (reuse found params)
                trainer = ModelTrainer(model_name=model_name, params=params)
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                # n_jobs=1 for safety
                score = np.mean(cross_val_score(trainer.model, X, y, cv=cv, scoring='roc_auc', n_jobs=1))
                
                logger.info(f"{model_name} Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                import gc
                gc.collect() # Free memory
                continue

        return best_model_name, best_params, best_score

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor
    
    # 1. Load data
    loader = DataLoader()
    train_df, _, _ = loader.load_all()
    
    # 2. Preprocess
    preprocessor = Preprocessor(use_power_transform=True)
    X = preprocessor.fit_transform(train_df)
    y = train_df['diagnosed_diabetes']

    # 3. Full External Stratified K-Fold CV with dynamic tuning
    # We use 3 folds here to save time for the user, but 5 is standard
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_metrics = []
    fold_best_models = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"\n========== Fold {fold} ==========")
        logger.info(f"Hyperparameter tuning and model selection...")

        # Find best model for this fold
        # Reduced n_trials to 10 for speed/memory balance
        best_model_name, best_params, best_score = ModelTrainer.optuna_tune_model(X_train, y_train, n_trials=10, cv_folds=3)

        logger.info(f"Fold {fold} Winner: {best_model_name} | CV AUC: {best_score:.4f}")

        # Train the best model on this fold (with early stopping)
        trainer = ModelTrainer(model_name=best_model_name, params=best_params)
        trainer.train(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
        
        # Evaluate
        metrics = trainer.evaluate(X_val, y_val)
        fold_metrics.append(metrics)
        fold_best_models.append((best_model_name, best_params, best_score))

    # Average metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    print("\nCV Average Metrics:", avg_metrics)

    # 4. Determine overall best model
    model_scores = {}
    for model_name, params, score in fold_best_models:
        model_scores.setdefault(model_name, []).append(score)

    overall_best_model_name = max(model_scores, key=lambda k: np.mean(model_scores[k]))
    
    # Select best params (simple heuristic: take params from the highest scoring fold of that model)
    best_entry = max([m for m in fold_best_models if m[0] == overall_best_model_name], key=lambda x: x[2])
    overall_best_params = best_entry[1]

    logger.info(f"Overall Best Model: {overall_best_model_name}")
    logger.info(f"Mean CV AUC: {np.mean(model_scores[overall_best_model_name]):.4f}")

    # 5. Retrain best model on full data
    logger.info("Retraining best model on full dataset...")
    final_trainer = ModelTrainer(model_name=overall_best_model_name, params=overall_best_params)
    final_trainer.train(X, y)

    # 6. Feature importance
    if overall_best_model_name != 'logistic_regression':
        fi = final_trainer.feature_importance(X.columns)
        if fi is not None:
            print("\nTop 10 Features:")
            print(fi.head(10))
    
    # 7. Save model and preprocessor
    logger.info("Saving artifacts...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_trainer.model, "models/final_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Done!")