import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
import logging
from scipy.stats import skew

logger = logging.getLogger(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    A comprehensive Preprocessing pipeline for the Diabetes Prediction Challenge.
    
    Features:
    - Automated handling of numerical and categorical columns.
    - Missing value imputation.
    - Scaling (Standard or PowerTransform).
    - Encoding for categorical variables.
    """
    
    def __init__(self, target_col: str = 'diagnosed_diabetes', skew_threshold: float = 1.0):
        self.target_col = target_col
        self.skew_threshold = skew_threshold
        self.pipeline = None
        self.num_cols = None
        self.skewed_cols = None
        self.stable_cols = None
        self.cat_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting Preprocessor...")
        
        X_clean = X.drop(columns=[self.target_col], errors='ignore')
        if 'id' in X_clean.columns:
            X_clean = X_clean.drop(columns=['id'])
        
        # Identify numeric & categorical columns
        self.num_cols = X_clean.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Identify skewed numeric columns
        skew_values = X_clean[self.num_cols].apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skew_values[abs(skew_values) > self.skew_threshold].index.tolist()
        self.stable_cols = [c for c in self.num_cols if c not in self.skewed_cols]
        
        logger.info(f"Numeric Columns: {len(self.num_cols)}, Skewed: {self.skewed_cols}, Stable: {self.stable_cols}")
        logger.info(f"Categorical Columns: {self.cat_cols}")
        
        # Build transformers
        transformers = []

        if self.skewed_cols:
            skew_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('power', PowerTransformer(method='yeo-johnson'))
            ])
            transformers.append(('skew_num', skew_pipeline, self.skewed_cols))
        
        if self.stable_cols:
            stable_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('stable_num', stable_pipeline, self.stable_cols))
        
        if self.cat_cols:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipeline, self.cat_cols))
        
        self.pipeline = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        self.pipeline.fit(X_clean, y)
        
        logger.info("Preprocessor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming data...")
        X_clean = X.drop(columns=[self.target_col, 'id'], errors='ignore')
        X_processed = self.pipeline.transform(X_clean)
        feature_names = self.pipeline.get_feature_names_out()
        return pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import DataLoader
    
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    train_df, _, _ = loader.load_all()
    
    preprocessor = Preprocessor(skew_threshold=1.0)
    X_processed = preprocessor.fit_transform(train_df)
    
    print("Processed Data Shape:", X_processed.shape)
    print("Processed Data Head:")
    print(X_processed.head())