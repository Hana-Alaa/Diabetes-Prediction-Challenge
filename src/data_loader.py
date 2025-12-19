import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict

# Configure logging to show timestamp and level
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A robust data loader for the Diabetes Prediction Challenge.
    
    Handles loading of CSV files, memory optimization, and basic path validation.
    """

    def __init__(self, data_dir: str = "Data"):
        """
        Initialize the DataLoader.

        Args:
            data_dir (str): Path to the directory containing the dataset. Defaults to "Data".
        """
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        self.sample_sub_path = self.data_dir / "sample_submission.csv"

    def _check_file(self, path: Path) -> None:
        """Raises FileNotFoundError if the file does not exist."""
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes data types to reduce memory usage.
        
        - Downcasts integers and floats.
        - Converts object columns to category if unique values are low (heuristic).
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.select_dtypes(include=["int64", "int32"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64", "float32"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
            
        # Optional: Convert objects to category if efficient
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() / len(df) < 0.5:
                 df[col] = df[col].astype("category")

        end_mem = df.memory_usage().sum() / 1024**2
        logger.debug(f"Memory usage optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB")
        
        return df

    def _read_csv(self, path: Path, optimize: bool = True) -> pd.DataFrame:
        """Internal helper to read and optionally optimize a CSV file."""
        self._check_file(path)
        logger.info(f"Loading data from {path}...")
        df = pd.read_csv(path)
        if optimize:
            df = self._optimize_dtypes(df)
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads train, test, and sample submission files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, test, sample_submission)
        """
        train_df = self._read_csv(self.train_path)
        test_df = self._read_csv(self.test_path)
        sample_sub = self._read_csv(self.sample_sub_path, optimize=False) # No need to optimize sample sub usually

        logger.info(f"Successfully loaded all datasets.")
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")

        return train_df, test_df, sample_sub

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads only train and test datasets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train, test)
        """
        train_df = self._read_csv(self.train_path)
        test_df = self._read_csv(self.test_path)
        
        logger.info(f"Successfully loaded train and test datasets.")
        return train_df, test_df


if __name__ == "__main__":
    # Example usage
    try:
        loader = DataLoader()
        train, test, sub = loader.load_all()
        print(train.head())
    except Exception as e:
        logger.critical(f"Failed to load data: {e}")
