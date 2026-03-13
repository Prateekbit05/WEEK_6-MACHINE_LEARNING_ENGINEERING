"""
Production Data Pipeline
========================
Optimized data pipeline for large-scale network intrusion detection.

Features:
- Memory-efficient chunk processing
- Parallel processing support
- Comprehensive data cleaning
- Outlier detection & treatment
- Class imbalance handling (SMOTE)
- Data versioning
- Detailed logging
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import yaml
import gc
import sys
import warnings
from datetime import datetime
from dataclasses import dataclass, field

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.helpers import (
    calculate_file_hash,
    calculate_dataframe_hash,
    save_json,
    load_json,
    get_data_info,
    optimize_dtypes,
    timer_decorator,
    force_garbage_collection,
    detect_outliers_iqr,
    detect_outliers_zscore,
    create_directory,
    validate_dataframe,
    ProgressTracker
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# DATA CLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data configuration container."""
    raw_path: str
    processed_path: str
    sample_path: str
    train_path: str
    test_path: str
    validation_path: str
    chunk_size: int = 50000
    sample_size: int = 100000
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    use_sampling: bool = True
    stratify: bool = True


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration container."""
    numerical_strategy: str = "median"
    categorical_strategy: str = "mode"
    drop_threshold: float = 0.5
    outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    outlier_treatment: str = "cap"
    scaling_method: str = "standard"
    handle_imbalance: bool = True
    imbalance_method: str = "smote"
    min_samples_per_class: int = 10


@dataclass
class PipelineMetadata:
    """Pipeline execution metadata."""
    pipeline_version: str = "1.0.0"
    execution_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data_hash: Optional[str] = None
    rows_processed: int = 0
    columns_processed: int = 0
    processing_time_seconds: float = 0.0
    status: str = "initialized"


# =============================================================================
# MAIN DATA PIPELINE CLASS
# =============================================================================

class DataPipeline:
    """
    Production-grade data pipeline for ML projects.
    
    Features:
        - Chunked processing for large files
        - Memory optimization
        - Comprehensive data cleaning
        - Outlier detection & treatment
        - Class imbalance handling
        - Data versioning
        - Detailed logging & metrics
    
    Usage:
        >>> pipeline = DataPipeline("src/config/config.yaml")
        >>> pipeline.run_pipeline()
    """
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        Initialize the data pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize configurations
        self.data_config = self._init_data_config()
        self.preprocess_config = self._init_preprocessing_config()
        
        # Initialize metadata
        self.metadata = PipelineMetadata()
        
        # Initialize scalers and encoders
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = []
        self.target_column = None
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("=" * 70)
        logger.info("🚀 DataPipeline Initialized")
        logger.info(f"   Config: {self.config_path}")
        logger.info(f"   Raw data: {self.data_config.raw_path}")
        logger.info("=" * 70)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"Configuration loaded from {self.config_path}")
        return config
    
    def _init_data_config(self) -> DataConfig:
        """Initialize data configuration."""
        data_cfg = self.config.get('data', {})
        return DataConfig(
            raw_path=data_cfg.get('raw_path', 'src/data/raw/data.csv'),
            processed_path=data_cfg.get('processed_path', 'src/data/processed/final.csv'),
            sample_path=data_cfg.get('sample_path', 'src/data/processed/sample.csv'),
            train_path=data_cfg.get('train_path', 'src/data/processed/train.csv'),
            test_path=data_cfg.get('test_path', 'src/data/processed/test.csv'),
            validation_path=data_cfg.get('validation_path', 'src/data/processed/validation.csv'),
            chunk_size=data_cfg.get('chunk_size', 50000),
            sample_size=data_cfg.get('sample_size', 100000),
            test_size=data_cfg.get('test_size', 0.2),
            validation_size=data_cfg.get('validation_size', 0.1),
            random_state=data_cfg.get('random_state', 42),
            use_sampling=data_cfg.get('use_sampling', True),
            stratify=data_cfg.get('stratify', True)
        )
    
    def _init_preprocessing_config(self) -> PreprocessingConfig:
        """Initialize preprocessing configuration."""
        prep_cfg = self.config.get('preprocessing', {})
        missing_cfg = prep_cfg.get('missing_values', {})
        outlier_cfg = prep_cfg.get('outliers', {})
        scaling_cfg = prep_cfg.get('scaling', {})
        imbalance_cfg = prep_cfg.get('imbalance', {})
        
        return PreprocessingConfig(
            numerical_strategy=missing_cfg.get('numerical_strategy', 'median'),
            categorical_strategy=missing_cfg.get('categorical_strategy', 'mode'),
            drop_threshold=missing_cfg.get('drop_threshold', 0.5),
            outlier_method=outlier_cfg.get('detection_method', 'iqr'),
            iqr_multiplier=outlier_cfg.get('iqr_multiplier', 1.5),
            zscore_threshold=outlier_cfg.get('zscore_threshold', 3.0),
            outlier_treatment=outlier_cfg.get('treatment', 'cap'),
            scaling_method=scaling_cfg.get('method', 'standard'),
            handle_imbalance=imbalance_cfg.get('handle', True),
            imbalance_method=imbalance_cfg.get('method', 'smote'),
            min_samples_per_class=imbalance_cfg.get('min_samples_per_class', 10)
        )
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            Path(self.data_config.processed_path).parent,
            Path(self.config.get('logging', {}).get('log_dir', 'src/logs')),
            Path(self.config.get('output', {}).get('reports_dir', 'src/reports')),
            Path(self.config.get('versioning', {}).get('metadata_path', 'src/data/metadata'))
        ]
        
        for directory in directories:
            create_directory(directory)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    @timer_decorator
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Analyze dataset structure without loading entire file.
        
        Returns:
            Dictionary with dataset information
        """
        logger.info("📊 Analyzing dataset structure...")
        
        filepath = Path(self.data_config.raw_path)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        # Get file size
        file_size_gb = filepath.stat().st_size / (1024**3)
        logger.info(f"   File size: {file_size_gb:.2f} GB")
        
        # Read sample to get structure
        sample = pd.read_csv(filepath, nrows=1000, low_memory=False)
        
        # Count total rows (approximate for large files)
        if file_size_gb > 1:
            # Estimate based on sample
            sample_size_bytes = sample.memory_usage(deep=True).sum()
            estimated_rows = int(filepath.stat().st_size / (sample_size_bytes / 1000))
            logger.info(f"   Estimated rows: ~{estimated_rows:,}")
        else:
            # Count actual rows
            with open(filepath, 'r') as f:
                estimated_rows = sum(1 for _ in f) - 1
            logger.info(f"   Total rows: {estimated_rows:,}")
        
        info = {
            'file_path': str(filepath),
            'file_size_gb': round(file_size_gb, 2),
            'estimated_rows': estimated_rows,
            'columns': sample.columns.tolist(),
            'dtypes': sample.dtypes.astype(str).to_dict(),
            'n_columns': len(sample.columns),
            'sample_memory_mb': round(sample.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Identify target column
        target_candidates = self.config.get('features', {}).get(
            'target_candidates', 
            ['Attack', 'Label', 'attack', 'label', 'target']
        )
        
        for candidate in target_candidates:
            if candidate in sample.columns:
                info['target_column'] = candidate
                info['target_classes'] = sample[candidate].nunique()
                logger.info(f"   Target column: {candidate} ({info['target_classes']} classes)")
                break
        
        logger.info(f"   Columns: {info['n_columns']}")
        
        return info
    
    @timer_decorator
    def create_sample(self, force: bool = False) -> pd.DataFrame:
        """
        Create a representative sample from large dataset.
        
        Args:
            force: Force recreation even if sample exists
        
        Returns:
            Sampled DataFrame
        """
        sample_path = Path(self.data_config.sample_path)
        
        # Check if sample already exists
        if sample_path.exists() and not force:
            logger.info(f"📂 Loading existing sample: {sample_path}")
            df = pd.read_csv(sample_path, low_memory=False)
            logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            return df
        
        logger.info(f"📊 Creating sample of {self.data_config.sample_size:,} rows...")
        
        filepath = Path(self.data_config.raw_path)
        sample_size = self.data_config.sample_size
        chunk_size = self.data_config.chunk_size
        
        chunks = []
        rows_collected = 0
        chunk_num = 0
        
        # Progress tracking
        progress = ProgressTracker(sample_size, "Sampling")
        
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            chunk_num += 1
            
            # Calculate how many rows to sample from this chunk
            remaining = sample_size - rows_collected
            if remaining <= 0:
                break
            
            # Sample proportionally or take what's needed
            chunk_sample_size = min(remaining, len(chunk))
            
            if chunk_sample_size > 0:
                # Random sampling from chunk
                chunk_sample = chunk.sample(
                    n=chunk_sample_size, 
                    random_state=self.data_config.random_state
                )
                chunks.append(chunk_sample)
                rows_collected += len(chunk_sample)
                
                progress.update(len(chunk_sample))
            
            # Memory management
            if chunk_num % 10 == 0:
                force_garbage_collection()
        
        # Combine all chunks
        sample_df = pd.concat(chunks, ignore_index=True)
        
        # Shuffle the combined sample
        sample_df = sample_df.sample(frac=1, random_state=self.data_config.random_state).reset_index(drop=True)
        
        # Save sample
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_path, index=False)
        
        logger.info(f"✅ Sample created: {sample_df.shape[0]:,} rows × {sample_df.shape[1]} columns")
        logger.info(f"   Saved to: {sample_path}")
        
        # Update metadata
        self.metadata.rows_processed = len(sample_df)
        self.metadata.columns_processed = len(sample_df.columns)
        
        return sample_df
    
    # =========================================================================
    # DATA CLEANING
    # =========================================================================
    
    @timer_decorator
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Steps:
            1. Remove duplicates
            2. Handle missing values
            3. Handle infinite values
            4. Detect and treat outliers
            5. Optimize data types
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Starting data cleaning...")
        initial_shape = df.shape
        
        df = df.copy()
        
        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 2: Drop high-missing columns
        df = self._drop_high_missing_columns(df)
        
        # Step 3: Handle infinite values
        df = self._handle_infinite_values(df)
        
        # Step 4: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 5: Detect and treat outliers
        df = self._handle_outliers(df)
        
        # Step 6: Optimize data types
        df = optimize_dtypes(df, verbose=True)
        
        logger.info(f"✅ Cleaning complete: {initial_shape} → {df.shape}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_len = len(df)
        df = df.drop_duplicates(keep='first')
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.info(f"   Removed {removed:,} duplicate rows ({removed/initial_len*100:.2f}%)")
        else:
            logger.info("   No duplicates found")
        
        return df
    
    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with missing values above threshold."""
        threshold = self.preprocess_config.drop_threshold
        
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > threshold].index.tolist()
        
        if high_missing_cols:
            logger.info(f"   Dropping {len(high_missing_cols)} columns with >{threshold*100}% missing")
            df = df.drop(columns=high_missing_cols)
        
        return df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_count = np.isinf(df[numerical_cols]).sum().sum()
        
        if inf_count > 0:
            df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
            logger.info(f"   Replaced {inf_count:,} infinite values with NaN")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        # Numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            strategy = self.preprocess_config.numerical_strategy
            
            if strategy == "median":
                df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            elif strategy == "mean":
                df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            elif strategy == "zero":
                df[numerical_cols] = df[numerical_cols].fillna(0)
            
            logger.info(f"   Filled numerical missing values with {strategy}")
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            strategy = self.preprocess_config.categorical_strategy
            
            for col in categorical_cols:
                if df[col].isnull().any():
                    if strategy == "mode":
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val[0])
                    elif strategy == "unknown":
                        df[col] = df[col].fillna("Unknown")
            
            logger.info(f"   Filled categorical missing values with {strategy}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and treat outliers."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        treatment = self.preprocess_config.outlier_treatment
        
        if treatment == "none":
            return df
        
        outliers_treated = 0
        
        for col in numerical_cols:
            if self.preprocess_config.outlier_method == "iqr":
                result = detect_outliers_iqr(
                    df[col].dropna(), 
                    multiplier=self.preprocess_config.iqr_multiplier
                )
            else:
                result = detect_outliers_zscore(
                    df[col].dropna(),
                    threshold=self.preprocess_config.zscore_threshold
                )
            
            if result['outlier_count'] > 0:
                outliers_treated += result['outlier_count']
                
                if treatment == "cap":
                    # Cap outliers at bounds
                    if 'lower_bound' in result:
                        df[col] = df[col].clip(
                            lower=result['lower_bound'],
                            upper=result['upper_bound']
                        )
                    else:
                        # For z-score, cap at mean ± 3*std
                        mean, std = df[col].mean(), df[col].std()
                        df[col] = df[col].clip(
                            lower=mean - 3*std,
                            upper=mean + 3*std
                        )
        
        if outliers_treated > 0:
            logger.info(f"   Treated {outliers_treated:,} outliers using {treatment} method")
        
        return df
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    def _identify_target_column(self, df: pd.DataFrame) -> str:
        """Identify the target column."""
        target_col = self.config.get('features', {}).get('target_column', 'Attack')
        target_candidates = self.config.get('features', {}).get(
            'target_candidates',
            ['Attack', 'Label', 'attack', 'label', 'target', 'class']
        )
        
        if target_col in df.columns:
            return target_col
        
        for candidate in target_candidates:
            if candidate in df.columns:
                logger.info(f"   Found target column: {candidate}")
                return candidate
        
        raise ValueError(f"No target column found. Available columns: {df.columns.tolist()[:10]}...")
    
    def _prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling."""
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Keep only numerical features (for initial modeling)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numerical_cols]
        
        # Store feature columns
        self.feature_columns = numerical_cols
        self.target_column = target_col
        
        logger.info(f"   Features: {len(numerical_cols)} numerical columns")
        logger.info(f"   Target: {target_col} ({y.nunique()} classes)")
        
        return X, y
    
    def _handle_rare_classes(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        min_samples: int = 2
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove classes with too few samples."""
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < min_samples].index.tolist()
        
        if rare_classes:
            logger.warning(f"   Removing {len(rare_classes)} rare classes with < {min_samples} samples")
            mask = ~y.isin(rare_classes)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
        
        return X, y
    
    # =========================================================================
    # CLASS IMBALANCE HANDLING
    # =========================================================================
    
    @timer_decorator
    def handle_imbalance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE or other methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        
        Returns:
            Resampled X and y
        """
        if not self.preprocess_config.handle_imbalance:
            return X, y
        
        method = self.preprocess_config.imbalance_method
        
        logger.info(f"⚖️ Handling class imbalance using {method}...")
        
        # Calculate class distribution
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        logger.info(f"   Initial imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio < 3:
            logger.info("   Classes are relatively balanced, skipping resampling")
            return X, y
        
        # Check minimum samples per class
        min_samples = self.preprocess_config.min_samples_per_class
        small_classes = class_counts[class_counts < min_samples].index.tolist()
        
        if small_classes:
            logger.warning(f"   Removing {len(small_classes)} classes with < {min_samples} samples")
            mask = ~y.isin(small_classes)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
        
        try:
            if method == "smote":
                from imblearn.over_sampling import SMOTE
                
                # Adjust k_neighbors based on smallest class
                min_class_size = y.value_counts().min()
                k_neighbors = min(5, min_class_size - 1)
                
                if k_neighbors < 1:
                    logger.warning("   Not enough samples for SMOTE, using random oversampling")
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(random_state=self.data_config.random_state)
                else:
                    sampler = SMOTE(
                        k_neighbors=k_neighbors,
                        random_state=self.data_config.random_state,
                        n_jobs=-1
                    )
                
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
            elif method == "adasyn":
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(random_state=self.data_config.random_state, n_jobs=-1)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
            elif method == "random_oversample":
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=self.data_config.random_state)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
            elif method == "random_undersample":
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=self.data_config.random_state)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
            else:
                logger.warning(f"   Unknown imbalance method: {method}, skipping")
                return X, y
            
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name)
            
            new_imbalance = y_resampled.value_counts().max() / y_resampled.value_counts().min()
            logger.info(f"   Resampled: {len(X):,} → {len(X_resampled):,} samples")
            logger.info(f"   New imbalance ratio: {new_imbalance:.2f}:1")
            
            return X_resampled, y_resampled
            
        except ImportError:
            logger.warning("   imblearn not installed. Install with: pip install imbalanced-learn")
            logger.warning("   Skipping imbalance handling")
            return X, y
        except Exception as e:
            logger.error(f"   Error handling imbalance: {e}")
            return X, y
    
    # =========================================================================
    # TRAIN/TEST SPLIT
    # =========================================================================
    
    @timer_decorator
    def create_train_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        apply_imbalance_handling: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split with proper stratification.
        
        Args:
            df: Input DataFrame (if None, loads from sample)
            apply_imbalance_handling: Whether to apply SMOTE/etc.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("📊 Creating train/test split...")
        
        # Load data if not provided
        if df is None:
            sample_path = Path(self.data_config.sample_path)
            if not sample_path.exists():
                self.create_sample()
            df = pd.read_csv(sample_path, low_memory=False)
        
        logger.info(f"   Input data: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Clean data
        df = self.clean_data(df)
        
        # Identify target
        target_col = self._identify_target_column(df)
        
        # Prepare features
        X, y = self._prepare_features(df, target_col)
        
        # Handle rare classes
        X, y = self._handle_rare_classes(X, y, min_samples=2)
        
        # Log class distribution
        logger.info("   Class distribution:")
        for cls, count in y.value_counts().head(10).items():
            pct = count / len(y) * 100
            logger.info(f"      {cls}: {count:,} ({pct:.2f}%)")
        
        # Determine if stratification is possible
        class_counts = y.value_counts()
        use_stratify = (
            self.data_config.stratify and 
            y.nunique() < 100 and 
            class_counts.min() >= 2
        )
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.data_config.test_size,
                random_state=self.data_config.random_state,
                stratify=y if use_stratify else None,
                shuffle=True
            )
            logger.info(f"   Split successful (stratified={use_stratify})")
            
        except ValueError as e:
            logger.warning(f"   Stratified split failed: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.data_config.test_size,
                random_state=self.data_config.random_state,
                stratify=None,
                shuffle=True
            )
            logger.info("   Split successful (without stratification)")
        
        # Handle imbalance on training data only
        if apply_imbalance_handling and self.preprocess_config.handle_imbalance:
            X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        # Reset indices
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        logger.info(f"   Train set: {len(X_train):,} samples")
        logger.info(f"   Test set: {len(X_test):,} samples")
        
        # Save datasets
        self._save_datasets(X_train, X_test, y_train, y_test, target_col)
        
        return X_train, X_test, y_train, y_test
    
    def _save_datasets(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        target_col: str
    ) -> None:
        """Save train and test datasets."""
        # Combine features and target
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save paths
        train_path = Path(self.data_config.train_path)
        test_path = Path(self.data_config.test_path)
        final_path = Path(self.data_config.processed_path)
        
        # Create directories
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save files
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save combined final.csv
        final_df = pd.concat([train_df, test_df], ignore_index=True)
        final_df.to_csv(final_path, index=False)
        
        logger.info(f"✅ Saved: {train_path} ({train_df.shape})")
        logger.info(f"✅ Saved: {test_path} ({test_df.shape})")
        logger.info(f"✅ Saved: {final_path} ({final_df.shape})")
        
        # Save feature information
        feature_info = {
            'features': self.feature_columns,
            'target_column': target_col,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(train_df),
            'n_test_samples': len(test_df),
            'class_distribution': y_train.value_counts().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        save_json(feature_info, "src/data/processed/feature_info.json")
    
    # =========================================================================
    # DATA VERSIONING
    # =========================================================================
    
    def version_data(self, df: pd.DataFrame) -> str:
        """
        Create a version hash for the dataset.
        
        Args:
            df: DataFrame to version
        
        Returns:
            Version hash string
        """
        data_hash = calculate_dataframe_hash(df)
        
        version_info = {
            'hash': data_hash,
            'timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        }
        
        metadata_path = Path(self.config.get('versioning', {}).get(
            'metadata_path', 
            'src/data/metadata'
        ))
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        save_json(version_info, metadata_path / f"version_{data_hash[:8]}.json")
        
        logger.info(f"📌 Data version: {data_hash[:8]}")
        
        return data_hash
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    @timer_decorator
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete data pipeline.
        
        Returns:
            Pipeline execution results
        """
        start_time = datetime.now()
        
        logger.info("=" * 70)
        logger.info("🚀 DAY 1 DATA PIPELINE - STARTING")
        logger.info("=" * 70)
        
        results = {
            'status': 'started',
            'start_time': start_time.isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Analyze dataset
            logger.info("\n📋 Step 1: Analyzing dataset...")
            info = self.get_dataset_info()
            save_json(info, "src/data/dataset_info.json")
            results['steps']['dataset_info'] = 'completed'
            
            # Step 2: Create sample
            logger.info("\n📋 Step 2: Creating sample...")
            sample = self.create_sample()
            results['steps']['sampling'] = 'completed'
            
            # Step 3: Version data
            logger.info("\n📋 Step 3: Versioning data...")
            data_hash = self.version_data(sample)
            results['data_hash'] = data_hash
            results['steps']['versioning'] = 'completed'
            
            # Step 4: Create train/test split
            logger.info("\n📋 Step 4: Creating train/test split...")
            X_train, X_test, y_train, y_test = self.create_train_test_split(sample)
            results['steps']['train_test_split'] = 'completed'
            
            # Final results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results['status'] = 'completed'
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = duration
            results['train_samples'] = len(X_train)
            results['test_samples'] = len(X_test)
            results['n_features'] = len(self.feature_columns)
            results['n_classes'] = y_train.nunique()
            
            # Save pipeline results
            save_json(results, "src/data/processed/pipeline_results.json")
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ DAY 1 DATA PIPELINE - COMPLETED")
            logger.info("=" * 70)
            logger.info(f"   Duration: {duration:.2f} seconds")
            logger.info(f"   Train samples: {len(X_train):,}")
            logger.info(f"   Test samples: {len(X_test):,}")
            logger.info(f"   Features: {len(self.feature_columns)}")
            logger.info(f"   Classes: {y_train.nunique()}")
            logger.info("=" * 70)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            save_json(results, "src/data/processed/pipeline_results.json")
            raise


# =============================================================================
# BACKWARD COMPATIBILITY - LargeDataPipeline alias
# =============================================================================

class LargeDataPipeline(DataPipeline):
    """Alias for DataPipeline for backward compatibility."""
    pass


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 DAY 1 - DATA PIPELINE EXECUTION")
    print("=" * 70 + "\n")
    
    # Initialize and run pipeline
    pipeline = DataPipeline("src/config/config.yaml")
    results = pipeline.run_pipeline()
    
    print("\n✅ Day 1 Pipeline Completed Successfully!")
    print(f"   Results saved to: src/data/processed/pipeline_results.json")