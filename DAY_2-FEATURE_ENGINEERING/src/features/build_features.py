"""
Feature Engineering Pipeline
============================
Production-grade feature engineering for ML pipeline.

Features:
- Categorical encoding (OneHot, Label, Target)
- Numerical transformations (log, sqrt, power)
- Interaction and ratio features
- Aggregation features
- Statistical features
- Binning features

Day 2 - ML Engineering Week
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import yaml
import sys
from datetime import datetime
from dataclasses import dataclass, field

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.helpers import (
    save_json, 
    load_json, 
    create_directory, 
    timer_decorator,
    get_numerical_columns,
    get_categorical_columns,
    optimize_dtypes
)
from features.transformers import (
    LogTransformer,
    SqrtTransformer,
    PowerTransformer,
    InteractionTransformer,
    RatioTransformer,
    AggregationTransformer,
    BinningTransformer,
    StatisticalTransformer,
    FeatureTransformerPipeline
)

logger = get_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    # Transformations
    n_log_features: int = 10
    n_sqrt_features: int = 10
    n_power_features: int = 5
    power_degree: int = 2
    
    # Interactions
    max_interactions: int = 15
    max_ratios: int = 15
    interaction_top_features: int = 6
    
    # Aggregations
    create_aggregations: bool = True
    aggregation_functions: List[str] = field(
        default_factory=lambda: ['sum', 'mean', 'std', 'max', 'min', 'median']
    )
    
    # Binning
    create_bins: bool = True
    n_bins: int = 5
    binning_strategy: str = 'quantile'
    
    # Statistical
    create_statistical: bool = True


@dataclass 
class FeatureEngineeringResults:
    """Results from feature engineering."""
    original_features: int = 0
    final_features: int = 0
    new_features_created: int = 0
    features_list: List[str] = field(default_factory=list)
    transformations_applied: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# MAIN FEATURE ENGINEER CLASS
# =============================================================================

class FeatureEngineer:
    """
    Production Feature Engineering Pipeline.
    
    Features:
        - Categorical encoding (OneHot, Label)
        - Numerical transformations
        - Feature interactions
        - Aggregation features
        - 10+ new features generated
    
    Usage:
        >>> engineer = FeatureEngineer("src/config/config.yaml")
        >>> X_train, X_test, y_train, y_test = engineer.run_pipeline()
    """
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize Feature Engineer."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.fe_config = self._init_fe_config()
        
        # Store encoders and scalers for consistency
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.target_encoder: Optional[LabelEncoder] = None
        
        # Results tracking
        self.results = FeatureEngineeringResults()
        
        # Feature tracking
        self.original_features: List[str] = []
        self.engineered_features: List[str] = []
        
        logger.info("=" * 70)
        logger.info("🔧 FeatureEngineer Initialized")
        logger.info("=" * 70)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_fe_config(self) -> FeatureEngineeringConfig:
        """Initialize feature engineering configuration."""
        fe_cfg = self.config.get('feature_engineering', {})
        trans_cfg = fe_cfg.get('transformations', {})
        inter_cfg = fe_cfg.get('interactions', {})
        agg_cfg = fe_cfg.get('aggregations', {})
        bin_cfg = fe_cfg.get('binning', {})
        stat_cfg = fe_cfg.get('statistical', {})
        
        return FeatureEngineeringConfig(
            n_log_features=trans_cfg.get('log', {}).get('n_features', 10),
            n_sqrt_features=trans_cfg.get('sqrt', {}).get('n_features', 10),
            n_power_features=trans_cfg.get('power', {}).get('n_features', 5),
            power_degree=trans_cfg.get('power', {}).get('power', 2),
            max_interactions=inter_cfg.get('max_interactions', 15),
            max_ratios=fe_cfg.get('ratios', {}).get('max_ratios', 15),
            interaction_top_features=inter_cfg.get('top_features', 6),
            create_aggregations=agg_cfg.get('enabled', True),
            aggregation_functions=agg_cfg.get('functions', ['sum', 'mean', 'std', 'max', 'min']),
            create_bins=bin_cfg.get('enabled', True),
            n_bins=bin_cfg.get('n_bins', 5),
            binning_strategy=bin_cfg.get('strategy', 'quantile'),
            create_statistical=stat_cfg.get('enabled', True)
        )
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    @timer_decorator
    def load_data(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load train and test data from Day 1."""
        logger.info("📂 Loading train and test data...")
        
        train_path = Path(self.config['data']['input_train'])
        test_path = Path(self.config['data']['input_test'])
        
        if not train_path.exists():
            logger.error(f"❌ Train data not found: {train_path}")
            logger.info("   Please copy train.csv from Day 1 to inputs/ folder")
            return None
        
        if not test_path.exists():
            logger.error(f"❌ Test data not found: {test_path}")
            return None
        
        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)
        
        logger.info(f"   Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
        logger.info(f"   Test: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
        
        return train_df, test_df
    
    def separate_features_target(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]]:
        """Separate features and target variable."""
        logger.info("🎯 Separating features and target...")
        
        # Find target column
        target_candidates = self.config.get('target', {}).get(
            'candidates', 
            ['Attack', 'Label', 'attack', 'label', 'target']
        )
        
        target_col = None
        for col in target_candidates:
            if col in train_df.columns:
                target_col = col
                break
        
        if target_col is None:
            logger.error("❌ No target column found!")
            logger.info(f"   Available columns: {train_df.columns.tolist()[:10]}...")
            return None
        
        logger.info(f"   Target column: {target_col}")
        
        # Separate
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Store original features
        self.original_features = X_train.columns.tolist()
        self.results.original_features = len(self.original_features)
        
        # Encode target if string
        if y_train.dtype == 'object' or y_test.dtype == 'object':
            logger.info("   Encoding target variable...")
            self.target_encoder = LabelEncoder()
            y_train = pd.Series(
                self.target_encoder.fit_transform(y_train.astype(str)),
                index=y_train.index,
                name=target_col
            )
            y_test = pd.Series(
                self.target_encoder.transform(y_test.astype(str)),
                index=y_test.index,
                name=target_col
            )
            logger.info(f"   Target classes: {len(self.target_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test, target_col
    
    # =========================================================================
    # CATEGORICAL ENCODING
    # =========================================================================
    
    @timer_decorator
    def encode_categorical(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features."""
        cat_cols = get_categorical_columns(X_train)
        
        if len(cat_cols) == 0:
            logger.info("   No categorical features to encode")
            return X_train, X_test
        
        logger.info(f"📝 Encoding {len(cat_cols)} categorical features...")
        
        encoding_cfg = self.config.get('encoding', {})
        onehot_max = encoding_cfg.get('onehot', {}).get('max_categories', 10)
        
        for col in cat_cols:
            n_unique = X_train[col].nunique()
            
            if n_unique <= onehot_max:
                # One-Hot Encoding for low cardinality
                logger.debug(f"   OneHot encoding: {col} ({n_unique} categories)")
                
                dummies_train = pd.get_dummies(
                    X_train[col], 
                    prefix=col, 
                    drop_first=True
                )
                dummies_test = pd.get_dummies(
                    X_test[col], 
                    prefix=col, 
                    drop_first=True
                )
                
                # Align columns
                for c in dummies_train.columns:
                    if c not in dummies_test.columns:
                        dummies_test[c] = 0
                dummies_test = dummies_test[dummies_train.columns]
                
                X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)
                X_test = pd.concat([X_test.drop(columns=[col]), dummies_test], axis=1)
                
            else:
                # Label Encoding for high cardinality
                logger.debug(f"   Label encoding: {col} ({n_unique} categories)")
                
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                
                # Handle unseen categories in test
                X_test[col] = X_test[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else 'unknown'
                )
                le_classes = np.append(le.classes_, 'unknown')
                le.classes_ = le_classes
                X_test[col] = le.transform(X_test[col])
                
                self.label_encoders[col] = le
        
        self.results.transformations_applied.append('categorical_encoding')
        logger.info(f"   Encoded features: {X_train.shape[1]}")
        
        return X_train, X_test
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    @timer_decorator
    def create_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create new features using transformers."""
        logger.info("🔧 Creating new features...")
        
        initial_features = len(X_train.columns)
        
        # Get numerical columns for transformation
        num_cols = get_numerical_columns(X_train)
        logger.info(f"   Numerical features available: {len(num_cols)}")
        
        # Build transformer pipeline
        transformers = []
        
        # 1. Log Transformation
        log_transformer = LogTransformer(n_features=self.fe_config.n_log_features)
        log_transformer.fit(X_train)
        X_train = log_transformer.transform(X_train)
        X_test = log_transformer.transform(X_test)
        logger.info(f"   ✅ Log features: +{len(X_train.columns) - initial_features}")
        self.results.transformations_applied.append('log_transform')
        
        # 2. Sqrt Transformation
        current = len(X_train.columns)
        sqrt_transformer = SqrtTransformer(n_features=self.fe_config.n_sqrt_features)
        sqrt_transformer.fit(X_train)
        X_train = sqrt_transformer.transform(X_train)
        X_test = sqrt_transformer.transform(X_test)
        logger.info(f"   ✅ Sqrt features: +{len(X_train.columns) - current}")
        self.results.transformations_applied.append('sqrt_transform')
        
        # 3. Power Transformation
        current = len(X_train.columns)
        power_transformer = PowerTransformer(
            n_features=self.fe_config.n_power_features,
            power=self.fe_config.power_degree
        )
        power_transformer.fit(X_train)
        X_train = power_transformer.transform(X_train)
        X_test = power_transformer.transform(X_test)
        logger.info(f"   ✅ Power features: +{len(X_train.columns) - current}")
        self.results.transformations_applied.append('power_transform')
        
        # 4. Interaction Features
        current = len(X_train.columns)
        interaction_transformer = InteractionTransformer(
            max_interactions=self.fe_config.max_interactions,
            top_features=self.fe_config.interaction_top_features
        )
        interaction_transformer.fit(X_train)
        X_train = interaction_transformer.transform(X_train)
        X_test = interaction_transformer.transform(X_test)
        logger.info(f"   ✅ Interaction features: +{len(X_train.columns) - current}")
        self.results.transformations_applied.append('interactions')
        
        # 5. Ratio Features
        current = len(X_train.columns)
        ratio_transformer = RatioTransformer(
            max_ratios=self.fe_config.max_ratios,
            top_features=self.fe_config.interaction_top_features
        )
        ratio_transformer.fit(X_train)
        X_train = ratio_transformer.transform(X_train)
        X_test = ratio_transformer.transform(X_test)
        logger.info(f"   ✅ Ratio features: +{len(X_train.columns) - current}")
        self.results.transformations_applied.append('ratios')
        
        # 6. Aggregation Features
        if self.fe_config.create_aggregations:
            current = len(X_train.columns)
            agg_transformer = AggregationTransformer(
                n_features=15,
                functions=self.fe_config.aggregation_functions
            )
            agg_transformer.fit(X_train)
            X_train = agg_transformer.transform(X_train)
            X_test = agg_transformer.transform(X_test)
            logger.info(f"   ✅ Aggregation features: +{len(X_train.columns) - current}")
            self.results.transformations_applied.append('aggregations')
        
        # 7. Binning Features
        if self.fe_config.create_bins:
            current = len(X_train.columns)
            bin_transformer = BinningTransformer(
                n_features=10,
                n_bins=self.fe_config.n_bins,
                strategy=self.fe_config.binning_strategy
            )
            bin_transformer.fit(X_train)
            X_train = bin_transformer.transform(X_train)
            X_test = bin_transformer.transform(X_test)
            logger.info(f"   ✅ Binning features: +{len(X_train.columns) - current}")
            self.results.transformations_applied.append('binning')
        
        # 8. Statistical Features
        if self.fe_config.create_statistical:
            current = len(X_train.columns)
            stat_transformer = StatisticalTransformer(n_features=10)
            stat_transformer.fit(X_train)
            X_train = stat_transformer.transform(X_train)
            X_test = stat_transformer.transform(X_test)
            logger.info(f"   ✅ Statistical features: +{len(X_train.columns) - current}")
            self.results.transformations_applied.append('statistical')
        
        # Summary
        new_features = len(X_train.columns) - initial_features
        logger.info(f"   📊 Total new features created: {new_features}")
        
        self.results.new_features_created = new_features
        
        return X_train, X_test
    
    # =========================================================================
    # DATA CLEANING
    # =========================================================================
    
    @timer_decorator
    def handle_inf_nan(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle infinite and NaN values."""
        logger.info("🧹 Handling infinite and NaN values...")
        
        # Replace infinities
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Count NaNs
        train_nans = X_train.isnull().sum().sum()
        test_nans = X_test.isnull().sum().sum()
        logger.info(f"   NaN values - Train: {train_nans:,}, Test: {test_nans:,}")
        
        # Fill NaNs with median (calculated from train)
        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                median_val = X_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        # Verify no NaNs remain
        remaining_nans = X_train.isnull().sum().sum() + X_test.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"   ⚠️ Remaining NaNs: {remaining_nans}")
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
        
        logger.info("   ✅ All NaN/Inf values handled")
        
        return X_train, X_test
    
    # =========================================================================
    # SCALING
    # =========================================================================
    
    @timer_decorator
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features."""
        scaling_method = self.config.get('scaling', {}).get('method', 'standard')
        
        if scaling_method == 'none':
            logger.info("   Skipping scaling (configured as 'none')")
            return X_train, X_test
        
        logger.info(f"📏 Scaling features using {scaling_method}...")
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            logger.warning(f"   Unknown scaling method: {scaling_method}")
            return X_train, X_test
        
        # Fit on train, transform both
        columns = X_train.columns.tolist()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=columns, index=X_test.index)
        
        logger.info(f"   ✅ Scaled {len(columns)} features")
        
        return X_train, X_test
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    def save_engineered_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        target_col: str
    ) -> None:
        """Save engineered features to files."""
        logger.info("💾 Saving engineered features...")
        
        # Combine features and target
        train_df = pd.concat([
            X_train.reset_index(drop=True), 
            y_train.reset_index(drop=True)
        ], axis=1)
        test_df = pd.concat([
            X_test.reset_index(drop=True), 
            y_test.reset_index(drop=True)
        ], axis=1)
        
        # Get paths
        train_path = Path(self.config['data']['output_engineered_train'])
        test_path = Path(self.config['data']['output_engineered_test'])
        
        # Create directories
        create_directory(train_path.parent)
        
        # Save
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"   ✅ Train saved: {train_path} ({train_df.shape})")
        logger.info(f"   ✅ Test saved: {test_path} ({test_df.shape})")
        
        # Update results
        self.results.final_features = len(X_train.columns)
        self.results.features_list = X_train.columns.tolist()
        self.engineered_features = X_train.columns.tolist()
        
        # Save feature info
        feature_info = {
            'original_features': self.results.original_features,
            'final_features': self.results.final_features,
            'new_features_created': self.results.new_features_created,
            'feature_names': self.results.features_list,
            'transformations_applied': self.results.transformations_applied,
            'target_column': target_col,
            'timestamp': self.results.timestamp
        }
        
        info_path = self.config['data'].get(
            'engineered_info_path', 
            'src/features/engineered_features.json'
        )
        save_json(feature_info, info_path)
        logger.info(f"   ✅ Feature info saved: {info_path}")
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    @timer_decorator
    def run_pipeline(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Run complete feature engineering pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or None if failed
        """
        logger.info("=" * 70)
        logger.info("🚀 FEATURE ENGINEERING PIPELINE - STARTING")
        logger.info("=" * 70)
        
        try:
            # Step 1: Load data
            result = self.load_data()
            if result is None:
                return None
            train_df, test_df = result
            
            # Step 2: Separate features and target
            result = self.separate_features_target(train_df, test_df)
            if result is None:
                return None
            X_train, X_test, y_train, y_test, target_col = result
            
            logger.info(f"   Original features: {len(X_train.columns)}")
            
            # Step 3: Encode categorical
            X_train, X_test = self.encode_categorical(X_train, X_test)
            
            # Step 4: Create new features
            X_train, X_test = self.create_features(X_train, X_test)
            
            # Step 5: Handle inf/nan
            X_train, X_test = self.handle_inf_nan(X_train, X_test)
            
            # Step 6: Scale features (optional)
            # X_train, X_test = self.scale_features(X_train, X_test)
            
            # Step 7: Save results
            self.save_engineered_features(X_train, X_test, y_train, y_test, target_col)
            
            # Summary
            logger.info("=" * 70)
            logger.info("✅ FEATURE ENGINEERING COMPLETED")
            logger.info("=" * 70)
            logger.info(f"   Original features: {self.results.original_features}")
            logger.info(f"   Final features: {self.results.final_features}")
            logger.info(f"   New features created: {self.results.new_features_created}")
            logger.info(f"   Transformations: {', '.join(self.results.transformations_applied)}")
            logger.info("=" * 70)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Change to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    os.chdir(project_root)
    
    print(f"\n📁 Working directory: {os.getcwd()}\n")
    
    # Run pipeline
    engineer = FeatureEngineer("src/config/config.yaml")
    result = engineer.run_pipeline()
    
    if result:
        X_train, X_test, y_train, y_test = result
        print("\n" + "=" * 50)
        print("✅ Feature Engineering Completed Successfully!")
        print("=" * 50)
        print(f"\n📊 Results:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape: {y_test.shape}")
        print(f"\n📁 Output files:")
        print(f"   - src/data/processed/X_train_engineered.csv")
        print(f"   - src/data/processed/X_test_engineered.csv")
        print(f"   - src/features/engineered_features.json")
    else:
        print("\n❌ Feature engineering failed!")