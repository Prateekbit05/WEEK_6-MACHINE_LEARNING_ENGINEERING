"""
Feature Selection Pipeline
==========================
Production-grade feature selection using multiple methods.

Methods:
- Variance Threshold
- Correlation Filter
- Mutual Information
- Tree-based Importance
- Recursive Feature Elimination (RFE)
- Ensemble Voting

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

from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    RFE,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.helpers import save_json, create_directory, timer_decorator

logger = get_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SelectionConfig:
    """Configuration for feature selection."""
    n_features: int = 50
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    min_votes: int = 2


@dataclass
class SelectionResults:
    """Results from feature selection."""
    initial_features: int = 0
    final_features: int = 0
    features_removed: int = 0
    selected_features: List[str] = field(default_factory=list)
    feature_scores: Dict[str, Dict] = field(default_factory=dict)
    methods_used: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# MAIN FEATURE SELECTOR CLASS
# =============================================================================

class FeatureSelector:
    """
    Production Feature Selection Pipeline.
    
    Methods:
        - Variance Threshold
        - Correlation Filter
        - Mutual Information
        - Tree-based Selection
        - RFE (Recursive Feature Elimination)
        - Ensemble Voting
    
    Usage:
        >>> selector = FeatureSelector("src/config/config.yaml")
        >>> X_train, X_test, y_train, y_test = selector.run_selection_pipeline()
    """
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize Feature Selector."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.selection_config = self._init_selection_config()
        
        # Encoders
        self.label_encoder = LabelEncoder()
        
        # Results tracking
        self.results = SelectionResults()
        self.feature_scores: Dict[str, Dict] = {}
        
        # Plots directory
        self.plots_dir = Path(self.config.get('output', {}).get('plots_dir', 'src/notebooks/plots'))
        create_directory(self.plots_dir)
        
        logger.info("=" * 70)
        logger.info("🎯 FeatureSelector Initialized")
        logger.info(f"   Target features: {self.selection_config.n_features}")
        logger.info("=" * 70)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_selection_config(self) -> SelectionConfig:
        """Initialize selection configuration."""
        fs_cfg = self.config.get('feature_selection', {})
        
        return SelectionConfig(
            n_features=fs_cfg.get('n_features', 50),
            variance_threshold=fs_cfg.get('variance', {}).get('threshold', 0.01),
            correlation_threshold=fs_cfg.get('correlation', {}).get('threshold', 0.95),
            min_votes=fs_cfg.get('voting', {}).get('min_votes', 2)
        )
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    @timer_decorator
    def load_engineered_data(self) -> Optional[Tuple]:
        """Load engineered features from previous step."""
        logger.info("📂 Loading engineered features...")
        
        train_path = Path(self.config['data']['output_engineered_train'])
        test_path = Path(self.config['data']['output_engineered_test'])
        
        if not train_path.exists():
            logger.error(f"❌ Engineered train data not found: {train_path}")
            logger.info("   Run build_features.py first!")
            return None
        
        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)
        
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
            return None
        
        # Separate features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Clean target - handle NaN and mixed types
        logger.info("   Cleaning target variable...")
        
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        
        X_train = X_train[valid_train].reset_index(drop=True)
        y_train = y_train[valid_train].reset_index(drop=True)
        X_test = X_test[valid_test].reset_index(drop=True)
        y_test = y_test[valid_test].reset_index(drop=True)
        
        # Encode target
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        y_train = pd.Series(y_train_encoded, name=target_col)
        y_test = pd.Series(y_test_encoded, name=target_col)
        
        logger.info(f"   Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
        logger.info(f"   Test: {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
        logger.info(f"   Target classes: {len(self.label_encoder.classes_)}")
        
        self.results.initial_features = len(X_train.columns)
        
        return X_train, X_test, y_train, y_test, target_col
    
    # =========================================================================
    # SELECTION METHODS
    # =========================================================================
    
    @timer_decorator
    def remove_low_variance(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove features with low variance."""
        threshold = self.selection_config.variance_threshold
        logger.info(f"📉 Removing features with variance < {threshold}...")
        
        initial = len(X_train.columns)
        
        # Calculate variance
        variances = X_train.var()
        variances = variances.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Keep features above threshold
        keep_features = variances[variances >= threshold].index.tolist()
        
        removed = initial - len(keep_features)
        logger.info(f"   Removed {removed} low-variance features")
        
        self.results.methods_used.append('variance_threshold')
        
        return X_train[keep_features], X_test[keep_features]
    
    @timer_decorator
    def remove_correlated(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove highly correlated features."""
        threshold = self.selection_config.correlation_threshold
        logger.info(f"🔗 Removing features with correlation > {threshold}...")
        
        initial = len(X_train.columns)
        
        # Calculate correlation matrix
        corr_matrix = X_train.corr().abs()
        corr_matrix = corr_matrix.fillna(0)
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        
        keep_features = [col for col in X_train.columns if col not in to_drop]
        
        removed = initial - len(keep_features)
        logger.info(f"   Removed {removed} correlated features")
        
        self.results.methods_used.append('correlation_filter')
        
        # Save correlation heatmap
        self._plot_correlation_matrix(X_train[keep_features[:30]])
        
        return X_train[keep_features], X_test[keep_features]
    
    @timer_decorator
    def mutual_information_selection(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[Dict[str, float], List[str]]:
        """Select features based on mutual information."""
        logger.info("📊 Computing mutual information scores...")
        
        # Clean data
        X_clean = X_train.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Make non-negative for MI calculation
        for col in X_clean.columns:
            min_val = X_clean[col].min()
            if min_val < 0:
                X_clean[col] = X_clean[col] - min_val
        
        try:
            mi_scores = mutual_info_classif(
                X_clean,
                y_train,
                discrete_features=False,
                random_state=42,
                n_neighbors=5
            )
        except Exception as e:
            logger.warning(f"   MI calculation failed: {e}")
            mi_scores = X_clean.var().values
            logger.info("   Using variance as fallback")
        
        # Create scores dictionary
        mi_dict = dict(zip(X_train.columns, mi_scores))
        
        # Sort and select top features
        mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("   Top 10 features by mutual information:")
        for feat, score in mi_sorted[:10]:
            logger.info(f"      {feat}: {score:.4f}")
        
        top_features = [f[0] for f in mi_sorted[:self.selection_config.n_features]]
        
        self.feature_scores['mutual_information'] = mi_dict
        self.results.methods_used.append('mutual_information')
        
        return mi_dict, top_features
    
    @timer_decorator
    def tree_based_selection(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[Dict[str, float], List[str]]:
        """Select features using Random Forest importance."""
        logger.info("🌲 Computing tree-based feature importance...")
        
        # Clean data
        X_clean = X_train.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        try:
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_clean, y_train)
            
            importance_dict = dict(zip(X_train.columns, rf.feature_importances_))
            
        except Exception as e:
            logger.warning(f"   Random Forest failed: {e}")
            importance_dict = dict(zip(X_train.columns, X_clean.var().values))
            logger.info("   Using variance as fallback")
        
        # Sort and select
        importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("   Top 10 features by tree importance:")
        for feat, score in importance_sorted[:10]:
            logger.info(f"      {feat}: {score:.4f}")
        
        top_features = [f[0] for f in importance_sorted[:self.selection_config.n_features]]
        
        self.feature_scores['tree_based'] = importance_dict
        self.results.methods_used.append('tree_based')
        
        return importance_dict, top_features
    
    @timer_decorator
    def rfe_selection(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_features: int = 30
    ) -> Tuple[Dict[str, int], List[str]]:
        """Recursive Feature Elimination."""
        logger.info(f"🔄 Running RFE (target: {n_features} features)...")
        
        # Clean data
        X_clean = X_train.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Limit features for RFE (computational efficiency)
        if len(X_clean.columns) > 100:
            # Pre-select using variance
            variances = X_clean.var().nlargest(100)
            X_clean = X_clean[variances.index]
            logger.info(f"   Pre-selected top 100 features by variance")
        
        try:
            # Use lightweight estimator
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=min(n_features, len(X_clean.columns)),
                step=5,
                verbose=0
            )
            
            rfe.fit(X_clean, y_train)
            
            # Get rankings
            ranking_dict = dict(zip(X_clean.columns, rfe.ranking_))
            selected = X_clean.columns[rfe.support_].tolist()
            
        except Exception as e:
            logger.warning(f"   RFE failed: {e}")
            # Fallback to variance-based selection
            variances = X_clean.var().nlargest(n_features)
            ranking_dict = {col: i+1 for i, col in enumerate(variances.index)}
            selected = variances.index.tolist()
        
        logger.info(f"   RFE selected {len(selected)} features")
        
        self.feature_scores['rfe'] = ranking_dict
        self.results.methods_used.append('rfe')
        
        return ranking_dict, selected
    
    @timer_decorator
    def gradient_boosting_selection(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[Dict[str, float], List[str]]:
        """Select features using Gradient Boosting importance."""
        logger.info("🚀 Computing Gradient Boosting importance...")
        
        # Clean data
        X_clean = X_train.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Sample for speed if large dataset
        if len(X_clean) > 50000:
            sample_idx = np.random.choice(len(X_clean), 50000, replace=False)
            X_sample = X_clean.iloc[sample_idx]
            y_sample = y_train.iloc[sample_idx]
        else:
            X_sample = X_clean
            y_sample = y_train
        
        try:
            gb = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                subsample=0.8
            )
            gb.fit(X_sample, y_sample)
            
            importance_dict = dict(zip(X_train.columns, gb.feature_importances_))
            
        except Exception as e:
            logger.warning(f"   Gradient Boosting failed: {e}")
            importance_dict = dict(zip(X_train.columns, X_clean.var().values))
        
        # Sort and select
        importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in importance_sorted[:self.selection_config.n_features]]
        
        logger.info(f"   Top 5 GB features: {top_features[:5]}")
        
        self.feature_scores['gradient_boosting'] = importance_dict
        self.results.methods_used.append('gradient_boosting')
        
        return importance_dict, top_features
    
    # =========================================================================
    # ENSEMBLE SELECTION
    # =========================================================================
    
    def combine_selections(
        self, 
        selection_results: Dict[str, List[str]]
    ) -> List[str]:
        """Combine results from multiple selection methods using voting."""
        logger.info("🗳️ Combining selection results (voting)...")
        
        # Count votes for each feature
        feature_votes = {}
        for method, features in selection_results.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Sort by votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("   Top 20 features by votes:")
        for feat, votes in sorted_features[:20]:
            logger.info(f"      {feat}: {votes} votes")
        
        # Select features with minimum votes
        min_votes = self.selection_config.min_votes
        final_features = [f for f, v in sorted_features if v >= min_votes]
        
        # Ensure we have enough features
        if len(final_features) < self.selection_config.n_features:
            logger.info(f"   Expanding selection (min_votes threshold too strict)")
            final_features = [f[0] for f in sorted_features[:self.selection_config.n_features]]
        elif len(final_features) > self.selection_config.n_features:
            final_features = final_features[:self.selection_config.n_features]
        
        logger.info(f"   Final features selected: {len(final_features)}")
        
        return final_features
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def _plot_correlation_matrix(self, X: pd.DataFrame) -> None:
        """Plot correlation matrix heatmap."""
        try:
            plt.figure(figsize=(14, 12))
            corr = X.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            sns.heatmap(
                corr,
                mask=mask,
                cmap='RdBu_r',
                center=0,
                annot=False,
                square=True,
                linewidths=0.5
            )
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = self.plots_dir / 'feature_correlation.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ Correlation plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"   Could not save correlation plot: {e}")
    
    def plot_feature_importance(
        self, 
        importance_dict: Dict[str, float], 
        method_name: str,
        top_n: int = 20
    ) -> None:
        """Plot feature importance bar chart."""
        try:
            sorted_features = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            features = [f[0][:30] for f in sorted_features]  # Truncate names
            scores = [f[1] for f in sorted_features]
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
            
            plt.barh(range(len(features)), scores, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top {top_n} Features - {method_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            filename = f"importance_{method_name.lower().replace(' ', '_')}.png"
            save_path = self.plots_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ Importance plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"   Could not save importance plot: {e}")
    
    def plot_feature_votes(self, selection_results: Dict[str, List[str]]) -> None:
        """Plot feature selection voting results."""
        try:
            # Count votes
            feature_votes = {}
            for method, features in selection_results.items():
                for feature in features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
            sorted_features = sorted(
                feature_votes.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:25]
            
            features = [f[0][:25] for f in sorted_features]
            votes = [f[1] for f in sorted_features]
            
            plt.figure(figsize=(12, 8))
            colors = ['green' if v >= self.selection_config.min_votes else 'lightcoral' 
                     for v in votes]
            
            plt.barh(range(len(features)), votes, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Number of Votes', fontsize=12)
            plt.title('Feature Selection Voting Results', fontsize=14, fontweight='bold')
            plt.axvline(x=self.selection_config.min_votes, color='red', 
                       linestyle='--', label=f'Min votes threshold ({self.selection_config.min_votes})')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            save_path = self.plots_dir / 'feature_selection_votes.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   ✅ Voting plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"   Could not save voting plot: {e}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    def save_selected_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        selected_features: List[str]
    ) -> None:
        """Save final selected features."""
        logger.info("💾 Saving selected features...")
        
        # Select features
        X_train_final = X_train[selected_features].reset_index(drop=True)
        X_test_final = X_test[selected_features].reset_index(drop=True)
        
        # Decode target labels back to original
        y_train_decoded = pd.Series(
            self.label_encoder.inverse_transform(y_train),
            name='target'
        )
        y_test_decoded = pd.Series(
            self.label_encoder.inverse_transform(y_test),
            name='target'
        )
        
        # Combine
        train_final = pd.concat([X_train_final, y_train_decoded], axis=1)
        test_final = pd.concat([X_test_final, y_test_decoded], axis=1)
        
        # Save paths
        train_path = Path(self.config['data']['output_final_train'])
        test_path = Path(self.config['data']['output_final_test'])
        
        create_directory(train_path.parent)
        
        train_final.to_csv(train_path, index=False)
        test_final.to_csv(test_path, index=False)
        
        logger.info(f"   ✅ Train saved: {train_path} ({train_final.shape})")
        logger.info(f"   ✅ Test saved: {test_path} ({test_final.shape})")
        
        # Update results
        self.results.final_features = len(selected_features)
        self.results.selected_features = selected_features
        self.results.features_removed = self.results.initial_features - len(selected_features)
        
        # Save feature list JSON
        feature_list = {
            'n_features': len(selected_features),
            'features': selected_features,
            'initial_features': self.results.initial_features,
            'features_removed': self.results.features_removed,
            'selection_methods': self.results.methods_used,
            'feature_scores': {
                method: {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                        for k, v in scores.items()}
                for method, scores in self.feature_scores.items()
            },
            'target_classes': self.label_encoder.classes_.tolist(),
            'timestamp': self.results.timestamp
        }
        
        feature_list_path = Path(self.config['data'].get(
            'feature_info_path',
            'src/features/feature_list.json'
        ))
        save_json(feature_list, feature_list_path)
        logger.info(f"   ✅ Feature list saved: {feature_list_path}")
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    @timer_decorator
    def run_selection_pipeline(
        self
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Run complete feature selection pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) with selected features
        """
        logger.info("=" * 70)
        logger.info("🎯 FEATURE SELECTION PIPELINE - STARTING")
        logger.info("=" * 70)
        
        try:
            # Step 1: Load data
            result = self.load_engineered_data()
            if result is None:
                return None
            X_train, X_test, y_train, y_test, target_col = result
            
            logger.info(f"   Starting features: {len(X_train.columns)}")
            
            # Step 2: Remove low variance
            X_train, X_test = self.remove_low_variance(X_train, X_test)
            logger.info(f"   After variance filter: {len(X_train.columns)}")
            
            # Step 3: Remove highly correlated
            X_train, X_test = self.remove_correlated(X_train, X_test)
            logger.info(f"   After correlation filter: {len(X_train.columns)}")
            
            # Step 4: Multiple selection methods
            selection_results = {}
            
            # Mutual Information
            mi_scores, mi_features = self.mutual_information_selection(X_train, y_train)
            selection_results['mutual_info'] = mi_features
            self.plot_feature_importance(mi_scores, "Mutual Information")
            
            # Tree-based
            tree_scores, tree_features = self.tree_based_selection(X_train, y_train)
            selection_results['tree_based'] = tree_features
            self.plot_feature_importance(tree_scores, "Tree Based")
            
            # Gradient Boosting
            gb_scores, gb_features = self.gradient_boosting_selection(X_train, y_train)
            selection_results['gradient_boosting'] = gb_features
            self.plot_feature_importance(gb_scores, "Gradient Boosting")
            
            # RFE
            rfe_config = self.config.get('feature_selection', {}).get('rfe', {})
            if rfe_config.get('enabled', True):
                rfe_rankings, rfe_features = self.rfe_selection(
                    X_train, y_train,
                    n_features=rfe_config.get('n_features_to_select', 30)
                )
                selection_results['rfe'] = rfe_features
            
            # Step 5: Combine selections using voting
            self.plot_feature_votes(selection_results)
            final_features = self.combine_selections(selection_results)
            
            # Step 6: Save results
            self.save_selected_features(
                X_train, X_test, y_train, y_test, final_features
            )
            
            # Summary
            logger.info("=" * 70)
            logger.info("✅ FEATURE SELECTION COMPLETED")
            logger.info("=" * 70)
            logger.info(f"   Initial features: {self.results.initial_features}")
            logger.info(f"   Final features: {self.results.final_features}")
            logger.info(f"   Features removed: {self.results.features_removed}")
            logger.info(f"   Methods used: {', '.join(self.results.methods_used)}")
            logger.info("=" * 70)
            
            return (
                X_train[final_features], 
                X_test[final_features], 
                y_train, 
                y_test
            )
            
        except Exception as e:
            logger.error(f"❌ Selection pipeline failed: {e}")
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
    
    # Run selection pipeline
    selector = FeatureSelector("src/config/config.yaml")
    result = selector.run_selection_pipeline()
    
    if result:
        X_train, X_test, y_train, y_test = result
        print("\n" + "=" * 50)
        print("✅ Feature Selection Completed Successfully!")
        print("=" * 50)
        print(f"\n📊 Results:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape: {X_test.shape}")
        print(f"\n📁 Output files:")
        print(f"   - src/data/processed/final_features/X_train_final.csv")
        print(f"   - src/data/processed/final_features/X_test_final.csv")
        print(f"   - src/features/feature_list.json")
        print(f"   - src/notebooks/plots/importance_*.png")
    else:
        print("\n❌ Feature selection failed!")