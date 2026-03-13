"""
Day 4: Hyperparameter Tuning Module
"""

import os
import sys
import yaml
import json
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Optional imports
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning with multiple methods"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config = self._load_config()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_models = {}
        self.results = {}
        
        # Create output directories
        self.output_dir = self.project_root / "outputs"
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        
        for dir_path in [self.output_dir, self.models_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("HyperparameterTuner initialized")
        logger.info(f"  Optuna available: {OPTUNA_AVAILABLE}")
        logger.info(f"  XGBoost available: {XGBOOST_AVAILABLE}")
        logger.info(f"  LightGBM available: {LIGHTGBM_AVAILABLE}")
    
    def _load_config(self):
        """Load configuration from YAML"""
        config_path = self.project_root / "src" / "config" / "config.yaml"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return self._default_config()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'tuning': {
                'method': 'grid_search',
                'cv_folds': 5,
                'scoring': 'accuracy',
                'n_iter': 50,
                'random_state': 42
            },
            'models': {
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }
        }
    
    def load_data(self):
        """Load training and test data"""
        try:
            input_dir = self.project_root / "inputs"
            
            # Load X data
            self.X_train = pd.read_csv(input_dir / "X_train_final.csv")
            self.X_test = pd.read_csv(input_dir / "X_test_final.csv")
            
            # Load y data
            y_train_df = pd.read_csv(input_dir / "y_train.csv")
            y_test_df = pd.read_csv(input_dir / "y_test.csv")
            
            # Convert to series (handle different column names)
            if 'target' in y_train_df.columns:
                self.y_train = y_train_df['target']
                self.y_test = y_test_df['target']
            else:
                # Use first column
                self.y_train = y_train_df.iloc[:, 0]
                self.y_test = y_test_df.iloc[:, 0]
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  X_train shape: {self.X_train.shape}")
            logger.info(f"  X_test shape: {self.X_test.shape}")
            logger.info(f"  y_train shape: {self.y_train.shape}")
            logger.info(f"  y_test shape: {self.y_test.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def tune_random_forest(self, method='grid_search'):
        """Tune Random Forest"""
        logger.info("\n--- Tuning Random Forest ---")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        if method == 'grid_search':
            search = GridSearchCV(
                rf, param_grid, 
                cv=5, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                rf, param_grid, 
                n_iter=20, cv=5, 
                scoring='accuracy', 
                n_jobs=-1, verbose=1, 
                random_state=42
            )
        
        search.fit(self.X_train, self.y_train)
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.best_models['random_forest'] = best_model
        self.results['random_forest'] = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_accuracy': accuracy
        }
        
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return best_model
    
    def tune_xgboost(self, method='grid_search'):
        """Tune XGBoost"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping")
            return None
        
        logger.info("\n--- Tuning XGBoost ---")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_clf = xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        if method == 'grid_search':
            search = GridSearchCV(
                xgb_clf, param_grid, 
                cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                xgb_clf, param_grid, 
                n_iter=15, cv=3, 
                scoring='accuracy', 
                n_jobs=-1, verbose=1, 
                random_state=42
            )
        
        search.fit(self.X_train, self.y_train)
        
        best_model = search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.best_models['xgboost'] = best_model
        self.results['xgboost'] = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_accuracy': accuracy
        }
        
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return best_model
    
    def tune_with_optuna(self, n_trials=50):
        """Tune using Optuna (Bayesian optimization)"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using grid search instead")
            return self.tune_random_forest('grid_search')
        
        logger.info("\n--- Tuning with Optuna (Bayesian) ---")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Train best model
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(self.X_train, self.y_train)
        
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.best_models['optuna_rf'] = best_model
        self.results['optuna_rf'] = {
            'best_params': best_params,
            'best_cv_score': study.best_value,
            'test_accuracy': accuracy
        }
        
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return best_model
    
    def save_results(self):
        """Save models and results"""
        logger.info("\n--- Saving Results ---")
        
        # Save best models
        for name, model in self.best_models.items():
            model_path = self.models_dir / f"best_{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved model: {model_path}")
        
        # Save results summary
        results_path = self.output_dir / "tuning_results.json"
        
        # Convert numpy types to Python types
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'best_params': {k: (int(v) if isinstance(v, np.integer) else 
                                   float(v) if isinstance(v, np.floating) else v)
                               for k, v in result['best_params'].items()},
                'best_cv_score': float(result['best_cv_score']),
                'test_accuracy': float(result['test_accuracy'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results: {results_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TUNING SUMMARY")
        logger.info("="*50)
        
        for model_name, result in self.results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  CV Score: {result['best_cv_score']:.4f}")
            logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
    
    def run_pipeline(self, method='grid_search'):
        """Run complete tuning pipeline"""
        logger.info("\n" + "="*60)
        logger.info("DAY 4 - HYPERPARAMETER TUNING PIPELINE")
        logger.info("="*60)
        
        # Load data
        logger.info("Loading data...")
        if not self.load_data():
            return False
        
        # Tune models
        if method == 'bayesian':
            self.tune_with_optuna(n_trials=30)
        else:
            self.tune_random_forest(method)
            self.tune_xgboost(method)
        
        # Save results
        self.save_results()
        
        logger.info("\n" + "="*60)
        logger.info("TUNING COMPLETE!")
        logger.info("="*60)
        
        return True


if __name__ == "__main__":
    tuner = HyperparameterTuner()
    tuner.run_pipeline(method='random_search')