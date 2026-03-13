"""
Day 3 - Model Training Pipeline - FIXED
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.helpers import save_json, save_model
from training.models import ModelFactory
from training.cv_trainer import CrossValidationTrainer

logger = get_logger(__name__)


class ModelTrainer:
    """Model Training Pipeline."""
    
    def __init__(self, config_path="src/config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.model_factory = ModelFactory(self.config)
        self.cv_trainer = CrossValidationTrainer(n_splits=5)
        
        # Results storage - using dict, not dataclass
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = ""
        self.feature_names = []
        self.n_classes = 0
        
        # Create directories
        Path("src/models").mkdir(parents=True, exist_ok=True)
        Path("src/evaluation").mkdir(parents=True, exist_ok=True)
        Path("plots").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelTrainer initialized")
    
    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_data(self):
        """Load and validate data."""
        logger.info("Loading data...")
        
        train_path = Path("inputs/X_train_final.csv")
        test_path = Path("inputs/X_test_final.csv")
        
        if not train_path.exists():
            logger.error(f"Train data not found: {train_path}")
            return None
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Find target
        target_col = 'target'
        if target_col not in train_df.columns:
            for col in ['Attack', 'Label', 'label', 'class']:
                if col in train_df.columns:
                    target_col = col
                    break
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col].astype(int)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].astype(int)
        
        self.feature_names = X_train.columns.tolist()
        self.n_classes = len(np.unique(y_train))
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Classes: {self.n_classes}")
        logger.info(f"Target range: {y_train.min()} to {y_train.max()}")
        
        # Validate
        assert y_train.min() == 0, f"Labels should start at 0, got {y_train.min()}"
        assert y_train.max() == self.n_classes - 1, f"Max label should be {self.n_classes - 1}, got {y_train.max()}"
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train, y_train):
        """Train all models with CV."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        models = self.model_factory.get_models(self.n_classes)
        logger.info(f"Models: {list(models.keys())}")
        
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # CV returns a dict now, not a dataclass
            result = self.cv_trainer.train_with_cv(
                pipeline, X_train, y_train, model_name
            )
            
            self.cv_results[model_name] = result
        
        return self.cv_results
    
    def select_best_model(self, X_train, y_train):
        """Select and train best model."""
        logger.info("\n" + "="*60)
        logger.info("SELECTING BEST MODEL")
        logger.info("="*60)
        
        # Find best by F1 - cv_results is dict of dicts
        best_name = max(
            self.cv_results.keys(), 
            key=lambda k: self.cv_results[k]['f1_mean']
        )
        best_result = self.cv_results[best_name]
        
        logger.info(f"Best: {best_name}")
        logger.info(f"CV F1: {best_result['f1_mean']:.4f} ± {best_result['f1_std']:.4f}")
        
        # Get fresh model
        models = self.model_factory.get_models(self.n_classes)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[best_name])
        ])
        
        logger.info("Training on full data...")
        pipeline.fit(X_train, y_train)
        
        self.best_model = pipeline
        self.best_model_name = best_name
        
        return best_name, pipeline
    
    def evaluate(self, model, X_test, y_test, model_name):
        """Evaluate model on test set."""
        logger.info("\n" + "="*60)
        logger.info(f"EVALUATING {model_name}")
        logger.info("="*60)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC
        try:
            y_proba = model.predict_proba(X_test)
            if self.n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                y_test_bin = label_binarize(y_test, classes=range(self.n_classes))
                metrics['roc_auc'] = roc_auc_score(
                    y_test_bin, y_proba, 
                    average='weighted', multi_class='ovr'
                )
        except Exception as e:
            logger.warning(f"ROC-AUC failed: {e}")
            metrics['roc_auc'] = 0.0
        
        logger.info("Test Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, cm
    
    def save_results(self, metrics, cm):
        """Save all results."""
        logger.info("\nSaving results...")
        
        # Save model
        save_model(self.best_model, "src/models/best_model.pkl")
        logger.info("  Saved: src/models/best_model.pkl")
        
        # Save metrics
        results = {
            'best_model': self.best_model_name,
            'cv_results': self.cv_results,
            'test_metrics': metrics,
            'n_classes': self.n_classes,
            'timestamp': datetime.now().isoformat()
        }
        save_json(results, "src/evaluation/metrics.json")
        logger.info("  Saved: src/evaluation/metrics.json")
        
        # Create plots
        self._create_plots(cm)
        
        # Create report
        self._create_report(metrics)
    
    def _create_plots(self, cm):
        """Create visualization plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Confusion matrix
        plt.figure(figsize=(12, 10))
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        sns.heatmap(cm_norm, annot=False, cmap='Blues', fmt='.2f')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f"cm_{self.best_model_name.lower().replace(' ', '_')}.png"
        plt.savefig(f'plots/{filename}', dpi=150)
        plt.close()
        logger.info(f"  Saved: plots/{filename}")
        
        # Model comparison
        if self.cv_results:
            plt.figure(figsize=(12, 6))
            
            names = list(self.cv_results.keys())
            f1_means = [self.cv_results[n]['f1_mean'] for n in names]
            f1_stds = [self.cv_results[n]['f1_std'] for n in names]
            
            x = np.arange(len(names))
            bars = plt.bar(x, f1_means, yerr=f1_stds, capsize=5, 
                          color='steelblue', alpha=0.8)
            
            plt.ylabel('F1 Score', fontsize=12)
            plt.title('Model Comparison (5-Fold CV)', fontsize=14)
            plt.xticks(x, names, rotation=45, ha='right')
            plt.ylim(0, max(f1_means) * 1.2)
            
            for bar, score in zip(bars, f1_means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('plots/model_comparison.png', dpi=150)
            plt.close()
            logger.info("  Saved: plots/model_comparison.png")
    
    def _create_report(self, metrics):
        """Create MODEL-COMPARISON.md report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        lines = [
            "# Model Comparison Report",
            "## Day 3 - ML Engineering Week",
            "",
            f"**Generated:** {timestamp}",
            "",
            "---",
            "",
            "## Cross-Validation Results (5-Fold)",
            "",
            "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Time |",
            "|-------|----------|-----------|--------|----------|---------|------|"
        ]
        
        # Sort by F1
        sorted_results = sorted(
            self.cv_results.items(), 
            key=lambda x: x[1]['f1_mean'], 
            reverse=True
        )
        
        for name, res in sorted_results:
            lines.append(
                f"| {name} | "
                f"{res['accuracy_mean']:.4f} | "
                f"{res['precision_mean']:.4f} | "
                f"{res['recall_mean']:.4f} | "
                f"{res['f1_mean']:.4f} | "
                f"{res['roc_auc_mean']:.4f} | "
                f"{res['training_time']:.1f}s |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            f"## Best Model: {self.best_model_name}",
            "",
            "### Test Set Metrics",
            ""
        ])
        
        for k, v in metrics.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: {v:.4f}")
        
        lines.extend([
            "",
            "---",
            "",
            "## Output Files",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `src/models/best_model.pkl` | Trained model |",
            "| `src/evaluation/metrics.json` | All metrics |",
            "| `plots/model_comparison.png` | Comparison chart |",
            f"| `plots/cm_{self.best_model_name.lower().replace(' ', '_')}.png` | Confusion matrix |",
            "",
            "---",
            "",
            "## Day 3 Complete!"
        ])
        
        with open("reports/MODEL-COMPARISON.md", 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info("  Saved: reports/MODEL-COMPARISON.md")
    
    def run_pipeline(self):
        """Run complete pipeline."""
        logger.info("\n" + "="*60)
        logger.info("DAY 3 - MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Load
            result = self.load_data()
            if result is None:
                return None
            X_train, X_test, y_train, y_test = result
            
            # Train
            self.train_all_models(X_train, y_train)
            
            # Select best
            best_name, best_model = self.select_best_model(X_train, y_train)
            
            # Evaluate
            metrics, cm = self.evaluate(best_model, X_test, y_test, best_name)
            
            # Save
            self.save_results(metrics, cm)
            
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED!")
            logger.info("="*60)
            logger.info(f"Best Model: {best_name}")
            logger.info(f"Test F1: {metrics['f1_score']:.4f}")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    trainer = ModelTrainer()
    trainer.run_pipeline()