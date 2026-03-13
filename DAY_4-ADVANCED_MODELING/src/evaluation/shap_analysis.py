"""
Day 4: SHAP Analysis Module
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP-based model explainability"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.model = None
        self.X_test = None
        self.y_test = None
        self.shap_values = None
        
        self.output_dir = self.project_root / "outputs"
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SHAPAnalyzer initialized")
        logger.info(f"  SHAP available: {SHAP_AVAILABLE}")
    
    def load_model_and_data(self):
        """Load model and test data"""
        try:
            # Load model
            models_dir = self.output_dir / "models"
            model_files = list(models_dir.glob("best_*.joblib"))
            
            if not model_files:
                logger.error("No trained models found. Run tuning first.")
                return False
            
            model_path = model_files[0]
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model: {model_path.name}")
            
            # Load test data
            input_dir = self.project_root / "inputs"
            self.X_test = pd.read_csv(input_dir / "X_test_final.csv")
            
            y_test_df = pd.read_csv(input_dir / "y_test.csv")
            if 'target' in y_test_df.columns:
                self.y_test = y_test_df['target']
            else:
                self.y_test = y_test_df.iloc[:, 0]
            
            logger.info(f"Loaded test data: {self.X_test.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model/data: {e}")
            return False
    
    def compute_shap_values(self, max_samples=100):
        """Compute SHAP values"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return False
        
        logger.info(f"Computing SHAP values (max {max_samples} samples)...")
        
        # Sample data for faster computation
        n_samples = min(max_samples, len(self.X_test))
        X_sample = self.X_test.iloc[:n_samples]
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        self.X_sample = X_sample
        logger.info("SHAP values computed successfully")
        
        return True
    
    def plot_summary(self):
        """Create SHAP summary plot"""
        if not SHAP_AVAILABLE or self.shap_values is None:
            return
        
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_sample, 
            show=False
        )
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {self.plots_dir / 'shap_summary.png'}")
    
    def plot_feature_importance(self):
        """Create feature importance bar plot"""
        if not SHAP_AVAILABLE or self.shap_values is None:
            return
        
        logger.info("Creating feature importance plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_sample, 
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {self.plots_dir / 'shap_importance.png'}")
    
    def save_feature_importance(self):
        """Save feature importance to CSV"""
        if self.shap_values is None:
            return
        
        importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.X_sample.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(self.output_dir / "shap_feature_importance.csv", index=False)
        logger.info(f"Saved: {self.output_dir / 'shap_feature_importance.csv'}")
        
        # Print top features
        logger.info("\nTop 10 Important Features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def run_analysis(self):
        """Run complete SHAP analysis"""
        logger.info("\n" + "="*60)
        logger.info("SHAP ANALYSIS")
        logger.info("="*60)
        
        logger.info("Loading model and data...")
        if not self.load_model_and_data():
            return False
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return False
        
        self.compute_shap_values(max_samples=100)
        self.plot_summary()
        self.plot_feature_importance()
        self.save_feature_importance()
        
        logger.info("\n" + "="*60)
        logger.info("SHAP ANALYSIS COMPLETE!")
        logger.info("="*60)
        
        return True


if __name__ == "__main__":
    analyzer = SHAPAnalyzer()
    analyzer.run_analysis()