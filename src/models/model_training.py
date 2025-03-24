"""
Model training module for sports analytics pipeline.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Base class for model training."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the model trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save_model(self, model: Any, filename: str) -> None:
        """Save a trained model and scaler to disk.
        
        Args:
            model: Trained model to save
            filename: Name of the output file
        """
        # Create a dictionary containing both the model and the scaler
        model_data = {
            'model': model,
            'scaler': self.scaler if hasattr(self, 'scaler') else None
        }
        
        filepath = os.path.join(self.model_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model and scaler saved to {filepath}")
    
    def load_model(self, filename: str) -> Any:
        """Load a trained model and scaler from disk.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Loaded model
        """
        filepath = os.path.join(self.model_dir, filename)
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if the loaded data is a dictionary with model and scaler
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            # Also load the scaler if available
            if 'scaler' in model_data and model_data['scaler'] is not None:
                self.scaler = model_data['scaler']
        else:
            # For backward compatibility with older models
            model = model_data
            
        logger.info(f"Model loaded from {filepath}")
        return model


class GameOutcomePredictor(ModelTrainer):
    """Model for predicting game outcomes."""
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42, model_dir: str = "models"):
        """Initialize the game outcome predictor.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'logistic_regression')
            random_state: Random seed for reproducibility
            model_dir: Directory to save trained models
        """
        super().__init__(model_dir)
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Preprocessed features
        """
        # Create a copy to avoid SettingWithCopyWarning
        X_processed = X.copy()
        
        # Handle categorical features (if any) using one-hot encoding
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        
        # Log the preprocessing steps
        logger.info(f"Preprocessed features: {X_processed.shape[1]} features")
        if categorical_cols:
            logger.info(f"One-hot encoded {len(categorical_cols)} categorical features")
        
        return X_processed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a game outcome prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels (win/loss)
            
        Returns:
            Trained model
        """
        # Preprocess the features
        X_train_processed = self.preprocess(X_train)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        
        # Create and train the model
        if self.model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            logger.info("Training Random Forest model")
        elif self.model_type == "logistic_regression":
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            logger.info("Training Logistic Regression model")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model.fit(X_train_scaled, y_train)
        logger.info(f"Model trained on {X_train_scaled.shape[0]} samples")
        
        return model
    
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with a trained model.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        # Preprocess the features
        X_processed = self.preprocess(X)
        
        # Scale the features
        X_scaled = self.scaler.transform(X_processed)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        logger.info(f"Made predictions for {X_scaled.shape[0]} samples")
        
        return predictions
    
    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with a trained model.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        # Preprocess the features
        X_processed = self.preprocess(X)
        
        # Scale the features
        X_scaled = self.scaler.transform(X_processed)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X_scaled)
        logger.info(f"Got prediction probabilities for {X_scaled.shape[0]} samples")
        
        return probabilities
    
    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a game outcome prediction model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(model, X_test)
        
        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        
        # Log the evaluation metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Calculate and log the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion matrix:\n{cm}")
        
        return metrics