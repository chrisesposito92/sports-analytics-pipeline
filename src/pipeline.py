"""
Main pipeline script for sports analytics.
"""

import os
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime

# Import modules
from data.data_collection import APIDataCollector, CSVDataCollector
from data.data_processing import SportsDataProcessor
from features.feature_engineering import TeamFeatureEngineer
from models.model_training import GameOutcomePredictor
from visualization.visualize import SportsVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sports Analytics Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "process", "train", "predict", "full"],
        default="full",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["api", "csv"],
        default="csv",
        help="Data source type"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input file for CSV data source"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API URL for API data source"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for API data source"
    )
    return parser.parse_args()


def collect_data(args):
    """Collect data from the specified source."""
    if args.data_source == "api":
        if not args.api_url:
            raise ValueError("API URL is required for API data source")
        
        collector = APIDataCollector(args.api_url, args.api_key)
        data = collector.collect("games")
        collector.save_data(data, f"games_{datetime.now().strftime('%Y%m%d')}.csv")
    
    elif args.data_source == "csv":
        if not args.input_file:
            raise ValueError("Input file is required for CSV data source")
        
        # Use the correct path directly instead of going through the collector's input_dir
        data = pd.read_csv(args.input_file)
        
        # Save a copy to the raw directory with the standardized name format
        output_path = os.path.join("data/raw", f"games_{datetime.now().strftime('%Y%m%d')}.csv")
        data.to_csv(output_path, index=False)
        logger.info(f"Data copied to {output_path}")
    
    return data


def process_data(data):
    """Process the collected data."""
    processor = SportsDataProcessor()
    processed_data = processor.process(data)
    processor.save_data(processed_data, f"processed_games_{datetime.now().strftime('%Y%m%d')}.csv")
    return processed_data


def engineer_features(data):
    """Engineer features from the processed data."""
    engineer = TeamFeatureEngineer()
    featured_data = engineer.create_features(data)
    
    # Save the featured data
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"featured_games_{datetime.now().strftime('%Y%m%d')}.csv")
    featured_data.to_csv(output_path, index=False)
    
    logger.info(f"Featured data saved to {output_path}")
    
    return featured_data


def train_model(data):
    """Train a model on the featured data."""
    # Convert categorical outcome to binary target
    # 'home_win' = 1, 'away_win' = 0
    y = (data['outcome'] == 'home_win').astype(int)
    
    # Select features for the model
    # Drop columns that shouldn't be used as features
    columns_to_drop = [
        'game_id', 'date', 'home_team', 'away_team', 'outcome', 
        'team_pairing', 'home_score', 'away_score'
    ]
    
    # Drop any other non-numeric columns
    non_numeric_cols = [col for col in data.columns 
                        if col not in columns_to_drop and 
                        not pd.api.types.is_numeric_dtype(data[col])]
    
    columns_to_drop += non_numeric_cols
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    
    X = data.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Fill any missing values with 0 (for simplicity)
    X = X.fillna(0)
    
    # Log the features being used
    logger.info(f"Training model with {X.shape[1]} features: {', '.join(X.columns)}")
    
    # Train the model
    predictor = GameOutcomePredictor(model_type="random_forest")
    model = predictor.train(X, y)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    predictor.save_model(model, f"game_outcome_model_{datetime.now().strftime('%Y%m%d')}.pkl")
    
    return model, predictor


def evaluate_model(model, predictor, data):
    """Evaluate the trained model."""
    # Prepare the target
    y = (data['outcome'] == 'home_win').astype(int)
    
    # Prepare features (same as in train_model)
    columns_to_drop = [
        'game_id', 'date', 'home_team', 'away_team', 'outcome', 
        'team_pairing', 'home_score', 'away_score'
    ]
    
    # Drop any other non-numeric columns
    non_numeric_cols = [col for col in data.columns 
                        if col not in columns_to_drop and 
                        not pd.api.types.is_numeric_dtype(data[col])]
    
    columns_to_drop += non_numeric_cols
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    
    X = data.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Fill any missing values with 0 (for simplicity)
    X = X.fillna(0)
    
    # Evaluate the model
    metrics = predictor.evaluate(model, X, y)
    
    # Create visualizations
    os.makedirs("visualizations", exist_ok=True)
    visualizer = SportsVisualizer(output_dir="visualizations")
    
    # Plot feature importance if the model supports it
    if hasattr(model, 'feature_importances_'):
        visualizer.plot_feature_importance(X.columns, model.feature_importances_)
    
    # Get prediction probabilities
    y_proba = predictor.predict_proba(model, X)
    
    # Plot prediction confidence
    visualizer.plot_prediction_confidence(y, y_proba[:, 1])
    
    # Create additional visualizations
    
    # Confusion matrix
    y_pred = predictor.predict(model, X)
    cm = np.array([[np.sum((y == 0) & (y_pred == 0)), np.sum((y == 0) & (y_pred == 1))],
                   [np.sum((y == 1) & (y_pred == 0)), np.sum((y == 1) & (y_pred == 1))]])
    visualizer.plot_confusion_matrix(cm, ["Away Win", "Home Win"])
    
    return metrics


def main():
    """Run the sports analytics pipeline."""
    args = parse_args()
    
    if args.mode in ["collect", "full"]:
        logger.info("Collecting data...")
        data = collect_data(args)
    else:
        # Load the most recent data file
        data_dir = "data/raw"
        data_files = [f for f in os.listdir(data_dir) if f.startswith("games_") and f.endswith(".csv")]
        if not data_files:
            raise ValueError("No data files found. Run the pipeline with --mode collect first.")
        
        latest_file = max(data_files)
        data = pd.read_csv(os.path.join(data_dir, latest_file))
        logger.info(f"Loaded data from {latest_file}")
    
    if args.mode in ["process", "full"]:
        logger.info("Processing data...")
        data = process_data(data)
    
    if args.mode in ["train", "full"]:
        logger.info("Engineering features...")
        data = engineer_features(data)
        
        logger.info("Training model...")
        model, predictor = train_model(data)
        
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, predictor, data)
        
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
