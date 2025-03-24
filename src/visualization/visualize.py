"""
Visualization module for sports analytics pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SportsVisualizer:
    """Visualizer for sports analytics."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the style for the visualizations
        sns.set(style="whitegrid")
        
        # Increase the font size for better readability
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })
    
    def save_plot(self, filename: str) -> None:
        """Save the current plot to a file.
        
        Args:
            filename: Name of the output file (without extension)
        """
        import os
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {filepath}")
    
    def plot_team_performance(self, data: pd.DataFrame, team_col: str, metric_col: str, date_col: str) -> None:
        """Plot team performance over time.
        
        Args:
            data: DataFrame containing team performance data
            team_col: Name of the column containing team names/IDs
            metric_col: Name of the column containing the performance metric
            date_col: Name of the column containing dates
        """
        # Create a figure with a larger size
        plt.figure(figsize=(12, 8))
        
        # Get the unique teams
        teams = data[team_col].unique()
        
        # Plot the performance of each team
        for team in teams:
            team_data = data[data[team_col] == team].sort_values(by=date_col)
            plt.plot(team_data[date_col], team_data[metric_col], label=team)
        
        # Add labels and title
        plt.xlabel("Date")
        plt.ylabel(metric_col.replace("_", " ").title())
        plt.title(f"Team Performance: {metric_col.replace('_', ' ').title()} Over Time")
        
        # Add a legend
        plt.legend(title="Team", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the plot
        self.save_plot(f"team_performance_{metric_col}")
    
    def plot_win_loss_distribution(self, data: pd.DataFrame, team_col: str, outcome_col: str) -> None:
        """Plot the distribution of wins and losses for each team.
        
        Args:
            data: DataFrame containing game results
            team_col: Name of the column containing team names/IDs
            outcome_col: Name of the column indicating win/loss
        """
        # Create a figure with a larger size
        plt.figure(figsize=(12, 8))
        
        # Count the wins and losses for each team
        win_loss_counts = data.groupby([team_col, outcome_col]).size().unstack(fill_value=0)
        
        # Sort the teams by the number of wins in descending order
        if 'win' in win_loss_counts.columns:
            win_loss_counts = win_loss_counts.sort_values(by='win', ascending=False)
        
        # Create a stacked bar chart
        win_loss_counts.plot(kind='bar', stacked=True, ax=plt.gca())
        
        # Add labels and title
        plt.xlabel("Team")
        plt.ylabel("Number of Games")
        plt.title("Win-Loss Distribution by Team")
        
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the plot
        self.save_plot("win_loss_distribution")
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray) -> None:
        """Plot the importance of features in a prediction model.
        
        Args:
            feature_names: Names of the features
            importances: Importance scores for each feature
        """
        # Create a DataFrame with the feature names and importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort the features by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        # Create a figure with a larger size
        plt.figure(figsize=(12, 10))
        
        # Create a horizontal bar chart
        sns.barplot(data=feature_importance_df, y='Feature', x='Importance')
        
        # Add labels and title
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance in Prediction Model")
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the plot
        self.save_plot("feature_importance")
    
    def plot_prediction_confidence(self, actual_outcomes: np.ndarray, predicted_probs: np.ndarray, threshold: float = 0.5) -> None:
        """Plot the confidence of predictions.
        
        Args:
            actual_outcomes: Array of actual outcomes (1 for win, 0 for loss)
            predicted_probs: Array of predicted probabilities for the positive class
            threshold: Probability threshold for classification
        """
        # Create a figure with a larger size
        plt.figure(figsize=(12, 8))
        
        # Create bins for the predicted probabilities
        bins = np.linspace(0, 1, 11)
        
        # Group the predictions into bins
        binned_probs = np.digitize(predicted_probs, bins) - 1
        
        # Calculate the actual win rate for each bin
        win_rates = []
        bin_counts = []
        for i in range(len(bins) - 1):
            bin_mask = binned_probs == i
            if np.sum(bin_mask) > 0:
                win_rate = np.mean(actual_outcomes[bin_mask])
                win_rates.append(win_rate)
                bin_counts.append(np.sum(bin_mask))
            else:
                win_rates.append(0)
                bin_counts.append(0)
        
        # Create a DataFrame for the binned predictions
        binned_df = pd.DataFrame({
            'Predicted Probability': (bins[:-1] + bins[1:]) / 2,
            'Actual Win Rate': win_rates,
            'Count': bin_counts
        })
        
        # Plot the calibration curve
        plt.bar(binned_df['Predicted Probability'], binned_df['Count'], width=0.08, alpha=0.3, label='Prediction Count')
        plt.plot(binned_df['Predicted Probability'], binned_df['Actual Win Rate'], 'ro-', label='Actual Win Rate')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Add labels and title
        plt.xlabel("Predicted Win Probability")
        plt.ylabel("Actual Win Rate / Prediction Count")
        plt.title("Prediction Calibration Plot")
        
        # Add a legend
        plt.legend()
        
        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the plot
        self.save_plot("prediction_calibration")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> None:
        """Plot a confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of the classes
        """
        # Create a figure with a larger size
        plt.figure(figsize=(10, 8))
        
        # Create a heatmap of the confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        
        # Add labels and title
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save the plot
        self.save_plot("confusion_matrix")