"""
Feature engineering module for sports analytics pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Base class for feature engineering."""
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw data.
        
        Args:
            data: DataFrame containing raw data
            
        Returns:
            DataFrame with engineered features
        """
        raise NotImplementedError("Subclasses must implement create_features()")


class TeamFeatureEngineer(FeatureEngineer):
    """Feature engineering for team-based sports data."""
    
    def __init__(self, home_team_col: str = "home_team", away_team_col: str = "away_team", date_col: str = "date"):
        """Initialize the team feature engineer.
        
        Args:
            home_team_col: Name of the column containing home team names
            away_team_col: Name of the column containing away team names
            date_col: Name of the column containing game dates
        """
        self.home_team_col = home_team_col
        self.away_team_col = away_team_col
        self.date_col = date_col
    
    def calculate_win_streak(self, data: pd.DataFrame, outcome_col: str = "outcome") -> pd.DataFrame:
        """Calculate win streak for each team.
        
        Args:
            data: DataFrame containing game results
            outcome_col: Name of the column indicating win/loss
            
        Returns:
            DataFrame with win streak feature added
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Ensure the data is sorted by date
        df = df.sort_values(by=self.date_col)
        
        # Create team records to track wins and streaks
        team_records = {}
        
        # Process each game to calculate streaks
        for idx, row in df.iterrows():
            home_team = row[self.home_team_col]
            away_team = row[self.away_team_col]
            outcome = row[outcome_col]
            
            # Initialize records if first time seeing a team
            if home_team not in team_records:
                team_records[home_team] = {'win_streak': 0, 'games': []}
            if away_team not in team_records:
                team_records[away_team] = {'win_streak': 0, 'games': []}
                
            # Update streaks based on outcome
            if outcome == 'home_win':
                # Home team won
                team_records[home_team]['win_streak'] = max(0, team_records[home_team]['win_streak']) + 1
                team_records[away_team]['win_streak'] = min(0, team_records[away_team]['win_streak']) - 1
            else:
                # Away team won
                team_records[home_team]['win_streak'] = min(0, team_records[home_team]['win_streak']) - 1
                team_records[away_team]['win_streak'] = max(0, team_records[away_team]['win_streak']) + 1
            
            # Add streak to the dataframe
            df.at[idx, 'home_win_streak'] = team_records[home_team]['win_streak']
            df.at[idx, 'away_win_streak'] = team_records[away_team]['win_streak']
            
        logger.info("Calculated win streak feature")
        
        return df
    
    def calculate_moving_averages(self, data: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Calculate moving averages for basketball metrics.
        
        Args:
            data: DataFrame containing the metrics
            windows: List of window sizes for the moving averages
            
        Returns:
            DataFrame with moving average features added
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Ensure the data is sorted by date
        df = df.sort_values(by=self.date_col)
        
        # Create a dictionary to store team-specific statistics
        team_stats = {}
        
        # Process each game to build team statistics
        for idx, row in df.iterrows():
            # Get the teams
            home_team = row[self.home_team_col]
            away_team = row[self.away_team_col]
            game_date = row[self.date_col]
            
            # Initialize team stats dictionaries if needed
            if home_team not in team_stats:
                team_stats[home_team] = {'games': []}
            if away_team not in team_stats:
                team_stats[away_team] = {'games': []}
            
            # Add this game's data to each team's history
            # Home team stats
            team_stats[home_team]['games'].append({
                'date': game_date,
                'points_scored': row['home_score'],
                'points_allowed': row['away_score'],
                'is_home': True,
                'outcome': row['outcome'] == 'home_win'
            })
            
            # Away team stats
            team_stats[away_team]['games'].append({
                'date': game_date,
                'points_scored': row['away_score'],
                'points_allowed': row['home_score'],
                'is_home': False,
                'outcome': row['outcome'] == 'away_win'
            })
            
            # Sort games by date for each team
            team_stats[home_team]['games'] = sorted(team_stats[home_team]['games'], key=lambda x: x['date'])
            team_stats[away_team]['games'] = sorted(team_stats[away_team]['games'], key=lambda x: x['date'])
            
            # Calculate moving averages for home team
            home_games = team_stats[home_team]['games']
            for window in windows:
                # Only calculate if we have enough games
                if len(home_games) > 0:
                    # Calculate moving averages for recent games
                    recent_games = home_games[-min(window, len(home_games)):]
                    pts_scored_avg = sum(g['points_scored'] for g in recent_games) / len(recent_games)
                    pts_allowed_avg = sum(g['points_allowed'] for g in recent_games) / len(recent_games)
                    df.at[idx, f'home_pts_scored_ma{window}'] = pts_scored_avg
                    df.at[idx, f'home_pts_allowed_ma{window}'] = pts_allowed_avg
                    
                    # Calculate win percentage in the window
                    wins = sum(1 for g in recent_games if g['outcome'])
                    df.at[idx, f'home_win_pct_ma{window}'] = wins / len(recent_games)
            
            # Calculate moving averages for away team
            away_games = team_stats[away_team]['games']
            for window in windows:
                # Only calculate if we have enough games
                if len(away_games) > 0:
                    # Calculate moving averages for recent games
                    recent_games = away_games[-min(window, len(away_games)):]
                    pts_scored_avg = sum(g['points_scored'] for g in recent_games) / len(recent_games)
                    pts_allowed_avg = sum(g['points_allowed'] for g in recent_games) / len(recent_games)
                    df.at[idx, f'away_pts_scored_ma{window}'] = pts_scored_avg
                    df.at[idx, f'away_pts_allowed_ma{window}'] = pts_allowed_avg
                    
                    # Calculate win percentage in the window
                    wins = sum(1 for g in recent_games if g['outcome'])
                    df.at[idx, f'away_win_pct_ma{window}'] = wins / len(recent_games)
        
        logger.info(f"Calculated moving averages with {len(windows)} window sizes")
        
        return df
    
    def calculate_head_to_head_stats(self, data: pd.DataFrame, outcome_col: str = "outcome") -> pd.DataFrame:
        """Calculate head-to-head statistics between teams.
        
        Args:
            data: DataFrame containing game results
            outcome_col: Name of the column indicating win/loss
            
        Returns:
            DataFrame with head-to-head statistics features added
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Create a unique identifier for each team pairing
        df['team_pairing'] = df.apply(
            lambda row: f"{min(row[self.home_team_col], row[self.away_team_col])}_vs_{max(row[self.home_team_col], row[self.away_team_col])}", 
            axis=1
        )
        
        # Create a dictionary to track head-to-head records
        h2h_records = {}
        
        # Process each game to build head-to-head statistics
        for idx, row in df.iterrows():
            home_team = row[self.home_team_col]
            away_team = row[self.away_team_col]
            game_date = row[self.date_col]
            pairing = row['team_pairing']
            outcome = row[outcome_col]
            
            # Initialize the pairing record if needed
            if pairing not in h2h_records:
                h2h_records[pairing] = {
                    'games': [],
                    'team1': min(home_team, away_team),
                    'team2': max(home_team, away_team)
                }
            
            # Record the game outcome
            h2h_records[pairing]['games'].append({
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_win': outcome == 'home_win'
            })
            
            # Sort games by date
            h2h_records[pairing]['games'] = sorted(h2h_records[pairing]['games'], key=lambda x: x['date'])
            
            # Calculate cumulative head-to-head stats
            team1 = h2h_records[pairing]['team1']
            team2 = h2h_records[pairing]['team2']
            
            # Count all previous meetings
            previous_games = [g for g in h2h_records[pairing]['games'] if g['date'] <= game_date]
            total_games = len(previous_games)
            
            # Count team1 wins
            team1_wins = sum(1 for g in previous_games if 
                             (g['home_team'] == team1 and g['home_win']) or 
                             (g['away_team'] == team1 and not g['home_win']))
            
            if total_games > 0:
                team1_win_rate = team1_wins / total_games
            else:
                team1_win_rate = 0.5  # Default to even odds if no history
                
            # Add the head-to-head win rates to the dataframe
            if home_team == team1:
                df.at[idx, 'home_h2h_win_rate'] = team1_win_rate
                df.at[idx, 'away_h2h_win_rate'] = 1 - team1_win_rate
            else:
                df.at[idx, 'home_h2h_win_rate'] = 1 - team1_win_rate
                df.at[idx, 'away_h2h_win_rate'] = team1_win_rate
        
        logger.info("Calculated head-to-head statistics features")
        
        return df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for team-based sports data.
        
        Args:
            data: DataFrame containing game results
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Apply feature engineering steps
        df = self.calculate_win_streak(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_head_to_head_stats(df)
        
        # Add point differential features
        df['home_point_diff'] = df['home_score'] - df['away_score']
        df['away_point_diff'] = df['away_score'] - df['home_score']
        
        # Add offensive and defensive efficiency metrics if available
        if all(col in df.columns for col in ['home_fg_attempted', 'away_fg_attempted']):
            # Offensive efficiency (points per 100 possessions)
            # Estimate possessions using FG attempts
            df['home_off_efficiency'] = df['home_score'] / df['home_fg_attempted'] * 100
            df['away_off_efficiency'] = df['away_score'] / df['away_fg_attempted'] * 100
            
        # Calculate game importance features (late season games matter more)
        # Get unique teams to determine season length
        unique_teams = set(df[self.home_team_col].unique()) | set(df[self.away_team_col].unique())
        num_teams = len(unique_teams)
        expected_games_per_team = (num_teams - 1) * 2  # Each team plays each other team twice (home and away)
        
        # Sort by date
        df = df.sort_values(by=self.date_col)
        
        # Add game number and season progress features
        game_counter = {team: 0 for team in unique_teams}
        for idx, row in df.iterrows():
            home_team = row[self.home_team_col]
            away_team = row[self.away_team_col]
            
            # Increment game counters
            game_counter[home_team] += 1
            game_counter[away_team] += 1
            
            # Add game number features
            df.at[idx, 'home_game_num'] = game_counter[home_team]
            df.at[idx, 'away_game_num'] = game_counter[away_team]
            
            # Calculate season progress (0-1 scale)
            df.at[idx, 'home_season_progress'] = min(1.0, game_counter[home_team] / expected_games_per_team)
            df.at[idx, 'away_season_progress'] = min(1.0, game_counter[away_team] / expected_games_per_team)
        
        logger.info(f"Created features for {len(df)} games")
        
        return df