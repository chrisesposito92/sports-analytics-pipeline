"""
Generate sample sports data for demonstration.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_teams(num_teams=10):
    """Generate a list of team names.
    
    Args:
        num_teams: Number of teams to generate
        
    Returns:
        List of team names
    """
    team_names = [
        "Bucks", "Celtics", "76ers", "Raptors", "Nets", 
        "Nuggets", "Trail Blazers", "Thunder", "Jazz", "Rockets",
        "Lakers", "Clippers", "Kings", "Spurs", "Timberwolves",
        "Warriors", "Pelicans", "Mavericks", "Grizzlies", "Suns"
    ]
    
    # Select a subset of teams if needed
    if num_teams <= len(team_names):
        return team_names[:num_teams]
    else:
        # If more teams are requested than available, add numbered teams
        additional_teams = [f"Team {i+1}" for i in range(len(team_names), num_teams)]
        return team_names + additional_teams


def generate_basketball_data(num_teams=10, num_games=200, start_date=None, end_date=None, seed=42):
    """Generate sample basketball game data.
    
    Args:
        num_teams: Number of teams in the league
        num_games: Number of games to generate
        start_date: Start date for the games (default: 1 year ago)
        end_date: End date for the games (default: today)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing the generated data
    """
    np.random.seed(seed)
    
    # Generate team names
    teams = generate_teams(num_teams)
    
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # Generate random dates for the games
    date_range = (end_date - start_date).days
    game_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(num_games)]
    game_dates.sort()  # Sort the dates in ascending order
    
    # Make team abilities (higher is better)
    team_abilities = {team: np.random.normal(loc=75, scale=10) for team in teams}
    
    # Generate game data
    games_data = []
    
    for game_idx, game_date in enumerate(game_dates):
        # Randomly select home and away teams (without repetition)
        home_team, away_team = np.random.choice(teams, size=2, replace=False)
        
        # Calculate base expected scores using team abilities
        home_base_score = team_abilities[home_team]
        away_base_score = team_abilities[away_team]
        
        # Add home court advantage (typically 3-5 points in basketball)
        home_base_score += 4
        
        # Add random noise to create variability
        home_score = int(max(0, np.random.normal(loc=home_base_score, scale=12)))
        away_score = int(max(0, np.random.normal(loc=away_base_score, scale=12)))
        
        # Determine the outcome
        if home_score > away_score:
            outcome = 'home_win'
        else:
            outcome = 'away_win'
        
        # Calculate additional statistics
        # For simplicity, we'll use random distributions based on the score
        
        # Field goals (roughly 40-50% of points come from field goals)
        home_fg_made = int(home_score * 0.38 / 2)  # Divide by 2 because each FG is worth 2 points
        home_fg_attempted = int(home_fg_made / np.random.uniform(0.40, 0.55))  # FG% between 40-55%
        
        away_fg_made = int(away_score * 0.38 / 2)
        away_fg_attempted = int(away_fg_made / np.random.uniform(0.40, 0.55))
        
        # Three pointers (roughly 30-40% of points come from three pointers)
        home_3pt_made = int(home_score * 0.32 / 3)  # Divide by 3 because each 3PT is worth 3 points
        home_3pt_attempted = int(home_3pt_made / np.random.uniform(0.32, 0.42))  # 3PT% between 32-42%
        
        away_3pt_made = int(away_score * 0.32 / 3)
        away_3pt_attempted = int(away_3pt_made / np.random.uniform(0.32, 0.42))
        
        # Free throws (roughly 15-25% of points come from free throws)
        home_ft_made = int(home_score * 0.20)
        home_ft_attempted = int(home_ft_made / np.random.uniform(0.70, 0.85))  # FT% between 70-85%
        
        away_ft_made = int(away_score * 0.20)
        away_ft_attempted = int(away_ft_made / np.random.uniform(0.70, 0.85))
        
        # Rebounds (typically 40-50 per team)
        home_rebounds = int(np.random.normal(loc=44, scale=5))
        away_rebounds = int(np.random.normal(loc=44, scale=5))
        
        # Assists (typically 20-30 per team)
        home_assists = int(np.random.normal(loc=24, scale=4))
        away_assists = int(np.random.normal(loc=24, scale=4))
        
        # Turnovers (typically 12-18 per team)
        home_turnovers = int(np.random.normal(loc=14, scale=3))
        away_turnovers = int(np.random.normal(loc=14, scale=3))
        
        # Add the game data
        games_data.append({
            'game_id': game_idx + 1,
            'date': game_date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'outcome': outcome,
            'home_fg_made': home_fg_made,
            'home_fg_attempted': home_fg_attempted,
            'home_3pt_made': home_3pt_made,
            'home_3pt_attempted': home_3pt_attempted,
            'home_ft_made': home_ft_made,
            'home_ft_attempted': home_ft_attempted,
            'home_rebounds': home_rebounds,
            'home_assists': home_assists,
            'home_turnovers': home_turnovers,
            'away_fg_made': away_fg_made,
            'away_fg_attempted': away_fg_attempted,
            'away_3pt_made': away_3pt_made,
            'away_3pt_attempted': away_3pt_attempted,
            'away_ft_made': away_ft_made,
            'away_ft_attempted': away_ft_attempted,
            'away_rebounds': away_rebounds,
            'away_assists': away_assists,
            'away_turnovers': away_turnovers
        })
    
    # Create a DataFrame
    df = pd.DataFrame(games_data)
    
    logger.info(f"Generated {len(df)} games with {num_teams} teams")
    
    return df


def main():
    """Generate sample data and save it to CSV."""
    parser = argparse.ArgumentParser(description="Generate sample basketball game data")
    parser.add_argument(
        "--num-teams", 
        type=int, 
        default=10, 
        help="Number of teams in the league"
    )
    parser.add_argument(
        "--num-games", 
        type=int, 
        default=200, 
        help="Number of games to generate"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/raw", 
        help="Directory to save the generated data"
    )
    args = parser.parse_args()
    
    # Generate the data
    df = generate_basketball_data(
        num_teams=args.num_teams,
        num_games=args.num_games,
        seed=args.seed
    )
    
    # Save the data to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"basketball_games_{datetime.now().strftime('%Y%m%d')}.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
