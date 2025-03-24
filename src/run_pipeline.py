"""
Convenience script to run the entire sports analytics pipeline.
"""

import os
import argparse
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the sports analytics pipeline")
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Generate and use sample data"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Input data file (CSV) to use instead of generating sample data"
    )
    parser.add_argument(
        "--num-teams",
        type=int,
        default=10,
        help="Number of teams for sample data generation"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=200,
        help="Number of games for sample data generation"
    )
    return parser.parse_args()


def generate_sample_data(num_teams, num_games):
    """Generate sample data for the pipeline."""
    logger.info("Generating sample data...")
    subprocess.run([
        "python", 
        "src/data/generate_sample_data.py",
        f"--num-teams={num_teams}",
        f"--num-games={num_games}"
    ])
    
    # Get the generated data file
    data_dir = "data/raw"
    data_files = [f for f in os.listdir(data_dir) if f.startswith("basketball_games_") and f.endswith(".csv")]
    if not data_files:
        logger.error("Failed to generate sample data.")
        return None
    
    # Return the path to the generated data file
    return os.path.join(data_dir, max(data_files))


def run_pipeline(data_file):
    """Run the sports analytics pipeline."""
    logger.info(f"Running pipeline with data file: {data_file}")
    subprocess.run([
        "python", 
        "src/pipeline.py",
        f"--mode=full",
        f"--data-source=csv",
        f"--input-file={data_file}"
    ])


def run_dashboard():
    """Run the Streamlit dashboard."""
    logger.info("Starting the dashboard...")
    subprocess.run(["streamlit", "run", "src/visualization/dashboard.py"])


def main():
    """Run the entire pipeline."""
    args = parse_args()
    
    # Step 1: Get the data
    if args.use_sample:
        data_file = generate_sample_data(args.num_teams, args.num_games)
        if data_file is None:
            return
    elif args.data_file:
        if not os.path.exists(args.data_file):
            logger.error(f"Data file not found: {args.data_file}")
            return
        data_file = args.data_file
    else:
        logger.error("No data source specified. Use --use-sample or --data-file.")
        return
    
    # Step 2: Run the pipeline
    run_pipeline(data_file)
    
    # Step 3: Start the dashboard
    run_dashboard()


if __name__ == "__main__":
    main()
