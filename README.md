# Sports Analytics Pipeline

An end-to-end sports analytics pipeline that tracks team/player performance and predicts game outcomes. This project demonstrates data engineering and machine learning practices by collecting, processing, and analyzing sports data to generate predictions for upcoming games.

## Features

- Data collection and ingestion from various sources
- Data cleaning and preprocessing
- Exploratory data analysis and visualization
- Feature engineering for sports analytics
- Machine learning models for game outcome prediction
- Interactive dashboard for visualizing team performance and predictions

## Project Structure

```
├── data/                   # Data directory
│   ├── raw/                # Raw data
│   └── processed/          # Processed data
├── models/                 # Saved ML models
├── notebooks/              # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb  # EDA notebook
├── src/                    # Source code
│   ├── data/               # Data processing scripts
│   │   ├── data_collection.py      # Data collection module
│   │   ├── data_processing.py      # Data processing module
│   │   └── generate_sample_data.py # Sample data generator
│   ├── features/           # Feature engineering
│   │   └── feature_engineering.py  # Feature engineering module
│   ├── models/             # ML models
│   │   └── model_training.py       # Model training module
│   ├── visualization/      # Visualization code
│   │   ├── dashboard.py            # Streamlit dashboard
│   │   └── visualize.py            # Visualization module
│   ├── pipeline.py         # Main pipeline script
│   └── run_pipeline.py     # Convenience script to run the pipeline
├── visualizations/         # Generated visualizations
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning models
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Interactive dashboard
- **Jupyter Notebooks**: Exploratory data analysis

## Setup

```bash
# Clone the repository
git clone https://github.com/chrisesposito92/sports-analytics-pipeline.git
cd sports-analytics-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

The pipeline can be run using the convenience script:

```bash
# Generate sample data and run the full pipeline
python src/run_pipeline.py --use-sample

# Run with specific number of teams and games
python src/run_pipeline.py --use-sample --num-teams 12 --num-games 300

# Run with your own data file
python src/run_pipeline.py --data-file path/to/your/data.csv
```

### Running Individual Components

The pipeline has several components that can be run independently:

```bash
# Generate sample data
python src/data/generate_sample_data.py --num-teams 10 --num-games 200

# Run the pipeline with specific mode
python src/pipeline.py --mode collect --data-source csv --input-file data/raw/your_data.csv
python src/pipeline.py --mode process
python src/pipeline.py --mode train

# Run the dashboard
streamlit run src/visualization/dashboard.py
```

## Data Sources

This project can work with various sports data sources:

1. **Sample Data**: Generated basketball game data with realistic statistics
2. **CSV Files**: Import your own sports data in CSV format
3. **API Integration**: Connect to sports data APIs (requires additional configuration)

## Machine Learning Approach

The machine learning approach involves:

1. **Feature Engineering**: Creating features like team win rates, scoring patterns, and head-to-head statistics
2. **Model Selection**: Using classification models to predict game outcomes (win/loss)
3. **Evaluation**: Assessing model performance with metrics like accuracy, precision, and recall
4. **Prediction**: Generating predictions for upcoming games with win probabilities

## Dashboard

The interactive dashboard provides:

1. **Team Performance**: Visualize team statistics and performance metrics
2. **Head-to-Head Analysis**: Compare teams and analyze matchup history
3. **Predictions**: Get win probabilities and predicted scores for upcoming games

To launch the dashboard:

```bash
streamlit run src/visualization/dashboard.py
```

## Future Improvements

- Add player-level statistics and analysis
- Implement more advanced prediction models
- Add time series analysis for team performance trends
- Integrate with live data sources for real-time updates
- Deploy the dashboard to a web server

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
