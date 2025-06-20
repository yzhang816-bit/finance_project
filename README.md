# Financial Market Analysis Project

## Project Overview
This project analyzes financial market data to understand the impact of market volatility on stock returns across different sectors. The analysis focuses on identifying patterns and developing predictive models that can inform investment strategies in various market conditions.

## Directory Structure
- `/code`: Contains all Python scripts for data collection, cleaning, analysis, and modeling
- `/data`: Raw and processed data files
- `/data/cleaned`: Cleaned and processed datasets ready for analysis
- `/reports`: Generated reports and analysis results
- `/visualizations`: Generated charts and visual outputs
- `/final_report`: Final compiled report in both Markdown and PDF formats
- `/models`: Saved model files and performance metrics

## Installation Requirements
This project requires Python 3.8+ and the following packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
yfinance
pandas-datareader
```

You can install all required packages using:
```
pip install -r requirements.txt
```

## Running the Project

### Step 1: Data Collection
Run the data collection script to gather stock price data, VIX data, and economic indicators:
```
python code/data_collection.py
```
This will download data from Yahoo Finance and save it to the `/data` directory.

### Step 2: Data Cleaning
Process and clean the raw data:
```
python code/data_cleaning.py
```
This will create cleaned datasets in the `/data/cleaned` directory.

### Step 3: Exploratory Analysis
Generate exploratory visualizations and initial analysis:
```
python code/exploratory_analysis.py
```
This will create visualizations in the `/visualizations` directory and save analysis results to the `/reports` directory.

### Step 4: Model Development
Build and evaluate predictive models:
```
python code/model_development.py
```
This will train models for different volatility regimes and sectors, evaluate their performance, and save results to the `/reports` directory.

### Step 5: Compile Final Report
Generate the final comprehensive report:
```
python code/compile_report.py
```
This will create the final report in both Markdown and PDF formats in the `/final_report` directory.

## Key Files
- `data_collection.py`: Gathers financial data from various sources
- `data_cleaning.py`: Processes and cleans the raw data
- `exploratory_analysis.py`: Performs initial data exploration and visualization
- `model_development.py`: Builds and evaluates predictive models
- `compile_report.py`: Generates the final comprehensive report

## Results
The main findings and results can be found in:
- `/final_report/full_report.pdf`: Complete project report with all findings
- `/final_report/executive_summary.md`: Brief summary of key findings
- `/visualizations`: All generated charts and visualizations
- `/reports`: Detailed analysis results and model performance metrics

## Reproducibility
To reproduce the entire project from scratch, run the scripts in the following order:
1. `data_collection.py`
2. `data_cleaning.py`
3. `exploratory_analysis.py`
4. `model_development.py`
5. `compile_report.py`

Each script is designed to run independently if the required input files exist.
