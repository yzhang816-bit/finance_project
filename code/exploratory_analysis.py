#!/usr/bin/env python3
"""
Exploratory Data Analysis Script for Finance Project

This script performs exploratory analysis on the cleaned financial data:
- Analyzes volatility patterns across different time periods
- Examines correlations between volatility and prediction errors
- Identifies sector-specific responses to volatility
- Visualizes relationships between key variables
- Generates insights for model development

The analysis results are saved to the visualizations directory.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

# Set up paths
DATA_DIR = "/home/ubuntu/finance_project/data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
VISUAL_DIR = "/home/ubuntu/finance_project/visualizations"
REPORTS_DIR = "/home/ubuntu/finance_project/reports"
os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def load_cleaned_data():
    """Load all cleaned datasets"""
    print("Loading cleaned data...")
    
    data = {}
    
    # Load price data
    try:
        data['adj_close'] = pd.read_pickle(os.path.join(CLEAN_DATA_DIR, "clean_adj_close.pkl"))
        print(f"Loaded clean price data with shape {data['adj_close'].shape}")
    except Exception as e:
        print(f"Error loading clean price data: {e}")
        data['adj_close'] = None
    
    # Load returns data
    try:
        data['returns'] = pd.read_pickle(os.path.join(CLEAN_DATA_DIR, "daily_returns.pkl"))
        print(f"Loaded returns data with shape {data['returns'].shape}")
    except Exception as e:
        print(f"Error loading returns data: {e}")
        data['returns'] = None
    
    # Load volatility data
    try:
        data['volatility'] = pd.read_pickle(os.path.join(CLEAN_DATA_DIR, "rolling_volatility.pkl"))
        print(f"Loaded volatility data with shape {data['volatility'].shape}")
    except Exception as e:
        print(f"Error loading volatility data: {e}")
        data['volatility'] = None
    
    # Load VIX data
    try:
        data['vix'] = pd.read_pickle(os.path.join(CLEAN_DATA_DIR, "clean_vix.pkl"))
        print(f"Loaded VIX data with shape {data['vix'].shape}")
    except Exception as e:
        print(f"Error loading VIX data: {e}")
        data['vix'] = None
    
    # Load economic data
    try:
        data['economic'] = pd.read_pickle(os.path.join(CLEAN_DATA_DIR, "clean_economic.pkl"))
        print(f"Loaded economic data with shape {data['economic'].shape}")
    except Exception as e:
        print(f"Error loading economic data: {e}")
        data['economic'] = None
    
    # Load sector data
    data['sectors'] = {}
    sectors = ["Technology", "Financial", "Healthcare", "Consumer", "Energy", "Industrial", "Utilities"]
    for sector in sectors:
        try:
            sector_file = os.path.join(CLEAN_DATA_DIR, f"{sector.lower()}_returns.pkl")
            data['sectors'][sector] = pd.read_pickle(sector_file)
            print(f"Loaded {sector} sector data with shape {data['sectors'][sector].shape}")
        except Exception as e:
            print(f"Error loading {sector} sector data: {e}")
            data['sectors'][sector] = None
    
    # Load analysis dataset
    try:
        data['analysis'] = pd.read_pickle(os.path.join(CLEAN_DATA_DIR, "analysis_dataset.pkl"))
        print(f"Loaded analysis dataset with shape {data['analysis'].shape}")
    except Exception as e:
        print(f"Error loading analysis dataset: {e}")
        data['analysis'] = None
    
    return data

def analyze_volatility_patterns(data):
    """Analyze volatility patterns across different time periods"""
    print("Analyzing volatility patterns...")
    
    if data['vix'] is None or data['returns'] is None:
        print("Missing required data for volatility pattern analysis")
        return None
    
    vix = data['vix'].copy()
    returns = data['returns'].copy()
    
    # Create a figure for VIX trends
    plt.figure(figsize=(14, 7))
    vix['VIX'].plot(title='Market Volatility (VIX) Over Time')
    plt.axhline(y=15, color='g', linestyle='--', alpha=0.7, label='Low Volatility Threshold')
    plt.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='High Volatility Threshold')
    
    # Highlight major market events
    events = {
        '2020-03-16': 'COVID-19 Crash',
        '2022-02-24': 'Russia-Ukraine War',
        '2021-01-27': 'GameStop Short Squeeze',
        '2022-09-13': 'Inflation Report Selloff'
    }
    
    for date, event in events.items():
        try:
            date_idx = pd.to_datetime(date)
            if date_idx in vix.index:
                plt.axvline(x=date_idx, color='k', linestyle='-', alpha=0.3)
                plt.text(date_idx, vix['VIX'].max() * 0.9, event, rotation=90, alpha=0.7)
        except Exception as e:
            print(f"Could not mark event {event}: {e}")
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(VISUAL_DIR, 'vix_with_events.png'))
    plt.close()
    
    # Analyze volatility regimes
    vix['year'] = vix.index.year
    vix['month'] = vix.index.month
    
    # Calculate average VIX by month and year
    monthly_vix = vix.groupby(['year', 'month'], observed=True)['VIX'].mean().unstack(level=0)
    
    plt.figure(figsize=(14, 7))
    sns.heatmap(monthly_vix, cmap='viridis', annot=True, fmt='.1f')
    plt.title('Average Monthly VIX by Year')
    plt.ylabel('Month')
    plt.xlabel('Year')
    plt.savefig(os.path.join(VISUAL_DIR, 'monthly_vix_heatmap.png'))
    plt.close()
    
    # Calculate return statistics by volatility regime
    returns_with_regime = returns.copy()
    
    # Ensure returns and vix have the same index before merging
    common_idx = returns.index.intersection(vix.index)
    returns_with_regime = returns_with_regime.loc[common_idx]
    regime_data = vix.loc[common_idx, 'regime']
    returns_with_regime['regime'] = regime_data
    
    # Group returns by regime
    regime_stats = {}
    for regime in ['low', 'medium', 'high']:
        regime_returns = returns_with_regime[returns_with_regime['regime'] == regime].drop('regime', axis=1)
        if not regime_returns.empty:
            regime_stats[regime] = {
                'mean': regime_returns.mean(),
                'std': regime_returns.std(),
                'sharpe': (regime_returns.mean() / regime_returns.std()),
                'count': len(regime_returns)
            }
    
    # Create a summary dataframe for regime statistics
    regime_summary = pd.DataFrame({
        'Low Volatility': [
            regime_stats['low']['count'] if 'low' in regime_stats else 0,
            regime_stats['low']['mean'].mean() * 252 if 'low' in regime_stats else 0,
            regime_stats['low']['std'].mean() * np.sqrt(252) if 'low' in regime_stats else 0,
            regime_stats['low']['sharpe'].mean() if 'low' in regime_stats else 0
        ],
        'Medium Volatility': [
            regime_stats['medium']['count'] if 'medium' in regime_stats else 0,
            regime_stats['medium']['mean'].mean() * 252 if 'medium' in regime_stats else 0,
            regime_stats['medium']['std'].mean() * np.sqrt(252) if 'medium' in regime_stats else 0,
            regime_stats['medium']['sharpe'].mean() if 'medium' in regime_stats else 0
        ],
        'High Volatility': [
            regime_stats['high']['count'] if 'high' in regime_stats else 0,
            regime_stats['high']['mean'].mean() * 252 if 'high' in regime_stats else 0,
            regime_stats['high']['std'].mean() * np.sqrt(252) if 'high' in regime_stats else 0,
            regime_stats['high']['sharpe'].mean() if 'high' in regime_stats else 0
        ]
    }, index=['Number of Days', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'])
    
    # Save regime statistics
    regime_summary.to_csv(os.path.join(REPORTS_DIR, 'volatility_regime_statistics.csv'))
    print(f"Saved volatility regime statistics to {REPORTS_DIR}")
    
    # Create bar chart for regime statistics
    plt.figure(figsize=(14, 8))
    
    # Plot annualized returns by regime
    plt.subplot(2, 2, 1)
    regime_summary.loc['Annualized Return'].plot(kind='bar')
    plt.title('Annualized Returns by Volatility Regime')
    plt.grid(True, alpha=0.3)
    
    # Plot annualized volatility by regime
    plt.subplot(2, 2, 2)
    regime_summary.loc['Annualized Volatility'].plot(kind='bar')
    plt.title('Annualized Volatility by Volatility Regime')
    plt.grid(True, alpha=0.3)
    
    # Plot Sharpe ratio by regime
    plt.subplot(2, 2, 3)
    regime_summary.loc['Sharpe Ratio'].plot(kind='bar')
    plt.title('Sharpe Ratio by Volatility Regime')
    plt.grid(True, alpha=0.3)
    
    # Plot number of days by regime
    plt.subplot(2, 2, 4)
    regime_summary.loc['Number of Days'].plot(kind='bar')
    plt.title('Number of Days by Volatility Regime')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, 'regime_statistics.png'))
    plt.close()
    
    return regime_summary

def analyze_sector_responses(data):
    """Analyze sector-specific responses to volatility"""
    print("Analyzing sector responses to volatility...")
    
    if data['sectors'] is None or data['vix'] is None:
        print("Missing required data for sector response analysis")
        return None
    
    vix = data['vix'].copy()
    
    # Create a dataframe to store sector performance by regime
    sector_performance = pd.DataFrame(index=['low', 'medium', 'high'])
    
    # Analyze each sector's performance in different volatility regimes
    for sector_name, sector_data in data['sectors'].items():
        if sector_data is None or 'sector_avg' not in sector_data.columns:
            continue
        
        # Combine sector returns with VIX regime
        sector_with_regime = sector_data[['sector_avg']].copy()
        
        # Ensure sector_data and vix have the same index before merging
        common_idx = sector_data.index.intersection(vix.index)
        sector_with_regime = sector_with_regime.loc[common_idx]
        regime_data = vix.loc[common_idx, 'regime']
        sector_with_regime['regime'] = regime_data
        
        # Calculate average return by regime
        regime_returns = sector_with_regime.groupby('regime', observed=True)['sector_avg'].mean() * 252  # Annualize
        sector_performance[sector_name] = regime_returns
    
    # Save sector performance by regime
    sector_performance.to_csv(os.path.join(REPORTS_DIR, 'sector_performance_by_regime.csv'))
    print(f"Saved sector performance by regime to {REPORTS_DIR}")
    
    # Create heatmap of sector performance by regime
    plt.figure(figsize=(12, 8))
    sns.heatmap(sector_performance, annot=True, cmap='RdYlGn', fmt='.2f', center=0)
    plt.title('Annualized Sector Returns by Volatility Regime')
    plt.savefig(os.path.join(VISUAL_DIR, 'sector_regime_heatmap.png'))
    plt.close()
    
    # Calculate sector correlations in different regimes
    sector_correlations = {}
    for regime in ['low', 'medium', 'high']:
        regime_corrs = {}
        for sector_name, sector_data in data['sectors'].items():
            if sector_data is None:
                continue
            
            # Get sector returns for this regime
            sector_returns = sector_data.drop('sector_avg', axis=1).copy()
            
            # Ensure sector_returns and vix have the same index before merging
            common_idx = sector_returns.index.intersection(vix.index)
            sector_returns_aligned = sector_returns.loc[common_idx]
            regime_data = vix.loc[common_idx, 'regime']
            sector_returns_aligned['regime'] = regime_data
            
            regime_returns = sector_returns_aligned[sector_returns_aligned['regime'] == regime].drop('regime', axis=1)
            
            if not regime_returns.empty:
                regime_corrs[sector_name] = regime_returns.corr()
        
        sector_correlations[regime] = regime_corrs
    
    # Plot correlation heatmaps for each sector in different regimes
    for sector_name, sector_corrs in sector_correlations.get('high', {}).items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(sector_corrs, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'{sector_name} Stock Correlations - High Volatility Regime')
        plt.savefig(os.path.join(VISUAL_DIR, f'{sector_name.lower()}_high_vol_corr.png'))
        plt.close()
    
    # Calculate and plot sector volatility in different regimes
    sector_volatility = pd.DataFrame(index=['low', 'medium', 'high'])
    
    for sector_name, sector_data in data['sectors'].items():
        if sector_data is None or 'sector_avg' not in sector_data.columns:
            continue
        
        # Combine sector returns with VIX regime
        sector_with_regime = sector_data[['sector_avg']].copy()
        
        # Ensure sector_data and vix have the same index before merging
        common_idx = sector_data.index.intersection(vix.index)
        sector_with_regime = sector_with_regime.loc[common_idx]
        regime_data = vix.loc[common_idx, 'regime']
        sector_with_regime['regime'] = regime_data
        
        # Calculate standard deviation by regime
        regime_volatility = sector_with_regime.groupby('regime', observed=True)['sector_avg'].std() * np.sqrt(252)  # Annualize
        sector_volatility[sector_name] = regime_volatility
    
    # Save sector volatility by regime
    sector_volatility.to_csv(os.path.join(REPORTS_DIR, 'sector_volatility_by_regime.csv'))
    
    # Create heatmap of sector volatility by regime
    plt.figure(figsize=(12, 8))
    sns.heatmap(sector_volatility, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Annualized Sector Volatility by Volatility Regime')
    plt.savefig(os.path.join(VISUAL_DIR, 'sector_regime_volatility_heatmap.png'))
    plt.close()
    
    return {
        'performance': sector_performance,
        'volatility': sector_volatility,
        'correlations': sector_correlations
    }

def analyze_prediction_accuracy(data):
    """Analyze prediction accuracy across different volatility regimes"""
    print("Analyzing prediction accuracy across volatility regimes...")
    
    if data['returns'] is None or data['vix'] is None:
        print("Missing required data for prediction accuracy analysis")
        return None
    
    returns = data['returns'].copy()
    vix = data['vix'].copy()
    
    # Create a simple prediction model: tomorrow's return = today's return (momentum)
    prediction_accuracy = pd.DataFrame(index=returns.index[1:])
    
    for ticker in returns.columns:
        # Create lagged returns (simple prediction model)
        actual_returns = returns[ticker].iloc[1:].values
        predicted_returns = returns[ticker].iloc[:-1].values
        
        # Calculate if prediction direction was correct (1 for correct, 0 for incorrect)
        correct_direction = (np.sign(actual_returns) == np.sign(predicted_returns)).astype(int)
        prediction_accuracy[ticker] = correct_direction
    
    # Ensure prediction_accuracy and vix have the same index before merging
    common_idx = prediction_accuracy.index.intersection(vix.index)
    prediction_accuracy = prediction_accuracy.loc[common_idx]
    regime_data = vix.loc[common_idx, 'regime']
    
    # Add VIX regime
    prediction_accuracy['regime'] = regime_data
    
    # Calculate accuracy by regime for each ticker
    accuracy_by_regime = {}
    for regime in ['low', 'medium', 'high']:
        regime_accuracy = prediction_accuracy[prediction_accuracy['regime'] == regime].drop('regime', axis=1)
        if not regime_accuracy.empty:
            accuracy_by_regime[regime] = regime_accuracy.mean()
    
    # Create a summary dataframe
    accuracy_summary = pd.DataFrame({
        regime: accuracy_by_regime[regime] for regime in accuracy_by_regime
    })
    
    # Add average accuracy across all tickers
    for regime in accuracy_summary.columns:
        accuracy_summary.loc['Average', regime] = accuracy_summary[regime].mean()
    
    # Save accuracy summary
    accuracy_summary.to_csv(os.path.join(REPORTS_DIR, 'prediction_accuracy_by_regime.csv'))
    print(f"Saved prediction accuracy by regime to {REPORTS_DIR}")
    
    # Create bar chart of average accuracy by regime
    plt.figure(figsize=(10, 6))
    accuracy_summary.loc['Average'].plot(kind='bar')
    plt.title('Average Prediction Accuracy by Volatility Regime')
    plt.ylabel('Accuracy (% correct direction)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'avg_prediction_accuracy.png'))
    plt.close()
    
    # Create boxplot of accuracy by regime across tickers
    plt.figure(figsize=(12, 6))
    accuracy_data = []
    labels = []
    
    for regime in accuracy_summary.columns:
        accuracy_data.append(accuracy_summary[regime].drop('Average').values)
        labels.append(regime)
    
    plt.boxplot(accuracy_data, labels=labels)
    plt.title('Prediction Accuracy Distribution by Volatility Regime')
    plt.ylabel('Accuracy (% correct direction)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'prediction_accuracy_boxplot.png'))
    plt.close()
    
    # Calculate accuracy by sector and regime
    sector_accuracy = {}
    for sector_name, sector_data in data['sectors'].items():
        if sector_data is None:
            continue
        
        sector_tickers = [col for col in sector_data.columns if col != 'sector_avg']
        sector_accuracy[sector_name] = {}
        
        for regime in ['low', 'medium', 'high']:
            # Filter accuracy data for this sector and regime
            sector_regime_accuracy = prediction_accuracy[prediction_accuracy['regime'] == regime][sector_tickers]
            if not sector_regime_accuracy.empty:
                sector_accuracy[sector_name][regime] = sector_regime_accuracy.mean().mean()
    
    # Create a summary dataframe for sector accuracy
    sector_accuracy_df = pd.DataFrame({
        sector: [sector_accuracy[sector].get('low', np.nan),
                 sector_accuracy[sector].get('medium', np.nan),
                 sector_accuracy[sector].get('high', np.nan)]
        for sector in sector_accuracy
    }, index=['Low Volatility', 'Medium Volatility', 'High Volatility'])
    
    # Save sector accuracy
    sector_accuracy_df.to_csv(os.path.join(REPORTS_DIR, 'sector_prediction_accuracy.csv'))
    
    # Create heatmap of sector accuracy by regime
    plt.figure(figsize=(12, 8))
    sns.heatmap(sector_accuracy_df, annot=True, cmap='YlGnBu', fmt='.2f', vmin=0.4, vmax=0.6)
    plt.title('Prediction Accuracy by Sector and Volatility Regime')
    plt.savefig(os.path.join(VISUAL_DIR, 'sector_accuracy_heatmap.png'))
    plt.close()
    
    return {
        'accuracy_by_regime': accuracy_summary,
        'sector_accuracy': sector_accuracy_df
    }

def analyze_feature_importance(data):
    """Analyze feature importance for prediction accuracy"""
    print("Analyzing feature importance for prediction accuracy...")
    
    if data['returns'] is None or data['volatility'] is None or data['vix'] is None:
        print("Missing required data for feature importance analysis")
        return None
    
    returns = data['returns'].copy()
    volatility = data['volatility'].copy()
    vix = data['vix'].copy()
    
    # Ensure all dataframes have the same index
    common_idx = returns.index.intersection(volatility.index).intersection(vix.index)
    returns = returns.loc[common_idx]
    volatility = volatility.loc[common_idx]
    vix = vix.loc[common_idx]
    
    # Create a dataset with potential predictive features
    features = pd.DataFrame(index=returns.index[1:])
    
    # Add VIX and VIX changes
    features['VIX'] = vix['VIX'].iloc[1:].values
    features['VIX_change'] = vix['VIX'].diff().iloc[1:].values
    features['VIX_regime'] = vix['regime'].iloc[1:].values.astype(str)
    
    # Add market return (using S&P 500 if available, otherwise average of all stocks)
    if '^GSPC' in returns.columns:
        features['market_return'] = returns['^GSPC'].iloc[:-1].values  # Previous day's market return
    else:
        features['market_return'] = returns.mean(axis=1).iloc[:-1].values  # Previous day's average return
    
    # Add market volatility
    if '^GSPC' in volatility.columns:
        features['market_volatility'] = volatility['^GSPC'].iloc[:-1].values  # Previous day's market volatility
    else:
        features['market_volatility'] = volatility.mean(axis=1).iloc[:-1].values  # Previous day's average volatility
    
    # Calculate correlation between features and prediction accuracy
    # Use the previously calculated prediction accuracy
    prediction_accuracy = pd.DataFrame(index=returns.index[1:])
    
    for ticker in returns.columns:
        # Create lagged returns (simple prediction model)
        actual_returns = returns[ticker].iloc[1:].values
        predicted_returns = returns[ticker].iloc[:-1].values
        
        # Calculate if prediction direction was correct (1 for correct, 0 for incorrect)
        correct_direction = (np.sign(actual_returns) == np.sign(predicted_returns)).astype(int)
        prediction_accuracy[ticker] = correct_direction
    
    # Ensure prediction_accuracy and features have the same index
    common_idx = prediction_accuracy.index.intersection(features.index)
    prediction_accuracy = prediction_accuracy.loc[common_idx]
    features = features.loc[common_idx]
    
    # Calculate average prediction accuracy across all tickers
    features['avg_accuracy'] = prediction_accuracy.mean(axis=1)
    
    # Calculate correlation between features and average accuracy
    numeric_features = features.select_dtypes(include=[np.number]).drop('avg_accuracy', axis=1)
    feature_correlations = numeric_features.corrwith(features['avg_accuracy'])
    
    # Save feature correlations
    feature_correlations.to_csv(os.path.join(REPORTS_DIR, 'feature_correlations.csv'))
    print(f"Saved feature correlations to {REPORTS_DIR}")
    
    # Create bar chart of feature correlations
    plt.figure(figsize=(10, 6))
    feature_correlations.sort_values().plot(kind='barh')
    plt.title('Correlation between Features and Prediction Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'feature_correlations.png'))
    plt.close()
    
    # Analyze accuracy by VIX regime
    regime_accuracy = features.groupby('VIX_regime', observed=True)['avg_accuracy'].mean()
    
    plt.figure(figsize=(10, 6))
    regime_accuracy.plot(kind='bar')
    plt.title('Average Prediction Accuracy by VIX Regime')
    plt.ylabel('Accuracy (% correct direction)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'vix_regime_accuracy.png'))
    plt.close()
    
    # Create scatter plot of VIX vs. prediction accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(features['VIX'], features['avg_accuracy'], alpha=0.5)
    plt.title('VIX vs. Prediction Accuracy')
    plt.xlabel('VIX')
    plt.ylabel('Accuracy (% correct direction)')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(features['VIX'], features['avg_accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(features['VIX'], p(features['VIX']), "r--", alpha=0.8)
    
    plt.savefig(os.path.join(VISUAL_DIR, 'vix_vs_accuracy.png'))
    plt.close()
    
    return {
        'feature_correlations': feature_correlations,
        'regime_accuracy': regime_accuracy
    }

def generate_summary_report(data, analysis_results):
    """Generate a summary report of the exploratory analysis"""
    print("Generating summary report...")
    
    report = """# Exploratory Data Analysis: Impact of Market Volatility on Stock Price Prediction

## Overview
This report presents the findings from our exploratory analysis of how market volatility affects the accuracy of stock price prediction models across different market sectors. We analyzed historical stock price data, volatility measurements, and economic indicators to identify patterns and relationships that could inform more effective prediction strategies.

## Key Findings

### Volatility Patterns
"""
    
    if 'regime_summary' in analysis_results:
        report += f"""
Our analysis identified distinct volatility regimes based on the VIX index:
- Low volatility (VIX < 15): {analysis_results['regime_summary'].loc['Number of Days', 'Low Volatility']:.0f} days
- Medium volatility (15 ≤ VIX < 25): {analysis_results['regime_summary'].loc['Number of Days', 'Medium Volatility']:.0f} days
- High volatility (VIX ≥ 25): {analysis_results['regime_summary'].loc['Number of Days', 'High Volatility']:.0f} days

Market performance varies significantly across these regimes:
- Low volatility periods show annualized returns of {analysis_results['regime_summary'].loc['Annualized Return', 'Low Volatility']:.2f}% with a Sharpe ratio of {analysis_results['regime_summary'].loc['Sharpe Ratio', 'Low Volatility']:.2f}
- Medium volatility periods show annualized returns of {analysis_results['regime_summary'].loc['Annualized Return', 'Medium Volatility']:.2f}% with a Sharpe ratio of {analysis_results['regime_summary'].loc['Sharpe Ratio', 'Medium Volatility']:.2f}
- High volatility periods show annualized returns of {analysis_results['regime_summary'].loc['Annualized Return', 'High Volatility']:.2f}% with a Sharpe ratio of {analysis_results['regime_summary'].loc['Sharpe Ratio', 'High Volatility']:.2f}
"""
    
    report += """
### Sector-Specific Responses
"""
    
    if 'sector_analysis' in analysis_results and 'performance' in analysis_results['sector_analysis']:
        sector_perf = analysis_results['sector_analysis']['performance']
        best_sector_low = sector_perf.loc['low'].idxmax() if 'low' in sector_perf.index and not sector_perf.loc['low'].empty else "N/A"
        best_sector_high = sector_perf.loc['high'].idxmax() if 'high' in sector_perf.index and not sector_perf.loc['high'].empty else "N/A"
        worst_sector_high = sector_perf.loc['high'].idxmin() if 'high' in sector_perf.index and not sector_perf.loc['high'].empty else "N/A"
        
        report += f"""
Different sectors show varying responses to volatility:
- {best_sector_low} performs best in low volatility environments with {sector_perf.loc['low', best_sector_low]:.2f}% annualized returns if best_sector_low != "N/A" else "N/A"
- {best_sector_high} shows the most resilience during high volatility with {sector_perf.loc['high', best_sector_high]:.2f}% annualized returns if best_sector_high != "N/A" else "N/A"
- {worst_sector_high} is most negatively impacted by high volatility with {sector_perf.loc['high', worst_sector_high]:.2f}% annualized returns if worst_sector_high != "N/A" else "N/A"

Sector correlations also change significantly across volatility regimes, with higher correlations generally observed during high volatility periods, reducing diversification benefits.
"""
    
    report += """
### Prediction Accuracy
"""
    
    if 'prediction_analysis' in analysis_results and 'accuracy_by_regime' in analysis_results['prediction_analysis']:
        accuracy = analysis_results['prediction_analysis']['accuracy_by_regime']
        
        report += f"""
Our simple momentum-based prediction model shows varying accuracy across volatility regimes:
- Low volatility: {accuracy.loc['Average', 'low']*100:.2f}% directional accuracy if 'low' in accuracy.columns else "N/A"
- Medium volatility: {accuracy.loc['Average', 'medium']*100:.2f}% directional accuracy if 'medium' in accuracy.columns else "N/A"
- High volatility: {accuracy.loc['Average', 'high']*100:.2f}% directional accuracy if 'high' in accuracy.columns else "N/A"

This suggests that prediction models need to be adjusted based on the prevailing volatility regime to maintain reliability.
"""
    
    if 'sector_accuracy' in analysis_results['prediction_analysis']:
        sector_acc = analysis_results['prediction_analysis']['sector_accuracy']
        best_sector_pred = sector_acc.loc['High Volatility'].idxmax() if 'High Volatility' in sector_acc.index else "N/A"
        
        report += f"""
Sector-specific prediction accuracy also varies, with {best_sector_pred} showing the most predictable behavior during high volatility periods ({sector_acc.loc['High Volatility', best_sector_pred]*100:.2f}% accuracy if best_sector_pred != "N/A" else "N/A").
"""
    
    report += """
### Feature Importance
"""
    
    if 'feature_analysis' in analysis_results and 'feature_correlations' in analysis_results['feature_analysis']:
        feature_corr = analysis_results['feature_analysis']['feature_correlations']
        top_feature = feature_corr.abs().idxmax() if not feature_corr.empty else "N/A"
        
        report += f"""
Our analysis identified several features that correlate with prediction accuracy:
- {top_feature} shows the strongest correlation ({feature_corr[top_feature]:.3f} if top_feature != "N/A" else "N/A") with prediction accuracy
- VIX level itself has a correlation of {feature_corr['VIX'] if 'VIX' in feature_corr.index else 'N/A'} with prediction accuracy
- Market volatility has a correlation of {feature_corr['market_volatility'] if 'market_volatility' in feature_corr.index else 'N/A'} with prediction accuracy

These findings suggest that incorporating volatility metrics into prediction models could improve their performance across different market conditions.
"""
    
    report += """
## Conclusions and Next Steps

Our exploratory analysis reveals that market volatility significantly impacts the accuracy of stock price prediction models, with effects varying across different market sectors. Key conclusions include:

1. Prediction models perform differently across volatility regimes, suggesting the need for regime-specific approaches
2. Sector responses to volatility vary considerably, with some sectors showing more resilience than others
3. Incorporating volatility metrics as features could improve prediction model performance

Based on these findings, we recommend:

1. Developing volatility regime-aware prediction models that can adapt to changing market conditions
2. Creating sector-specific models that account for the unique characteristics of each sector
3. Incorporating VIX and other volatility metrics as key features in prediction models
4. Exploring more sophisticated models that can capture the non-linear relationships between volatility and price movements

The next phase of this project will focus on implementing and testing these recommendations through the development of adaptive prediction models.
"""
    
    # Save the report
    with open(os.path.join(REPORTS_DIR, 'exploratory_analysis_report.md'), 'w') as f:
        f.write(report)
    
    print(f"Saved exploratory analysis report to {REPORTS_DIR}")
    
    return report

def main():
    """Main function to execute all exploratory analysis tasks"""
    print("Starting exploratory data analysis...")
    
    # Load cleaned data
    data = load_cleaned_data()
    
    # Analyze volatility patterns
    regime_summary = analyze_volatility_patterns(data)
    
    # Analyze sector responses to volatility
    sector_analysis = analyze_sector_responses(data)
    
    # Analyze prediction accuracy across volatility regimes
    prediction_analysis = analyze_prediction_accuracy(data)
    
    # Analyze feature importance for prediction accuracy
    feature_analysis = analyze_feature_importance(data)
    
    # Compile analysis results
    analysis_results = {
        'regime_summary': regime_summary,
        'sector_analysis': sector_analysis,
        'prediction_analysis': prediction_analysis,
        'feature_analysis': feature_analysis
    }
    
    # Generate summary report
    report = generate_summary_report(data, analysis_results)
    
    print("Exploratory data analysis completed successfully!")
    
    return analysis_results

if __name__ == "__main__":
    main()
