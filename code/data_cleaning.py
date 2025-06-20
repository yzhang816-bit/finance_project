#!/usr/bin/env python3
"""
Updated Data Cleaning Script for Finance Project

This script processes the raw financial data collected by data_collection.py:
- Handles missing values
- Removes outliers
- Normalizes data
- Creates derived features
- Extracts sector-specific data
- Prepares data for analysis

The cleaned data is saved to the data directory for further analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set up paths
DATA_DIR = "/home/ubuntu/finance_project/data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
VISUAL_DIR = "/home/ubuntu/finance_project/visualizations"
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# Define sectors and their tickers
SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "ADBE"],
    "Financial": ["JPM", "BAC", "GS", "MS", "V"],
    "Healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABBV"],
    "Consumer": ["AMZN", "WMT", "PG", "KO", "MCD"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
    "Industrial": ["CAT", "GE", "MMM", "HON", "BA"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"]
}

# Define market indices
INDICES = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow Jones, NASDAQ, Russell 2000

# Define volatility indicators
VOLATILITY = ["^VIX"]  # CBOE Volatility Index

def load_data():
    """Load all collected datasets"""
    print("Loading collected data...")
    
    data = {}
    
    # Load stock data
    try:
        data['all_stocks'] = pd.read_pickle(os.path.join(DATA_DIR, "all_stock_data.pkl"))
        print(f"Loaded stock data with shape {data['all_stocks'].shape}")
    except Exception as e:
        print(f"Error loading stock data: {e}")
        data['all_stocks'] = None
    
    # Load economic indicators
    try:
        data['economic'] = pd.read_pickle(os.path.join(DATA_DIR, "economic_indicators.pkl"))
        print(f"Loaded economic indicators with shape {data['economic'].shape}")
    except Exception as e:
        print(f"Error loading economic indicators: {e}")
        data['economic'] = None
    
    # Load company fundamentals
    try:
        with open(os.path.join(DATA_DIR, "company_fundamentals.pkl"), 'rb') as f:
            data['fundamentals'] = pickle.load(f)
        print(f"Loaded company fundamentals for {len(data['fundamentals'])} companies")
    except Exception as e:
        print(f"Error loading company fundamentals: {e}")
        data['fundamentals'] = None
    
    return data

def extract_price_data(all_stocks_data):
    """Extract price data for different categories from the all_stocks_data"""
    print("Extracting price data for different categories...")
    
    if all_stocks_data is None:
        print("No stock data available")
        return None
    
    # Initialize dictionaries to store extracted data
    extracted = {
        'sectors': {},
        'indices': None,
        'volatility': None
    }
    
    # Get all unique tickers from the data
    all_tickers = set()
    for col in all_stocks_data.columns:
        # Column is a tuple like ((price_type, ticker), ticker)
        ticker = col[1]  # Extract ticker from the second element
        all_tickers.add(ticker)
    
    print(f"Found {len(all_tickers)} unique tickers in the data")
    
    # Extract sector data
    for sector_name, sector_tickers in SECTORS.items():
        # Find tickers that exist in our data
        available_tickers = [ticker for ticker in sector_tickers if ticker in all_tickers]
        
        if not available_tickers:
            print(f"No data available for {sector_name} sector")
            continue
        
        # Extract data for this sector
        sector_data = pd.DataFrame(index=all_stocks_data.index)
        for ticker in available_tickers:
            # Extract all columns for this ticker
            ticker_cols = [col for col in all_stocks_data.columns if col[1] == ticker]
            if ticker_cols:
                for col in ticker_cols:
                    col_name = f"{col[0][0]}_{ticker}"  # Create a new column name
                    sector_data[col_name] = all_stocks_data[col]
        
        if not sector_data.empty:
            extracted['sectors'][sector_name] = sector_data
            print(f"Extracted data for {sector_name} sector with {len(available_tickers)} stocks")
    
    # Extract indices data
    available_indices = [idx for idx in INDICES if idx in all_tickers]
    if available_indices:
        indices_data = pd.DataFrame(index=all_stocks_data.index)
        for idx in available_indices:
            idx_cols = [col for col in all_stocks_data.columns if col[1] == idx]
            if idx_cols:
                for col in idx_cols:
                    col_name = f"{col[0][0]}_{idx}"
                    indices_data[col_name] = all_stocks_data[col]
        
        if not indices_data.empty:
            extracted['indices'] = indices_data
            print(f"Extracted data for {len(available_indices)} market indices")
    else:
        print("No market indices data available")
    
    # Extract volatility data
    available_vol = [vol for vol in VOLATILITY if vol in all_tickers]
    if available_vol:
        vol_data = pd.DataFrame(index=all_stocks_data.index)
        for vol in available_vol:
            vol_cols = [col for col in all_stocks_data.columns if col[1] == vol]
            if vol_cols:
                for col in vol_cols:
                    col_name = f"{col[0][0]}_{vol}"
                    vol_data[col_name] = all_stocks_data[col]
        
        if not vol_data.empty:
            extracted['volatility'] = vol_data
            print(f"Extracted volatility data")
    else:
        print("No volatility data available")
    
    return extracted

def extract_adj_close_prices(all_stocks_data):
    """Extract adjusted close prices for all tickers"""
    print("Extracting adjusted close prices...")
    
    if all_stocks_data is None:
        print("No stock data available")
        return None
    
    # Create a new dataframe for adjusted close prices
    adj_close = pd.DataFrame(index=all_stocks_data.index)
    
    # Get all unique tickers
    all_tickers = set()
    for col in all_stocks_data.columns:
        ticker = col[1]  # Extract ticker from the second element
        all_tickers.add(ticker)
    
    # Find the column name pattern for adjusted close prices
    # First, check if 'Adj Close' exists in any column
    adj_close_pattern = None
    for col in all_stocks_data.columns:
        if col[0][0] == 'Adj Close':
            adj_close_pattern = 'Adj Close'
            break
    
    # If not found, try 'Close' as fallback
    if adj_close_pattern is None:
        for col in all_stocks_data.columns:
            if col[0][0] == 'Close':
                adj_close_pattern = 'Close'
                break
    
    if adj_close_pattern is None:
        print("Could not find adjusted close or close price columns")
        return None
    
    # Extract adjusted close prices for each ticker
    for ticker in all_tickers:
        try:
            # Find the column for this ticker and price type
            for col in all_stocks_data.columns:
                if col[0][0] == adj_close_pattern and col[1] == ticker:
                    adj_close[ticker] = all_stocks_data[col]
                    break
        except Exception as e:
            print(f"Error extracting {adj_close_pattern} prices for {ticker}: {e}")
    
    print(f"Extracted {adj_close_pattern} prices for {len(adj_close.columns)} tickers")
    return adj_close

def clean_price_data(adj_close):
    """Clean and process price data"""
    print("Cleaning price data...")
    
    if adj_close is None or adj_close.empty:
        print("No price data available to clean")
        return None
    
    # Make a copy to avoid modifying the original
    df = adj_close.copy()
    
    # Handle missing values
    # Forward fill for most recent available price
    df = df.ffill()
    
    # If still missing (at the beginning), backward fill
    df = df.bfill()
    
    # Check for any remaining NaN values
    if df.isna().sum().sum() > 0:
        print(f"Warning: {df.isna().sum().sum()} NaN values remain after filling")
    
    # Save cleaned adjusted close prices
    df.to_pickle(os.path.join(CLEAN_DATA_DIR, "clean_adj_close.pkl"))
    print(f"Saved cleaned price data for {len(df.columns)} tickers")
    
    # Calculate daily returns
    returns = df.pct_change().dropna()
    returns.to_pickle(os.path.join(CLEAN_DATA_DIR, "daily_returns.pkl"))
    print("Calculated and saved daily returns")
    
    # Calculate rolling volatility (20-day standard deviation of returns)
    volatility = returns.rolling(window=20).std().dropna()
    volatility.to_pickle(os.path.join(CLEAN_DATA_DIR, "rolling_volatility.pkl"))
    print("Calculated and saved rolling volatility")
    
    return {
        'adj_close': df,
        'returns': returns,
        'volatility': volatility
    }

def extract_vix_data(all_stocks_data):
    """Extract VIX data from all stocks data"""
    print("Extracting VIX data...")
    
    if all_stocks_data is None:
        print("No stock data available")
        return None
    
    vix_data = pd.DataFrame(index=all_stocks_data.index)
    
    # Look for VIX ticker
    vix_ticker = "^VIX"
    vix_found = False
    
    # Try to find Close price for VIX
    for col in all_stocks_data.columns:
        if col[1] == vix_ticker and col[0][0] == 'Close':
            vix_data['VIX'] = all_stocks_data[col]
            vix_found = True
            break
    
    if not vix_found:
        print(f"VIX data not found in the dataset")
        # Create a synthetic VIX based on S&P 500 volatility as fallback
        try:
            sp500_ticker = "^GSPC"
            for col in all_stocks_data.columns:
                if col[1] == sp500_ticker and col[0][0] == 'Close':
                    sp500 = all_stocks_data[col]
                    sp500_returns = sp500.pct_change().dropna()
                    synthetic_vix = sp500_returns.rolling(window=20).std() * np.sqrt(252) * 100
                    vix_data['VIX'] = synthetic_vix
                    print("Created synthetic VIX from S&P 500 volatility")
                    vix_found = True
                    break
        except Exception as e:
            print(f"Error creating synthetic VIX: {e}")
    
    if not vix_found:
        print("Could not extract or create VIX data")
        return None
    
    # Handle missing values
    vix_data = vix_data.ffill().bfill()
    
    # Create volatility regimes based on VIX levels
    # Low volatility: VIX < 15
    # Medium volatility: 15 <= VIX < 25
    # High volatility: VIX >= 25
    vix_data['regime'] = pd.cut(vix_data['VIX'], 
                              bins=[0, 15, 25, float('inf')],
                              labels=['low', 'medium', 'high'])
    
    # Save cleaned volatility data
    vix_data.to_pickle(os.path.join(CLEAN_DATA_DIR, "clean_vix.pkl"))
    print("Saved VIX data with regime classification")
    
    return vix_data

def clean_economic_data(economic_data):
    """Clean and process economic indicators"""
    print("Cleaning economic data...")
    
    if economic_data is None or economic_data.empty:
        print("No economic data available to clean")
        return None
    
    # Make a copy to avoid modifying the original
    df = economic_data.copy()
    
    # Handle missing values
    # For economic data, forward fill is appropriate as most indicators are reported periodically
    df = df.ffill()
    
    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Converting economic data index to DatetimeIndex")
        df.index = pd.to_datetime(df.index)
    
    # Resample to daily frequency if needed (some economic indicators are monthly or quarterly)
    # This ensures alignment with daily stock data
    if df.index.inferred_freq != 'D':
        df = df.asfreq('D')
        # Forward fill after resampling
        df = df.ffill()
    
    # Save cleaned economic data
    df.to_pickle(os.path.join(CLEAN_DATA_DIR, "clean_economic.pkl"))
    print("Saved cleaned economic data")
    
    return df

def create_sector_datasets(price_data):
    """Create sector-specific datasets for analysis"""
    print("Creating sector-specific datasets...")
    
    if price_data is None or 'returns' not in price_data:
        print("No return data available for sector datasets")
        return None
    
    returns = price_data['returns']
    
    # Create sector-specific return datasets
    sector_data = {}
    for sector_name, tickers in SECTORS.items():
        # Filter for tickers in this sector that exist in our data
        available_tickers = [ticker for ticker in tickers if ticker in returns.columns]
        
        if not available_tickers:
            print(f"No data available for {sector_name} sector")
            continue
        
        # Create sector dataset
        sector_returns = returns[available_tickers]
        
        # Calculate sector average return
        sector_returns['sector_avg'] = sector_returns.mean(axis=1)
        
        # Save sector dataset
        sector_returns.to_pickle(os.path.join(CLEAN_DATA_DIR, f"{sector_name.lower()}_returns.pkl"))
        print(f"Saved {sector_name} sector returns with {len(available_tickers)} stocks")
        
        # Store for later use
        sector_data[sector_name] = sector_returns
    
    return sector_data

def create_analysis_dataset(price_data, vix_data, economic_data):
    """Create a combined dataset for analysis"""
    print("Creating combined analysis dataset...")
    
    if price_data is None or 'returns' not in price_data:
        print("No return data available for combined dataset")
        return None
    
    # Start with the stock returns
    combined = price_data['returns'].copy()
    
    # Add rolling volatility
    for col in price_data['volatility'].columns:
        combined[f"{col}_vol"] = price_data['volatility'][col]
    
    # Add VIX if available
    if vix_data is not None and not vix_data.empty:
        combined['VIX'] = vix_data['VIX']
        # Convert regime to numeric for easier analysis
        regime_map = {'low': 0, 'medium': 1, 'high': 2}
        combined['volatility_regime'] = vix_data['regime'].map(regime_map)
    
    # Add economic indicators if available
    if economic_data is not None and not economic_data.empty:
        # Align economic data with stock data dates
        for col in economic_data.columns:
            economic_series = economic_data[col].reindex(combined.index)
            combined[f"econ_{col}"] = economic_series
    
    # Drop rows with NaN values
    combined = combined.dropna()
    
    # Save combined dataset
    combined.to_pickle(os.path.join(CLEAN_DATA_DIR, "analysis_dataset.pkl"))
    print(f"Saved combined analysis dataset with {combined.shape[1]} features and {combined.shape[0]} observations")
    
    return combined

def generate_summary_statistics(price_data, sector_data=None):
    """Generate summary statistics for the cleaned data"""
    print("Generating summary statistics...")
    
    if price_data is None or 'returns' not in price_data or price_data['returns'].empty:
        print("No return data available for summary statistics")
        return None
    
    returns = price_data['returns']
    
    # Calculate basic statistics
    stats = returns.describe()
    
    # Calculate annualized returns and volatility
    annualized_returns = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = annualized_returns / annualized_volatility
    
    # Add to statistics dataframe
    stats.loc['annualized_return'] = annualized_returns
    stats.loc['annualized_volatility'] = annualized_volatility
    stats.loc['sharpe_ratio'] = sharpe_ratio
    
    # Save summary statistics
    stats.to_csv(os.path.join(CLEAN_DATA_DIR, "summary_statistics.csv"))
    print("Saved summary statistics")
    
    # Generate sector-specific statistics if available
    if sector_data:
        for sector_name, sector_returns in sector_data.items():
            sector_stats = sector_returns.describe()
            sector_annualized_returns = sector_returns.mean() * 252
            sector_annualized_volatility = sector_returns.std() * np.sqrt(252)
            sector_sharpe_ratio = sector_annualized_returns / sector_annualized_volatility
            
            sector_stats.loc['annualized_return'] = sector_annualized_returns
            sector_stats.loc['annualized_volatility'] = sector_annualized_volatility
            sector_stats.loc['sharpe_ratio'] = sector_sharpe_ratio
            
            sector_stats.to_csv(os.path.join(CLEAN_DATA_DIR, f"{sector_name.lower()}_statistics.csv"))
            print(f"Saved {sector_name} sector statistics")
    
    return stats

def generate_initial_visualizations(price_data, vix_data, sector_data=None):
    """Generate initial visualizations for exploratory analysis"""
    print("Generating initial visualizations...")
    
    if price_data is None or 'adj_close' not in price_data or price_data['adj_close'].empty:
        print("No price data available for visualizations")
        return False
    
    # 1. Plot price trends for major indices or representative stocks
    plt.figure(figsize=(12, 6))
    # Select a few representative stocks (one from each sector if possible)
    rep_stocks = []
    for sector, tickers in SECTORS.items():
        for ticker in tickers:
            if ticker in price_data['adj_close'].columns:
                rep_stocks.append(ticker)
                break
    
    # Limit to 5 stocks for clarity
    rep_stocks = rep_stocks[:5]
    
    # Normalize prices to 100 at the start for comparison
    normalized_prices = price_data['adj_close'][rep_stocks].div(price_data['adj_close'][rep_stocks].iloc[0]).mul(100)
    normalized_prices.plot(title='Normalized Price Trends (Base=100)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'price_trends.png'))
    plt.close()
    
    # 2. Plot VIX (volatility) over time if available
    if vix_data is not None and not vix_data.empty and 'VIX' in vix_data.columns:
        plt.figure(figsize=(12, 6))
        vix_data['VIX'].plot(title='Market Volatility (VIX) Over Time')
        plt.axhline(y=15, color='g', linestyle='--', alpha=0.7, label='Low Volatility Threshold')
        plt.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='High Volatility Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(VISUAL_DIR, 'vix_trend.png'))
        plt.close()
        
        # 3. Plot volatility regimes distribution
        if 'regime' in vix_data.columns:
            plt.figure(figsize=(10, 6))
            vix_data['regime'].value_counts().plot(kind='bar', title='Distribution of Volatility Regimes')
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(os.path.join(VISUAL_DIR, 'volatility_regimes.png'))
            plt.close()
    
    # 4. Plot rolling volatility for representative stocks
    plt.figure(figsize=(12, 6))
    price_data['volatility'][rep_stocks].plot(title='20-Day Rolling Volatility')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUAL_DIR, 'rolling_volatility.png'))
    plt.close()
    
    # 5. Plot sector performance comparison if available
    if sector_data:
        # Create a dataframe with sector average returns
        sector_avg_returns = pd.DataFrame()
        for sector_name, sector_returns in sector_data.items():
            if 'sector_avg' in sector_returns.columns:
                sector_avg_returns[sector_name] = sector_returns['sector_avg']
        
        if not sector_avg_returns.empty:
            # Calculate cumulative returns
            cumulative_returns = (1 + sector_avg_returns).cumprod()
            
            plt.figure(figsize=(12, 6))
            cumulative_returns.plot(title='Cumulative Sector Performance')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(VISUAL_DIR, 'sector_performance.png'))
            plt.close()
    
    print(f"Saved initial visualizations to {VISUAL_DIR}")
    return True

def main():
    """Main function to execute all data cleaning tasks"""
    print("Starting data cleaning process...")
    
    # Load all collected data
    data = load_data()
    
    # Extract adjusted close prices
    adj_close = extract_adj_close_prices(data['all_stocks'])
    
    # Clean price data
    price_data = clean_price_data(adj_close)
    
    # Extract VIX data
    vix_data = extract_vix_data(data['all_stocks'])
    
    # Clean economic data
    economic_data = clean_economic_data(data['economic'])
    
    # Create sector-specific datasets
    sector_data = create_sector_datasets(price_data)
    
    # Create combined dataset for analysis
    analysis_data = create_analysis_dataset(price_data, vix_data, economic_data)
    
    # Generate summary statistics
    stats = generate_summary_statistics(price_data, sector_data)
    
    # Generate initial visualizations
    generate_initial_visualizations(price_data, vix_data, sector_data)
    
    print("Data cleaning and initial analysis completed successfully!")
    
    return {
        'price_data': price_data,
        'vix_data': vix_data,
        'economic_data': economic_data,
        'sector_data': sector_data,
        'analysis_data': analysis_data,
        'stats': stats
    }

if __name__ == "__main__":
    main()
