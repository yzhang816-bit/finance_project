#!/usr/bin/env python3
"""
Data Collection Script for Finance Project

This script collects financial data from various sources including:
- Yahoo Finance API for stock price data
- CBOE VIX data for market volatility
- Federal Reserve Economic Data (FRED) for economic indicators

The collected data is saved to CSV files in the data directory.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import requests
import json
import time
from pandas_datareader import data as pdr

# Set up paths
DATA_DIR = "/home/ubuntu/finance_project/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configure date ranges
END_DATE = datetime.datetime.now()
START_DATE = END_DATE - datetime.timedelta(days=5*365)  # 5 years of data

# Define sectors and representative stocks
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

def download_stock_data():
    """Download historical stock data for all sectors and indices"""
    print("Downloading stock data...")
    
    # Combine all tickers
    all_tickers = []
    for sector, tickers in SECTORS.items():
        all_tickers.extend(tickers)
    all_tickers.extend(INDICES)
    all_tickers.extend(VOLATILITY)
    
    # Download data using yfinance directly
    data_frames = {}
    
    # Download data for each ticker
    for ticker in all_tickers:
        try:
            print(f"Downloading data for {ticker}...")
            ticker_data = yf.download(ticker, start=START_DATE, end=END_DATE)
            if not ticker_data.empty:
                data_frames[ticker] = ticker_data
                print(f"Successfully downloaded data for {ticker}")
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    
    # Create a multi-level column dataframe
    combined_data = pd.DataFrame()
    
    # Process each ticker's data
    for ticker, df in data_frames.items():
        # Create multi-level columns for each ticker
        for column in df.columns:
            combined_data[(column, ticker)] = df[column]
    
    # Save complete dataset
    combined_data.to_pickle(os.path.join(DATA_DIR, "all_stock_data.pkl"))
    
    # Save sector-specific datasets
    for sector, tickers in SECTORS.items():
        sector_columns = [(col, ticker) for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
                         for ticker in tickers if (col, ticker) in combined_data.columns]
        if sector_columns:
            sector_data = combined_data[sector_columns]
            sector_data.to_pickle(os.path.join(DATA_DIR, f"{sector.lower()}_stocks.pkl"))
            print(f"Saved {sector} sector data")
        
    # Save indices data
    indices_columns = [(col, ticker) for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
                      for ticker in INDICES if (col, ticker) in combined_data.columns]
    if indices_columns:
        indices_data = combined_data[indices_columns]
        indices_data.to_pickle(os.path.join(DATA_DIR, "market_indices.pkl"))
        print(f"Saved market indices data")
    
    # Save volatility data
    vix_columns = [(col, ticker) for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
                  for ticker in VOLATILITY if (col, ticker) in combined_data.columns]
    if vix_columns:
        vix_data = combined_data[vix_columns]
        vix_data.to_pickle(os.path.join(DATA_DIR, "volatility.pkl"))
        print(f"Saved volatility data")
    
    print(f"Stock data downloaded and saved to {DATA_DIR}")
    return combined_data

def download_economic_indicators():
    """Download economic indicators from FRED"""
    print("Downloading economic indicators...")
    
    try:
        # List of economic indicators to download
        indicators = {
            "GDP": "GDP",              # Gross Domestic Product
            "UNRATE": "UNRATE",        # Unemployment Rate
            "CPIAUCSL": "CPIAUCSL",    # Consumer Price Index
            "FEDFUNDS": "FEDFUNDS",    # Federal Funds Rate
            "T10Y2Y": "T10Y2Y",        # 10-Year Treasury Constant Maturity Minus 2-Year Treasury
            "INDPRO": "INDPRO",        # Industrial Production Index
            "M2": "M2"                 # M2 Money Stock
        }
        
        # Use pandas_datareader to get data from FRED
        economic_data = {}
        for name, series_id in indicators.items():
            try:
                data = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
                economic_data[name] = data
                print(f"Downloaded {name} data")
            except Exception as e:
                print(f"Error downloading {name}: {e}")
        
        # Combine all indicators into a single dataframe
        combined_data = pd.DataFrame()
        for name, data in economic_data.items():
            if not data.empty:
                combined_data[name] = data.iloc[:, 0]
        
        # Save economic data
        combined_data.to_pickle(os.path.join(DATA_DIR, "economic_indicators.pkl"))
        print(f"Economic indicators downloaded and saved to {DATA_DIR}")
        
        return combined_data
    
    except Exception as e:
        print(f"Error downloading economic indicators: {e}")
        return None

def download_company_fundamentals():
    """Download fundamental data for companies"""
    print("Downloading company fundamentals...")
    
    fundamentals = {}
    
    # Collect tickers from all sectors
    all_tickers = []
    for sector, tickers in SECTORS.items():
        all_tickers.extend(tickers)
    
    # Download fundamental data for each ticker
    for ticker in all_tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Get financial statements - handle potential attribute changes in yfinance
            try:
                balance_sheet = stock.balance_sheet
            except:
                balance_sheet = None
                print(f"Could not retrieve balance sheet for {ticker}")
                
            try:
                income_stmt = stock.income_stmt
            except:
                try:
                    income_stmt = stock.financials
                except:
                    income_stmt = None
                    print(f"Could not retrieve income statement for {ticker}")
                
            try:
                cash_flow = stock.cashflow
            except:
                cash_flow = None
                print(f"Could not retrieve cash flow for {ticker}")
            
            # Store data
            fundamentals[ticker] = {
                'info': info,
                'balance_sheet': balance_sheet,
                'income_stmt': income_stmt,
                'cash_flow': cash_flow
            }
            
            print(f"Downloaded fundamentals for {ticker}")
            time.sleep(1)  # Avoid hitting API rate limits
            
        except Exception as e:
            print(f"Error downloading fundamentals for {ticker}: {e}")
    
    # Save fundamentals data
    import pickle
    with open(os.path.join(DATA_DIR, "company_fundamentals.pkl"), 'wb') as f:
        pickle.dump(fundamentals, f)
    
    print(f"Company fundamentals downloaded and saved to {DATA_DIR}")
    return fundamentals

def main():
    """Main function to execute all data collection tasks"""
    print("Starting data collection process...")
    
    # Download all datasets
    stock_data = download_stock_data()
    economic_data = download_economic_indicators()
    fundamentals = download_company_fundamentals()
    
    print("Data collection completed successfully!")
    
    # Return summary of collected data
    summary = {
        "stock_data_shape": None if stock_data is None else stock_data.shape,
        "economic_data_shape": None if economic_data is None else economic_data.shape,
        "fundamentals_count": None if fundamentals is None else len(fundamentals)
    }
    
    print("Data collection summary:")
    print(summary)
    
    return summary

if __name__ == "__main__":
    main()
