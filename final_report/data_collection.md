# Data Collection and Preprocessing

## Data Sources
This project utilized financial market data from the following sources:

1. **Stock Price Data**: Historical daily price data for 40 major stocks across 7 sectors
   - Source: Yahoo Finance API
   - Period: 5 years of daily data
   - Variables: Open, High, Low, Close, Adjusted Close, Volume

2. **Market Volatility Data**: VIX index (market volatility indicator)
   - Source: Yahoo Finance API
   - Period: 5 years of daily data
   - Variables: VIX index values

3. **Economic Indicators**: Key macroeconomic variables
   - Source: Federal Reserve Economic Data (FRED)
   - Period: Monthly data aligned with stock data
   - Variables: Interest rates, money supply, inflation metrics

## Data Collection Process
The data collection process involved the following steps:

1. **API Integration**: Python scripts were developed to access financial data APIs
2. **Data Extraction**: Historical data was extracted for the selected stocks and indices
3. **Initial Validation**: Data was checked for completeness and accuracy
4. **Storage**: Raw data was stored in structured formats for further processing

## Data Preprocessing

### Cleaning Steps
1. **Missing Value Handling**:
   - Forward fill for minor gaps in time series data
   - Interpolation for economic indicators
   - Removal of stocks with significant missing data

2. **Outlier Detection and Treatment**:
   - Z-score based outlier detection
   - Winsorization of extreme values
   - Validation against market events

3. **Feature Engineering**:
   - Calculation of daily returns
   - Creation of rolling volatility measures
   - Development of technical indicators
   - Volatility regime classification

4. **Data Transformation**:
   - Log transformations for skewed distributions
   - Standardization for modeling
   - Temporal alignment of different data frequencies

### Derived Features
Several derived features were created to enhance the analysis:

1. **Return Metrics**:
   - Daily returns
   - Cumulative returns
   - Excess returns over market

2. **Volatility Measures**:
   - Rolling standard deviation (21-day window)
   - Volatility regimes (low, medium, high)
   - Relative volatility to market

3. **Sector Aggregates**:
   - Sector average returns
   - Sector volatility
   - Sector-specific indicators

4. **Technical Indicators**:
   - Moving averages
   - Relative strength
   - Momentum indicators

The cleaned and processed data was organized into structured datasets for exploratory analysis and modeling, ensuring consistency and reliability throughout the analytical process.
