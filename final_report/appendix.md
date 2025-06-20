# Appendix: Technical Details

## A. Data Processing Methodology

### A.1 Data Cleaning Procedures
```python
def clean_stock_data(df):
    # Remove rows with missing values
    df = df.dropna(how='all')
    
    # Forward fill minor gaps
    df = df.fillna(method='ffill', limit=3)
    
    # Calculate returns
    df['returns'] = df['Adj Close'].pct_change()
    
    # Remove outliers (3 standard deviations)
    mean = df['returns'].mean()
    std = df['returns'].std()
    df = df[(df['returns'] > mean - 3*std) & (df['returns'] < mean + 3*std)]
    
    return df
```

### A.2 Feature Engineering Details
```python
def engineer_features(df):
    # Calculate rolling volatility (21-day window)
    df['volatility_21d'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    
    # Create lagged features
    df['returns_lag1'] = df['returns'].shift(1)
    df['returns_lag2'] = df['returns'].shift(2)
    df['returns_lag3'] = df['returns'].shift(3)
    
    # Create moving averages
    df['ma_50'] = df['Adj Close'].rolling(window=50).mean()
    df['ma_200'] = df['Adj Close'].rolling(window=200).mean()
    
    # Create technical indicators
    df['rsi_14'] = calculate_rsi(df['Adj Close'], window=14)
    
    return df
```

### A.3 Volatility Regime Classification
```python
def classify_volatility_regime(vix_value):
    if vix_value < 15:
        return 'low'
    elif vix_value < 25:
        return 'medium'
    else:
        return 'high'

# Apply classification to VIX data
vix_df['regime'] = vix_df['VIX'].apply(classify_volatility_regime)
```

## B. Model Development Details

### B.1 Cross-Validation Approach
```python
from sklearn.model_selection import TimeSeriesSplit

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Model training and evaluation
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    performance = evaluate_model(y_test, predictions)
```

### B.2 Model Hyperparameters
```python
# Linear Regression
linear_model = LinearRegression()

# Ridge Regression
ridge_model = Ridge(alpha=1.0)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    bootstrap=True,
    random_state=42
)

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    random_state=42
)
```

### B.3 Feature Importance Calculation
```python
def extract_feature_importance(model, feature_names):
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    # For linear models
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    
    # Create DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df
```

## C. Performance Metrics

### C.1 Evaluation Metrics Formulas
```python
def calculate_metrics(y_true, y_pred):
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # Information Coefficient (Spearman Rank Correlation)
    ic = spearmanr(y_true, y_pred)[0]
    
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'IC': ic
    }
```

### C.2 Statistical Significance Testing
```python
def test_model_significance(model, X, y, n_permutations=1000):
    # Actual performance
    y_pred = model.predict(X)
    actual_r2 = r2_score(y, y_pred)
    
    # Permutation test
    r2_null = []
    for i in range(n_permutations):
        # Shuffle target values
        y_perm = y.sample(frac=1.0).reset_index(drop=True)
        
        # Predict and calculate R2
        y_pred_perm = model.predict(X)
        r2_perm = r2_score(y_perm, y_pred_perm)
        r2_null.append(r2_perm)
    
    # Calculate p-value
    p_value = np.mean([r2 >= actual_r2 for r2 in r2_null])
    
    return {
        'actual_r2': actual_r2,
        'null_mean_r2': np.mean(r2_null),
        'null_std_r2': np.std(r2_null),
        'p_value': p_value
    }
```

## D. Data Sources and References

### D.1 Data Sources
1. **Stock Price Data**: Yahoo Finance API (https://finance.yahoo.com/)
2. **Economic Data**: Federal Reserve Economic Data (FRED) (https://fred.stlouisfed.org/)
3. **Market Indices**: S&P 500, VIX Index via Yahoo Finance

### D.2 Software Libraries
- **Data Processing**: pandas (1.3.5), numpy (1.21.5)
- **Visualization**: matplotlib (3.5.1), seaborn (0.11.2)
- **Modeling**: scikit-learn (1.0.2)
- **Statistical Analysis**: scipy (1.7.3)

### D.3 Academic References
1. Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? The Review of Financial Studies, 21(4), 1509-1531.
2. Ang, A., & Bekaert, G. (2007). Stock return predictability: Is it there? The Review of Financial Studies, 20(3), 651-707.
3. Rapach, D. E., Strauss, J. K., & Zhou, G. (2010). Out-of-sample equity premium prediction: Combination forecasts and links to the real economy. The Review of Financial Studies, 23(2), 821-862.
4. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223-2273.
