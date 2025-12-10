# 🚀 GBM Streamlit App - Setup & Quick Start Guide

## Quick Start (2 Minutes)

### Option 1: Using the Quick Start Script (Recommended)

```bash
cd /Users/daniel/Desktop/School/gbm_streamlit

# Make script executable (macOS/Linux)
chmod +x run.py

# Run the quick start script
python run.py
```

The script will:
1. ✓ Check your Python version
2. ✓ Install all required dependencies
3. ✓ Launch the Streamlit app automatically

The app will open in your browser at `http://localhost:8501`

### Option 2: Manual Setup

```bash
# Navigate to project directory
cd /Users/daniel/Desktop/School/gbm_streamlit

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## What Gets Installed

From `requirements.txt`:
- **streamlit** - Web app framework
- **numpy** - Numerical computations
- **pandas** - Data manipulation
- **plotly** - Interactive charts
- **openpyxl** - Excel export support

## Project Structure

```
gbm_streamlit/
├── app.py                          # Main Streamlit app (main entry point)
├── run.py                          # Quick start script
├── requirements.txt                # Dependency list
├── README.md                       # Comprehensive documentation
├── SETUP_GUIDE.md                  # This file
├── .gitignore                      # Git ignore patterns
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_loader.py              # CSV loading & validation
│   ├── gbm_simulator.py            # GBM simulation engine
│   ├── plotting.py                 # Plotly visualizations
│   └── statistics.py               # Statistical calculations
│
└── data/                           # Data directory (for your CSV files)
    └── (Your stock data CSV files go here)
```

## First Time Usage

### 1. Start the App
Run one of the quick start methods above.

### 2. Load Your Data
- **Option A: Upload CSV**
  - Click "Upload CSV" in the sidebar
  - Select your stock data file (must have "Date" and "Close" columns)
  
- **Option B: Use Sample Data**
  - Click "Sample Data" to use auto-generated demo data

### 3. Explore the Simulation
- Sliders automatically appear once data is loaded
- Adjust **Drift (μ)** and **Volatility (σ)** values
- Change simulation parameters (days, paths, steps)
- Watch the chart update in real-time

### 4. Analyze Results
- View summary statistics in tabs
- Check distribution of final prices
- Compare returns metrics
- Export simulated data to CSV/Excel

## Preparing Your Stock Data

### CSV Format Requirements

Your CSV file must have:
- Column named **"Date"** or **"date"**
- Column named **"Close"** or **"close"**
- Dates in standard format (YYYY-MM-DD or similar)
- Price values as numbers

### Example CSV Structure

```csv
Date,Close
2023-01-01,150.25
2023-01-02,151.50
2023-01-03,149.75
2023-01-04,152.00
...
```

### Getting Real Data

Popular sources:
- **Yahoo Finance**: https://finance.yahoo.com/
- **Google Finance**: https://www.google.com/finance/
- **Alpha Vantage**: https://www.alphavantage.co/ (API)
- **FRED**: https://fred.stlouisfed.org/ (Economic data)

### Exporting from Excel

If you have data in Excel:
1. Select your Date and Close columns
2. File → Save As → Select "CSV (Comma delimited)"
3. Upload the CSV file to the app

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install dependencies again
```bash
pip install -r requirements.txt
```

### Issue: "CSV must contain 'Date' and 'Close' columns"
**Solution:** Check your CSV headers exactly match (case doesn't matter):
- ✓ "Date", "Close"
- ✓ "date", "close"
- ✓ "DATE", "CLOSE"
- ✗ "date_trading", "closing_price" (wrong names)

### Issue: App runs but won't open in browser
**Solution:** Manually open browser and go to:
```
http://localhost:8501
```

### Issue: "Need at least 2 data points"
**Solution:** Your CSV has too few rows. Load at least 2 prices.

### Issue: Simulation is very slow
**Solution:** Reduce these parameters:
- Number of Paths (try 500 instead of 5000)
- Time Steps per Path
- Number of Days to Simulate

## Understanding the Parameters

### Drift (μ)
- **What it means**: Expected annual return
- **Examples**:
  - 0.10 (10%) = bullish, stock expected to grow
  - 0.00 (0%) = random walk, no bias
  - -0.05 (-5%) = bearish, stock expected to decline
- **How to adjust**: Use estimated value or test scenarios

### Volatility (σ)
- **What it means**: Annual standard deviation of returns
- **Examples**:
  - 0.15 (15%) = low volatility (stable)
  - 0.30 (30%) = moderate volatility
  - 0.50+ (50%+) = high volatility (risky)
- **How to adjust**: Keep realistic or increase for worst-case scenarios

### Number of Days
- **Range**: 5 to 365 days
- **Typical**: 30-60 days for near-term projections
- **Effect**: Longer periods allow more drift/volatility to manifest

### Number of Paths
- **Range**: 10 to 10,000
- **Recommended**: 1,000-5,000
- **Effect**: More paths = more accurate statistics but slower

### Time Steps
- **What it means**: Resolution of simulation (daily updates)
- **Typical**: Same as number of days
- **Effect**: More steps = smoother paths but slower computation

## Streamlit Tips

### Keyboard Shortcuts
- `r` - Rerun script
- `c` - Clear cache
- `Ctrl+C` - Stop server (in terminal)

### Streamlit Configuration
Create `.streamlit/config.toml` to customize:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
maxUploadSize = 200
```

### Running on Different Port
```bash
streamlit run app.py --server.port 8502
```

## Performance Tips

### For Large Datasets (1000+ rows)
- Use 1000-2000 paths instead of 10,000
- Reduce time steps

### For Detailed Analysis
- Use more paths (5000-10000)
- Increase time steps
- Run on powerful computer (patience required!)

## Troubleshooting Checklist

- [ ] Python 3.8+ installed? → `python --version`
- [ ] Dependencies installed? → `pip list | grep streamlit`
- [ ] CSV file format correct? → Check headers
- [ ] Enough data points? → At least 2 rows
- [ ] Port 8501 available? → Check no other app uses it
- [ ] Internet connection? → Required for Plotly charts

## Getting Help

### Streamlit Documentation
https://docs.streamlit.io/

### GBM Mathematics
See README.md for references and mathematical explanation

### Debugging
Run with verbose output:
```bash
streamlit run app.py --logger.level=debug
```

## Next Steps

1. **Explore Different Scenarios**
   - Test bull/bear cases with different drift values
   - Compare high/low volatility outcomes

2. **Analyze Your Data**
   - Try different time windows
   - Compare different assets

3. **Export & Post-Process**
   - Download simulations as CSV
   - Use in Excel/Python for further analysis

4. **Customize**
   - Edit `app.py` to add more features
   - Create new modules in `utils/`
   - Modify color schemes and layouts

## Customization Ideas

1. **Add more visualization options**
   - 3D surface plots
   - Heatmaps of path correlations
   - Movie/animation of paths

2. **Expand simulation models**
   - Jump-Diffusion models
   - Heston Stochastic Volatility
   - GARCH models

3. **Add trading strategies**
   - Monte Carlo VaR calculations
   - Options pricing (Black-Scholes)
   - Portfolio optimization

4. **Enhance UI**
   - Dark mode toggle
   - Custom color schemes
   - Advanced filters

## System Requirements

- **OS**: macOS, Linux, or Windows
- **Python**: 3.8 or later
- **RAM**: 2GB minimum (4GB+ recommended)
- **Storage**: ~200MB for dependencies
- **Internet**: Required for Plotly rendering

## License & Disclaimer

This educational tool is provided as-is. 

**Important**: Stock market simulations are mathematical models and not predictions. Do not use as sole basis for investment decisions. Past performance does not guarantee future results.

---

**Happy simulating! 📊🚀**

For more information, see `README.md`
