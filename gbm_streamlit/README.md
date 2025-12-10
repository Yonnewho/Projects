# 📈 GBM Stock Market Simulator

A comprehensive Streamlit application that simulates stock price movements using **Geometric Brownian Motion (GBM)**, allowing you to compare real historical stock data with synthetic simulations.

## ✨ Features

- **📊 Real Data Upload**: Import your own stock data via CSV file
- **🎲 GBM Simulation**: Generate synthetic stock price paths
- **🎚️ Interactive Controls**: Adjust drift, volatility, number of simulations, and time horizon
- **📈 Interactive Charts**: Visualize real vs simulated prices with Plotly
- **📉 Statistical Analysis**: 
  - Summary statistics (mean, std, min, max, percentiles)
  - Distribution of final prices
  - Return metrics (daily/annualized volatility, Sharpe ratio)
  - Confidence intervals (5th-95th percentiles)
- **💾 Export Data**: Download simulated paths as CSV or Excel
- **🔍 Distribution Analysis**: Compare real returns with simulated outcomes

## 🛠️ Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## 📦 Installation

### 1. Clone or Setup the Project

```bash
cd /Users/daniel/Desktop/School/gbm_streamlit
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### To stop the server
Press `Ctrl+C` in the terminal where you ran the command.

## 📁 Project Structure

```
gbm_streamlit/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── GOOGL_2004-08-01_...csv    # Sample stock data
└── utils/
    ├── __init__.py
    ├── data_loader.py              # Data loading and validation
    ├── gbm_simulator.py            # GBM simulation engine
    ├── plotting.py                 # Plotly visualization functions
    └── statistics.py               # Statistical calculations
```

## 📚 How to Use

### Step 1: Load Stock Data

In the sidebar, choose to:
- **Upload CSV**: Select your own stock price CSV file (must have 'Date' and 'Close' columns)
- **Sample Data**: Use automatically generated synthetic data

Example CSV format:
```csv
Date,Close
2023-01-01,150.25
2023-01-02,151.50
2023-01-03,149.75
...
```

### Step 2: Configure GBM Parameters

The app automatically estimates parameters from your data:
- **Drift (μ)**: Expected annual return (adjust using slider)
- **Volatility (σ)**: Annual standard deviation (adjust using slider)

### Step 3: Set Simulation Parameters

- **Number of Days**: How far into the future to simulate (5-365 days)
- **Number of Paths**: How many simulations to run (10-10,000)
- **Time Steps**: Resolution of the simulation (higher = finer detail)

### Step 4: View Results

The app displays:
1. **Main Chart**: Real data vs all simulated paths with:
   - Black line: actual historical prices
   - Blue transparent lines: individual simulated paths
   - Green dashed line: mean of all simulations
   - Shaded area: 5th-95th percentile confidence interval

2. **Summary Statistics**: Comparison of real vs simulated data
3. **Distribution Analysis**: Histogram of final prices
4. **Returns Analysis**: Statistical comparison and probability metrics

### Step 5: Export Results

Click "Prepare Export File" to generate a downloadable file containing all simulated paths in either CSV or Excel format.

## 🧮 Geometric Brownian Motion Explained

### The Model

The GBM model describes stock price evolution as:

$$dS = \mu S \, dt + \sigma S \, dW$$

Where:
- **dS** = infinitesimal change in stock price
- **μ (Drift)** = expected annual return
- **σ (Volatility)** = annual standard deviation of returns
- **dt** = infinitesimal time increment
- **dW** = Wiener process (random Brownian motion)

### Discrete Implementation

For simulation, we use the discrete formula:

$$S_{t+\Delta t} = S_t \times \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} \, Z\right)$$

Where:
- $Z$ ~ Normal(0, 1) is a random normal variable
- The term $-\frac{\sigma^2}{2}$ ensures the model is log-normal
- Each simulation path generates a random trajectory

### Why GBM?

1. **No negative prices**: Stock prices can't go below zero
2. **Log-normal returns**: Matches empirical observation
3. **Analytical tractability**: Widely used in financial models (Black-Scholes, options pricing)
4. **Realistic variance**: Volatility is proportional to price level

## 📊 Understanding the Output

### Summary Statistics
- **Mean/Std Dev**: Price level across all simulations
- **Min/Max**: Range of possible prices
- **Percentiles**: Confidence bounds on outcomes

### Final Price Distribution
- Shows where prices are likely to end up after N days
- Mean, 5th, and 95th percentile marked
- Wider distribution = higher uncertainty

### Returns Analysis
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Annualized Vol**: How much price fluctuates yearly
- **Probability Above Real**: % of simulations ending above actual final price

## ⚙️ Advanced Tips

### Adjusting Drift
- **Higher drift**: Optimistic scenario (growing market)
- **Lower drift**: Pessimistic scenario (declining market)
- **Zero drift**: Random walk (no directional bias)

### Adjusting Volatility
- **Higher volatility**: More extreme outcomes, wider confidence bands
- **Lower volatility**: More stable, tighter bands
- Use estimated value for realistic scenarios

### Number of Paths
- **Fewer paths** (100-500): Faster, noisier
- **More paths** (5,000+): Accurate, slower
- Trade-off between accuracy and speed

## 🐛 Troubleshooting

### "CSV must contain 'Date' and 'Close' columns"
- Check your CSV file has headers named exactly "Date" and "Close" (case-insensitive)
- Ensure they're in the file

### "Need at least 2 data points"
- Your CSV has too few rows. Add more historical data.

### Simulations take too long
- Reduce number of paths or time steps
- Use fewer days to simulate

### Charts not displaying
- Ensure you have an internet connection (Plotly requires it)
- Try refreshing the page

## 📈 Example Use Cases

1. **Portfolio Planning**: Simulate future prices to estimate risk
2. **Option Pricing**: Generate price distributions for option valuation
3. **Scenario Analysis**: Compare bull/bear cases by adjusting drift
4. **Education**: Learn how randomness affects price trajectories
5. **Backtesting**: Generate synthetic data for testing trading strategies

## 📖 References

- Black, F., & Scholes, M. (1973). "The pricing of options and corporate liabilities". Journal of Political Economy
- Hull, J. C. (2018). "Options, Futures, and Other Derivatives" (10th edition)
- Wilmott, P. (2019). "Machine Learning in Finance"

## ⚖️ Disclaimer

**This application is for educational and research purposes only.**

- Simulations are mathematical models, not predictions
- Past performance does not guarantee future results
- Market behavior is influenced by countless unpredictable factors
- Do not use these simulations as the sole basis for investment decisions
- Consult a financial advisor before making investment decisions

## 🤝 Contributing

Feel free to modify and extend this application:
- Add more distribution models (Heston, Jump-Diffusion, etc.)
- Implement scenario analysis features
- Add more statistical metrics
- Improve visualization options

## 📝 License

This project is provided as-is for educational purposes.

---

**Enjoy exploring stock market simulations! 📊**
