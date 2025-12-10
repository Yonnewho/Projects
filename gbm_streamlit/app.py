"""
Main Streamlit application for GBM Stock Market Simulator.

This app allows users to:
1. Upload real stock data (CSV)
2. Simulate stock prices using Geometric Brownian Motion
3. Compare real data with simulated paths
4. Export simulated data to CSV
5. View summary statistics and distributions
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
from datetime import datetime, timedelta

# Import utility modules
from utils.data_loader import load_csv_data, load_builtin_data, estimate_parameters, validate_data, calculate_returns
from utils.gbm_simulator import simulate_gbm, get_path_statistics, create_simulated_dataframe
from utils.plotting import plot_real_vs_simulated, plot_distribution, plot_real_returns_distribution
from utils.statistics import (
    calculate_summary_stats,
    calculate_returns_stats,
    compare_distributions,
    create_summary_dataframe
)
from utils.lstm_trainer import train_lstm_model, predict_future_prices, get_last_lookback_sequence, evaluate_predictions


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title='GBM Stock Simulator',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; }
    h2 { color: #1f77b4; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for LSTM model
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'lstm_metrics' not in st.session_state:
    st.session_state.lstm_metrics = None
if 'lstm_scaler' not in st.session_state:
    st.session_state.lstm_scaler = None


# ============================================================================
# SIDEBAR: INPUT CONTROLS
# ============================================================================
st.sidebar.title('📊 GBM Simulator Controls')

# Data source selection
st.sidebar.subheader('1. Load Stock Data')
data_source = st.sidebar.radio(
    'Data Source:',
    ['Upload CSV', 'Sample Data', 'Built-in Data'],
    help='Choose to upload your own data, use sample data, or load Google stock data'
)

real_data = None
error_msg = None

if data_source == 'Upload CSV':
    uploaded_file = st.sidebar.file_uploader(
        'Upload CSV file',
        type=['csv'],
        help='CSV must have "Date" and "Close" columns'
    )
    
    if uploaded_file is not None:
        real_data, error_msg = load_csv_data(uploaded_file)
        if error_msg:
            st.sidebar.error(error_msg)
        else:
            st.sidebar.success(f'✓ Loaded {len(real_data)} data points')

elif data_source == 'Built-in Data':
    builtin_files = [
        'GOOG_2004-08-19_2025-08-20.csv',
        'GOOGL_2004-08-01_2024-12-18.csv'
    ]
    selected_file = st.sidebar.selectbox(
        'Select built-in dataset:',
        builtin_files,
        help='Use pre-loaded Google stock data'
    )
    
    if selected_file:
        real_data, error_msg = load_builtin_data(selected_file)
        if error_msg:
            st.sidebar.error(error_msg)
        else:
            st.sidebar.success(f'✓ Loaded {len(real_data)} data points from {selected_file}')

else:
    # Generate sample data (synthetic Google-like data)
    np.random.seed(42)
    n_days = 252  # 1 year
    start_price = 100
    returns = np.random.normal(0.0003, 0.015, n_days)
    prices = start_price * np.exp(np.cumsum(returns))
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    real_data = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    st.sidebar.info('📌 Using sample synthetic data (Google-like)')


# Validate data
if real_data is not None:
    is_valid, val_error = validate_data(real_data)
    if not is_valid:
        st.sidebar.error(val_error)
        st.stop()


st.sidebar.markdown('---')

# GBM Parameters
st.sidebar.subheader('2. GBM Parameters')

if real_data is not None:
    # Estimate parameters from real data
    estimated_drift, estimated_volatility = estimate_parameters(real_data['close'].values)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric('Estimated Drift', f'{estimated_drift:.2%}')
    with col2:
        st.metric('Estimated Vol', f'{estimated_volatility:.2%}')
    
    # User inputs
    drift = st.sidebar.slider(
        'Drift (μ)',
        min_value=-0.5,
        max_value=0.5,
        value=estimated_drift,
        step=0.01,
        help='Expected annual return. Adjust to test different scenarios.'
    )
    
    volatility = st.sidebar.slider(
        'Volatility (σ)',
        min_value=0.01,
        max_value=1.0,
        value=estimated_volatility,
        step=0.01,
        help='Annual standard deviation of returns.'
    )

st.sidebar.markdown('---')

# Simulation parameters
st.sidebar.subheader('3. Simulation Settings')

if real_data is not None:
    num_days = st.sidebar.slider(
        'Number of Days to Simulate',
        min_value=5,
        max_value=365,
        value=60,
        step=5,
        help='How far into the future to simulate'
    )
    
    num_paths = st.sidebar.slider(
        'Number of Simulated Paths',
        min_value=10,
        max_value=10000,
        value=1000,
        step=50,
        help='More paths = more accurate but slower'
    )
    
    num_steps = st.sidebar.slider(
        'Time Steps per Path',
        min_value=10,
        max_value=num_days,
        value=num_days,
        step=5,
        help='Higher resolution = finer simulation'
    )

st.sidebar.markdown('---')

# Model selection
st.sidebar.subheader('4. Model Selection')
model_type = st.sidebar.radio(
    'Prediction Model:',
    ['GBM Simulation', 'LSTM Neural Network', 'Both (Compare)'],
    help='Choose between classical GBM or deep learning LSTM'
)

# LSTM-specific parameters
lstm_enabled = model_type in ['LSTM Neural Network', 'Both (Compare)']
if lstm_enabled:
    st.sidebar.subheader('   LSTM Settings')
    lstm_lookback = st.sidebar.slider(
        '   Lookback Window',
        min_value=10,
        max_value=120,
        value=60,
        step=5,
        help='Number of days to look back for LSTM training'
    )
    
    lstm_epochs = st.sidebar.slider(
        '   Training Epochs',
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help='More epochs = longer training but potentially better accuracy'
    )
    
    lstm_train_button = st.sidebar.button('🚀 Train LSTM Model', key='train_lstm')

st.sidebar.markdown('---')

# Export button
st.sidebar.subheader('5. Export Results')
export_format = st.sidebar.selectbox('Export Format', ['CSV', 'Excel'])


# ============================================================================
# MAIN CONTENT: SIMULATION & RESULTS
# ============================================================================

if real_data is None:
    st.info('👈 Please load stock data using the sidebar controls.')
    st.stop()


st.title('📈 Geometric Brownian Motion Stock Simulator')

# Create two columns for layout
col_chart, col_info = st.columns([3, 1])

with col_info:
    st.subheader('Data Summary')
    stats_box = f"""
    **Real Data:**
    - Data Points: {len(real_data)}
    - Date Range: {real_data['date'].min().date()} to {real_data['date'].max().date()}
    - Current Price: ${real_data['close'].iloc[-1]:.2f}
    
    **Simulation:**
    - Drift (μ): {drift:.2%}
    - Volatility (σ): {volatility:.2%}
    - Paths: {num_paths:,}
    - Days: {num_days}
    """
    st.info(stats_box)


# Run simulation
with col_chart:
    st.subheader('Real vs Simulated Prices')
    
    # GBM simulation
    if model_type in ['GBM Simulation', 'Both (Compare)']:
        with st.spinner('Running GBM simulation...'):
            # Get initial price and date
            S0 = real_data['close'].iloc[-1]
            initial_date = real_data['date'].iloc[-1]
            
            # Run simulation
            times, paths = simulate_gbm(
                S0=S0,
                mu=drift,
                sigma=volatility,
                T=num_days,
                N=num_steps,
                M=num_paths
            )
            
            # Get statistics
            stats = get_path_statistics(paths)
        
        # Plot results - get both real and simulation charts
        fig_real, fig_sim = plot_real_vs_simulated(real_data, times, paths, stats)
        
        # Display both charts
        st.plotly_chart(fig_real, use_container_width=True)
        st.plotly_chart(fig_sim, use_container_width=True)
    
    # LSTM prediction
    elif model_type == 'LSTM Neural Network':
        if st.session_state.lstm_model is not None:
            with st.spinner('Running LSTM predictions...'):
                # Get last sequence and predict
                last_seq, lstm_scaler = get_last_lookback_sequence(
                    real_data['close'].values, 
                    lookback_window=lstm_lookback
                )
                
                lstm_predictions = predict_future_prices(
                    st.session_state.lstm_model,
                    last_seq,
                    st.session_state.lstm_scaler,
                    num_days,
                    lookback_window=lstm_lookback
                )
            
            # Plot LSTM predictions
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig_lstm = go.Figure()
            
            # Historical data
            fig_lstm.add_trace(go.Scatter(
                x=real_data['date'],
                y=real_data['close'],
                name='Historical Data',
                mode='lines',
                line=dict(color='blue', width=2)
            ))
            
            # Future predictions
            last_date = real_data['date'].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=num_days+1, freq='B')[1:]
            
            fig_lstm.add_trace(go.Scatter(
                x=future_dates[:len(lstm_predictions)],
                y=lstm_predictions,
                name='LSTM Prediction',
                mode='lines',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            fig_lstm.update_layout(
                title='Stock Price: Historical Data vs LSTM Prediction',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_lstm, use_container_width=True)
        else:
            st.info('👈 Please train the LSTM model first using the sidebar button.')
            st.stop()


# ============================================================================
# LSTM TRAINING (if enabled)
# ============================================================================

if lstm_enabled and lstm_train_button:
    training_container = st.container()
    
    with training_container:
        with st.spinner('🔄 Training LSTM model... This may take 1-2 minutes. Please wait...'):
            try:
                # Train LSTM
                lstm_model, lstm_scaler, history, metrics = train_lstm_model(
                    prices=real_data['close'].values,
                    lookback_window=lstm_lookback,
                    epochs=lstm_epochs,
                    batch_size=32,
                    layers=[64, 32],
                    dropout_rate=0.2,
                    verbose=0
                )
                
                # Store in session state
                st.session_state.lstm_model = lstm_model
                st.session_state.lstm_metrics = metrics
                st.session_state.lstm_scaler = lstm_scaler
            
            except Exception as e:
                st.error(f'❌ Error training LSTM: {str(e)}')
                st.stop()
        
        # Display training results after training completes
        st.success('✓ LSTM model trained successfully!')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Train RMSE', f"${metrics['train_rmse']:.2f}")
        with col2:
            st.metric('Test RMSE', f"${metrics['test_rmse']:.2f}")
        with col3:
            st.metric('Test MAE', f"${metrics['test_mae']:.2f}")
        with col4:
            st.metric('Test R²', f"{metrics['test_r2']:.4f}")
        
        st.info(f'📊 Model trained on {metrics["n_train"]} training samples and {metrics["n_test"]} test samples')


# ============================================================================
# STATISTICS & ANALYSIS SECTION
# ============================================================================

st.markdown('---')
st.subheader('📊 Statistical Analysis')

# Prepare data for both models
gbm_data = None
lstm_data = None

# GBM Setup
if model_type in ['GBM Simulation', 'Both (Compare)']:
    S0 = real_data['close'].iloc[-1]
    initial_date = real_data['date'].iloc[-1]
    
    times, paths = simulate_gbm(
        S0=S0,
        mu=drift,
        sigma=volatility,
        T=num_days,
        N=num_paths,
        M=num_steps
    )
    
    gbm_data = {
        'times': times,
        'paths': paths,
        'stats': get_path_statistics(paths)
    }

# LSTM Setup
if model_type in ['LSTM Neural Network', 'Both (Compare)'] and st.session_state.lstm_model is not None:
    last_seq, _ = get_last_lookback_sequence(
        real_data['close'].values,
        lookback_window=lstm_lookback
    )
    
    lstm_predictions = predict_future_prices(
        st.session_state.lstm_model,
        last_seq,
        st.session_state.lstm_scaler,
        num_days,
        lookback_window=lstm_lookback
    )
    
    lstm_data = {
        'predictions': lstm_predictions
    }

# Create tabs for different analyses
if model_type == 'Both (Compare)':
    tab1, tab2, tab3, tab4 = st.tabs(['Summary Statistics', 'Distribution Analysis', 'Returns Analysis', 'Model Comparison'])
else:
    tab1, tab2, tab3 = st.tabs(['Summary Statistics', 'Distribution Analysis', 'Returns Analysis'])

with tab1:
    col1, col2 = st.columns(2)
    
    # Calculate statistics
    real_prices = real_data['close'].values
    real_stats = calculate_summary_stats(real_prices)
    
    if gbm_data is not None:
        # Get simulated final prices
        sim_final_prices = gbm_data['paths'][:, -1]
        sim_prices_all = gbm_data['paths'].flatten()
        sim_stats = calculate_summary_stats(sim_prices_all)
        
        # Create comparison table
        comparison_df = create_summary_dataframe(real_stats, sim_stats)
        
        with col1:
            st.write('**Price Statistics (Real vs GBM)**')
            st.dataframe(
                comparison_df.style.format({
                    'Real Data': '${:.2f}',
                    'Simulated Data': '${:.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            st.write('**Final Price Statistics (GBM)**')
            final_stats = {
                'Mean': f"${np.mean(sim_final_prices):.2f}",
                'Std Dev': f"${np.std(sim_final_prices):.2f}",
                'Min': f"${np.min(sim_final_prices):.2f}",
                'Max': f"${np.max(sim_final_prices):.2f}",
                'Median': f"${np.median(sim_final_prices):.2f}",
                '5th %ile': f"${np.percentile(sim_final_prices, 5):.2f}",
                '95th %ile': f"${np.percentile(sim_final_prices, 95):.2f}",
            }
            st.dataframe(pd.DataFrame(final_stats.items(), columns=['Statistic', 'Value']))
    
    elif lstm_data is not None:
        # LSTM stats
        with col1:
            st.write('**Real Data Statistics**')
            real_display = {
                'Mean': f"${np.mean(real_prices):.2f}",
                'Std Dev': f"${np.std(real_prices):.2f}",
                'Min': f"${np.min(real_prices):.2f}",
                'Max': f"${np.max(real_prices):.2f}",
                'Median': f"${np.median(real_prices):.2f}",
            }
            st.dataframe(pd.DataFrame(real_display.items(), columns=['Metric', 'Value']))
        
        with col2:
            st.write('**LSTM Prediction Stats**')
            lstm_preds = lstm_data['predictions']
            lstm_display = {
                'Mean': f"${np.mean(lstm_preds):.2f}",
                'Std Dev': f"${np.std(lstm_preds):.2f}",
                'Min': f"${np.min(lstm_preds):.2f}",
                'Max': f"${np.max(lstm_preds):.2f}",
                'Median': f"${np.median(lstm_preds):.2f}",
            }
            st.dataframe(pd.DataFrame(lstm_display.items(), columns=['Metric', 'Value']))

with tab2:
    col1, col2 = st.columns(2)
    
    if gbm_data is not None:
        sim_final_prices = gbm_data['paths'][:, -1]
        
        with col1:
            st.write('**Final Price Distribution (GBM)**')
            fig_dist = plot_distribution(sim_final_prices)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.write('**Real Daily Returns Distribution**')
            if len(real_data) > 1:
                fig_returns = plot_real_returns_distribution(real_data)
                st.plotly_chart(fig_returns, use_container_width=True)
            else:
                st.warning('Not enough data points for returns analysis.')
    
    elif lstm_data is not None:
        import plotly.graph_objects as go
        
        lstm_preds = lstm_data['predictions']
        
        with col1:
            st.write('**LSTM Prediction Distribution**')
            fig_lstm_dist = go.Figure(data=[
                go.Histogram(x=lstm_preds, nbinsx=30, name='LSTM Predictions')
            ])
            fig_lstm_dist.update_layout(
                title='Distribution of LSTM Predictions',
                xaxis_title='Price ($)',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig_lstm_dist, use_container_width=True)
        
        with col2:
            st.write('**Real Daily Returns Distribution**')
            if len(real_data) > 1:
                fig_returns = plot_real_returns_distribution(real_data)
                st.plotly_chart(fig_returns, use_container_width=True)
            else:
                st.warning('Not enough data points for returns analysis.')

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**Real Data Return Metrics**')
        real_returns_stats = calculate_returns_stats(real_data['close'].values)
        
        returns_display = {
            'Daily Mean Return': f"{real_returns_stats['daily_mean']:.4%}",
            'Daily Volatility': f"{real_returns_stats['daily_std']:.4%}",
            'Total Return': f"{real_returns_stats['total_return']:.2%}",
            'Annualized Return': f"{real_returns_stats['annualized_return']:.2%}",
            'Annualized Vol': f"{real_returns_stats['annualized_volatility']:.2%}",
            'Sharpe Ratio': f"{real_returns_stats['sharpe_ratio']:.4f}",
        }
        st.dataframe(pd.DataFrame(returns_display.items(), columns=['Metric', 'Value']))
    
    with col2:
        if gbm_data is not None:
            st.write('**GBM Simulation vs Reality**')
            sim_final_prices = gbm_data['paths'][:, -1]
            real_prices = real_data['close'].values
            comparison = compare_distributions(real_prices, sim_final_prices)
            
            comparison_display = {
                'Real Final Price': f"${comparison['real_final_price']:.2f}",
                'Sim Mean Final Price': f"${comparison['simulated_mean_final_price']:.2f}",
                'Sim Std Dev': f"${comparison['simulated_std_final_price']:.2f}",
                'Prob Above Real': f"{comparison['probability_above_real']:.2%}",
                'Mean Difference': f"${comparison['mean_difference']:.2f}",
                'Diff %': f"{comparison['mean_difference_pct']:.2f}%",
            }
            st.dataframe(pd.DataFrame(comparison_display.items(), columns=['Metric', 'Value']))

# Add comparison tab when both models are used
if model_type == 'Both (Compare)':
    with tab4:
        st.subheader('Model Comparison: GBM vs LSTM')
        
        if gbm_data is not None and lstm_data is not None:
            import plotly.graph_objects as go
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('**Prediction Comparison**')
                
                fig_compare = go.Figure()
                
                # Historical data
                fig_compare.add_trace(go.Scatter(
                    x=real_data['date'],
                    y=real_data['close'],
                    name='Historical Data',
                    mode='lines',
                    line=dict(color='blue', width=2)
                ))
                
                # GBM mean and confidence intervals
                gbm_paths = gbm_data['paths']
                gbm_mean = np.mean(gbm_paths, axis=0)
                gbm_std = np.std(gbm_paths, axis=0)
                
                times = gbm_data['times']
                initial_date = real_data['date'].iloc[-1]
                future_dates = pd.date_range(start=initial_date, periods=len(times)+1, freq='B')[1:]
                
                fig_compare.add_trace(go.Scatter(
                    x=future_dates[:len(gbm_mean)],
                    y=gbm_mean,
                    name='GBM Mean',
                    mode='lines',
                    line=dict(color='green', width=2)
                ))
                
                fig_compare.add_trace(go.Scatter(
                    x=future_dates[:len(gbm_mean)],
                    y=gbm_mean + gbm_std,
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig_compare.add_trace(go.Scatter(
                    x=future_dates[:len(gbm_mean)],
                    y=gbm_mean - gbm_std,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='GBM ±1σ'
                ))
                
                # LSTM predictions
                lstm_preds = lstm_data['predictions']
                fig_compare.add_trace(go.Scatter(
                    x=future_dates[:len(lstm_preds)],
                    y=lstm_preds,
                    name='LSTM Prediction',
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_compare.update_layout(
                    title='GBM vs LSTM: Prediction Comparison',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
            
            with col2:
                st.write('**Model Metrics**')
                
                metrics_comp = {
                    'Model': ['GBM', 'LSTM'],
                    'Mean Final': [
                        f"${np.mean(gbm_data['paths'][:, -1]):.2f}",
                        f"${np.mean(lstm_data['predictions']):.2f}"
                    ],
                    'Std Dev': [
                        f"${np.std(gbm_data['paths'][:, -1]):.2f}",
                        f"${np.std(lstm_data['predictions']):.2f}"
                    ],
                    'Type': ['Parametric', 'Neural Network']
                }
                
                st.dataframe(pd.DataFrame(metrics_comp))
                
                st.markdown('**Summary:**')
                st.write("""
                - **GBM**: Classical financial model, provides probabilistic paths based on drift/volatility
                - **LSTM**: Deep learning model, learns from historical patterns
                
                Use GBM for traditional quantitative finance analysis.
                Use LSTM to capture complex, non-linear market behaviors.
                """)
        
        elif lstm_data is not None:
            st.write('**LSTM Model Metrics**')
            if st.session_state.lstm_metrics:
                metrics_display = {
                    'Train RMSE': f"${st.session_state.lstm_metrics['train_rmse']:.2f}",
                    'Test RMSE': f"${st.session_state.lstm_metrics['test_rmse']:.2f}",
                    'Test MAE': f"${st.session_state.lstm_metrics['test_mae']:.2f}",
                    'Test R²': f"{st.session_state.lstm_metrics['test_r2']:.4f}",
                    'Training Samples': f"{st.session_state.lstm_metrics['n_train']:,}",
                    'Test Samples': f"{st.session_state.lstm_metrics['n_test']}",
                }
                st.dataframe(pd.DataFrame(metrics_display.items(), columns=['Metric', 'Value']))
            else:
                st.info('Model not trained yet')


# ============================================================================
# EXPORT SECTION
# ============================================================================

st.markdown('---')
st.subheader('💾 Export Simulations/Predictions')

col1, col2 = st.columns([2, 1])

with col1:
    if st.button('📥 Prepare Export File', key='export_btn'):
        if gbm_data is not None:
            # GBM export
            sim_df = create_simulated_dataframe(gbm_data['times'], gbm_data['paths'], real_data['date'].iloc[-1])
            
            if export_format == 'CSV':
                csv_buffer = io.StringIO()
                sim_df.to_csv(csv_buffer)
                csv_content = csv_buffer.getvalue()
                
                st.download_button(
                    label='Download CSV (GBM)',
                    data=csv_content,
                    file_name=f'gbm_simulations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key='download_csv'
                )
            else:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    sim_df.to_excel(writer, sheet_name='GBM Simulations')
                excel_content = excel_buffer.getvalue()
                
                st.download_button(
                    label='Download Excel (GBM)',
                    data=excel_content,
                    file_name=f'gbm_simulations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_excel'
                )
            
            st.success(f'✓ Export ready! ({sim_df.shape[0]} rows × {sim_df.shape[1]} columns)')
        
        elif lstm_data is not None:
            # LSTM export
            future_dates = pd.date_range(start=real_data['date'].iloc[-1], periods=len(lstm_data['predictions'])+1, freq='B')[1:]
            lstm_df = pd.DataFrame({
                'date': future_dates[:len(lstm_data['predictions'])],
                'lstm_prediction': lstm_data['predictions']
            })
            
            if export_format == 'CSV':
                csv_buffer = io.StringIO()
                lstm_df.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                
                st.download_button(
                    label='Download CSV (LSTM)',
                    data=csv_content,
                    file_name=f'lstm_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key='download_csv_lstm'
                )
            else:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    lstm_df.to_excel(writer, sheet_name='LSTM Predictions', index=False)
                excel_content = excel_buffer.getvalue()
                
                st.download_button(
                    label='Download Excel (LSTM)',
                    data=excel_content,
                    file_name=f'lstm_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_excel_lstm'
                )
            
            st.success(f'✓ Export ready! ({lstm_df.shape[0]} rows × {lstm_df.shape[1]} columns)')

with col2:
    if gbm_data is not None:
        st.metric('Export Size', f'{len(gbm_data["paths"]):,} paths × {len(gbm_data["times"])} steps')
    elif lstm_data is not None:
        st.metric('Export Size', f'{len(lstm_data["predictions"])} days of predictions')


# ============================================================================
# FOOTER & INFO
# ============================================================================

st.markdown('---')
st.markdown("""
### 📖 About This App

This application simulates stock price movements using **Geometric Brownian Motion (GBM)**, a widely-used 
model in quantitative finance.

**The GBM Model:**
$$dS = \\mu S \\, dt + \\sigma S \\, dW$$

Where:
- $S$ = Stock price
- $\\mu$ = Drift (expected return)
- $\\sigma$ = Volatility
- $dW$ = Wiener process increment

**Disclaimer:** These simulations are for educational purposes only. Actual market behavior is influenced 
by countless factors and cannot be perfectly predicted. Past performance does not guarantee future results.
""")
