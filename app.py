import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Page config - FIXED: Removed invalid 'height' parameter
st.set_page_config(
    page_title="Oil Strip PFE Simulator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Oil Strip PFE Exposure Simulator")
st.markdown("**Jump-Diffusion Monte Carlo for Oil Swap Strips**")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Simulation Parameters")

# 1. Swap months (1-24, starts next month)
months = st.sidebar.slider("Swap Months", 1, 24, 12, help="1-24 monthly swaps")

# 2. Monthly volumes input
st.sidebar.subheader("📈 Monthly Volumes (MT)")
volumes = []
default_vol = 12000.0
for i in range(months):
    vol = st.sidebar.number_input(
        f"M{i+1}", 
        min_value=0.0, max_value=50000.0, 
        value=default_vol, 
        key=f"vol_{i}",
        help="Metric tons per month"
    )
    volumes.append(vol)

# Core parameters
col1, col2 = st.sidebar.columns(2)
volatility = col1.slider("Volatility", 0.10, 1.00, 0.30, 0.01)
spot_price = col2.number_input("Spot ($/bbl)", 50.0, 150.0, 80.0)

fixed_price = st.sidebar.number_input("Fixed Price ($/bbl)", 50.0, 150.0, 80.0)

# Jump-Diffusion with defaults
st.sidebar.subheader("⚡ Jump-Diffusion")
use_defaults = st.sidebar.checkbox("Use Defaults", value=True)
if use_defaults:
    lambda_jump, mean_jump, std_jump = 0.8, 0.0, 0.20
else:
    lambda_jump = st.sidebar.number_input("Lambda", 0.0, 5.0, 0.8, 0.01)
    mean_jump = st.sidebar.number_input("Mean Jump", -1.0, 1.0, 0.0, 0.01)
    std_jump = st.sidebar.number_input("Jump Std", 0.0, 1.0, 0.20, 0.01)

# Other parameters
drift_rate = st.sidebar.number_input("Drift", 0.0, 0.10, 0.03, 0.001)
risk_free = st.sidebar.number_input("Risk Free", 0.0, 0.10, 0.02, 0.001)
num_paths = st.sidebar.slider("MC Paths", 500, 3000, 1500, 250)

# Constants
FIXED_PRICES = np.full(months, fixed_price)
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT

# Time setup
BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1 / BUSINESS_DAYS_PER_YEAR
STRIP_LENGTH_YEARS = months / 12.0
SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)

# Simulation function
@st.cache_data
def run_simulation(_months, _notionals, _fixed_prices, _s0, _vol, _drift, _rf, 
                  _lambda, _mu_j, _sigma_j, _paths):
    
    # Generate paths
    M = len(time_points) - 1
    S = np.full((M + 1, _paths), _s0)
    
    jump_mean_e = np.exp(_mu_j + 0.5 * _sigma_j**2) - 1
    adj_mu = _drift - _lambda * jump_mean_e
    
    for t in range(M):
        Z = np.random.normal(size=_paths)
        N_jumps = np.random.poisson(_lambda * DAILY_TIMESTEP, size=_paths)
        max_j = max(1, N_jumps.max())
        jumps = np.random.normal(_mu_j, _sigma_j, (max_j, _paths))
        
        jump_impact = np.zeros(_paths)
        for i in range(_paths):
            if N_jumps[i] > 0:
                jump_impact[i] = jumps[:N_jumps[i], i].sum()
        
        S[t+1] = S[t] * np.exp(
            (adj_mu - 0.5 * _vol**2) * DAILY_TIMESTEP +
            _vol * np.sqrt(DAILY_TIMESTEP) * Z +
            jump_impact
        )
    
    # Exposure calculation
    PV_paths = np.zeros((SIMULATION_STEPS + 1, _paths))
    
    for t in range(SIMULATION_STEPS + 1):
        mtm = np.zeros(_paths)
        
        for i in range(_months):
            tenor_start = int(i * DAYS_PER_MONTH)
            tenor_end = int((i + 1) * DAYS_PER_MONTH)
            settle = tenor_end + 1
            
            if t >= settle:
                continue
            
            if tenor_start <= t < tenor_end:
                # Averaging period
                days_passed = t - tenor_start + 1
                days_remain = DAYS_PER_MONTH - days_passed
                p_real = np.mean(S[tenor_start:t+1], axis=0)
                p_future = S[t] * np.exp(_drift * days_remain * DAILY_TIMESTEP) if days_remain > 0 else 0
                p_final = (p_real * days_passed + p_future * days_remain) / DAYS_PER_MONTH
            else:
                # Forward
                t_to_tenor = time_points[tenor_start] - time_points[t]
                p_final = S[t] * np.exp(_drift * t_to_tenor)
            
            t_to_settle = time_points[settle] - time_points[t]
            df = np.exp(-_rf * t_to_settle)
            mtm += (p_final - _fixed_prices[i]) * _notionals[i] * df
        
        PV_paths[t] = mtm
    
    # Metrics
    exposure = np.maximum(0, PV_paths)
    pfe_95 = np.percentile(exposure, 95, axis=1)
    pfe_99 = np.percentile(exposure, 99, axis=1)
    ee = np.mean(exposure, axis=1)
    neg_95 = np.percentile(PV_paths, 5, axis=1)
    neg_99 = np.percentile(PV_paths, 1, axis=1)
    
    return {
        'time': time_points,
        'pfe_99': pfe_99, 'pfe_95': pfe_95, 'ee': ee,
        'neg_99': neg_99, 'neg_95': neg_95,
        'samples': PV_paths[:, np.random.choice(_paths, 10, replace=False)]
    }

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running Jump-Diffusion MC..."):
        results = run_simulation(
            months, NOTIONAL_PER_MONTH, FIXED_PRICES, spot_price, volatility,
            drift_rate, risk_free, lambda_jump, mean_jump, std_jump, num_paths
        )
        st.session_state.results = results
        st.success("✅ Simulation complete!")

# Results display
if 'results' in st.session_state:
    results = st.session_state.results
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Max PFE 99%", f"${np.max(results['pfe_99'])/1e6:.1f}M")
    with col2: st.metric("Max PFE 95%", f"${np.max
