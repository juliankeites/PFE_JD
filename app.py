import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt
import pandas as pd

# Page config
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
def run_simulation(_months, _notionals, _fixed_prices, _s0,
