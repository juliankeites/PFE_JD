import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Oil Strip PFE Simulator", layout="wide", initial_sidebar_state="expanded")

st.title("🛢️ Oil Strip PFE Exposure Simulator")
st.markdown("**Jump-Diffusion Monte Carlo for Oil Swap Strips**")
st.markdown("---")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar
st.sidebar.header("📊 Simulation Parameters")

months = st.sidebar.slider("Swap Months", 1, 24, 12)

st.sidebar.subheader("📈 Monthly Volumes (MT)")
volumes = []
default_vol = 12000.0
for i in range(months):
    vol = st.sidebar.number_input(f"M{i+1}", min_value=0.0, max_value=50000.0, 
                                 value=default_vol, key=f"vol_{i}")
    volumes.append(vol)

col1, col2 = st.sidebar.columns(2)
volatility = col1.slider("Volatility", 0.10, 1.00, 0.30, 0.01)
spot_price = col2.number_input("Spot ($/bbl)", 50.0, 150.0, 80.0)

fixed_price = st.sidebar.number_input("Fixed Price ($/bbl)", 50.0, 150.0, 80.0)

st.sidebar.subheader("⚡ Jump-Diffusion")
use_defaults = st.sidebar.checkbox("Use Defaults", value=True)
if use_defaults:
    lambda_jump, mean_jump, std_jump = 0.8, 0.0, 0.20
else:
    lambda_jump = st.sidebar.number_input("Lambda", 0.0, 5.0, 0.8, 0.01)
    mean_jump = st.sidebar.number_input("Mean Jump", -1.0, 1.0, 0.0, 0.01)
    std_jump = st.sidebar.number_input("Jump Std", 0.0, 1.0, 0.20, 0.01)

drift_rate = st.sidebar.number_input("Drift", 0.0, 0.10, 0.03, 0.001)
risk_free = st.sidebar.number_input("Risk Free", 0.0, 0.10, 0.02, 0.001)
num_paths = st.sidebar.slider("MC Paths", 500, 3000, 1500, 250)

# Constants - FIXED: All calculations after inputs
FIXED_PRICES = np.full(months, fixed_price)
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT

BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1 / BUSINESS_DAYS_PER_YEAR
STRIP_LENGTH_YEARS = months / 12.0
SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)

# ✅ BUTTON-ONLY EXECUTION - No auto-run
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    st.session_state.results = None  # Clear previous results
    
    with st.spinner("Running Jump-Diffusion Monte Carlo..."):
        # FIXED: All parentheses properly closed
        def run_simulation(months, notionals, fixed_prices, s0, vol, drift, rf, lambda_j, mu_j, sigma_j, paths, time_pts):
            M = len(time_pts) - 1
            S = np.full((M + 1, paths), s0)
            
            jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            adj_mu = drift - lambda_j * jump_mean_e
            
            for t in range(M):
                Z = np.random.normal(size=paths)
                N_jumps = np.random.poisson(lambda_j * DAILY_TIMESTEP, size=paths)  # ✅ FIXED
                max_j = max(1, N_jumps.max())
                jumps = np.random.normal(mu_j, sigma_j, (max_j, paths))
                
                jump_impact = np.zeros(paths)
                for i in range(paths):
                    if N_jumps[i] > 0:
                        jump_impact[i] = jumps[:N_jumps[i], i].sum()
                
                S[t+1] = S[t] * np.exp(
                    (adj_mu - 0.5 * vol**2) * DAILY_TIMESTEP +
                    vol * np.sqrt(DAILY_TIMESTEP) * Z +
                    jump_impact
                )
            
            PV_paths = np.zeros((SIMULATION_STEPS + 1, paths))
            
            for t in range(SIMULATION_STEPS + 1):
                mtm = np.zeros(paths)
                
                for i in range(months):
                    tenor_start = int(i * DAYS_PER_MONTH)
                    tenor_end = int((i + 1) * DAYS_PER_MONTH)
                    settle = tenor_end + 1
                    
                    if t >= settle:
                        continue
                    
                    if tenor_start <= t < tenor_end:
                        days_passed = t - tenor_start + 1
                        days_remain = DAYS_PER_MONTH - days_passed
                        p_real = np.mean(S[tenor_start:t+1], axis=0)
                        p_future = S[t] * np.exp(drift * days_remain * DAILY_TIMESTEP) if days_remain > 0 else 0
                        p_final = (p_real * days_passed + p_future * days_remain) / DAYS_PER_MONTH
                    else:
                        t_to_tenor = time_pts[tenor_start] - time_pts[t]
                        p_final = S[t] * np.exp(drift * t_to_tenor)
                    
                    t_to_settle = time_pts[settle] - time_pts[t]
                    df = np.exp(-rf * t_to_settle)
                    mtm += (p_final - fixed_prices[i]) * notionals[i] * df
