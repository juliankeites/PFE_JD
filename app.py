import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(page_title="Oil Strip Exposure Simulator", layout="wide")
st.title("🛢️ Oil Strip PFE Exposure Simulator (Jump-Diffusion Model)")

# --- Sidebar Inputs ---
st.sidebar.header("📊 Simulation Parameters")

# 1. Number of swap months (1-24)
months = st.sidebar.slider("Swap Months", 1, 24, 12, help="Number of monthly swaps (1-24 months)")

# 2. Monthly volumes input
st.sidebar.subheader("📈 Monthly Volumes (MT)")
default_volume = 12000.0
volumes = []
for i in range(months):
    vol = st.sidebar.number_input(f"Month {i+1} Volume (MT)", 0.0, 50000.0, default_volume, key=f"vol_{i}")
    volumes.append(vol)

# 3. Volatility
volatility = st.sidebar.slider("Volatility (Annual)", 0.1, 1.0, 0.3, 0.01)

# 4. Jump-Diffusion parameters with defaults
st.sidebar.subheader("⚡ Jump-Diffusion Parameters")
jump_params = {
    "Lambda (Jump Intensity)": 0.8,
    "Mean Jump Size": 0.0,
    "Std Dev Jump Size": 0.20
}
lambda_jump = st.sidebar.number_input("Lambda", 0.0, 5.0, jump_params["Lambda (Jump Intensity)"], 0.01)
mean_jump = st.sidebar.number_input("Mean Jump Size", -1.0, 1.0, jump_params["Mean Jump Size"], 0.01)
std_jump = st.sidebar.number_input("Std Dev Jump Size", 0.0, 1.0, jump_params["Std Dev Jump Size"], 0.01)

# Other fixed parameters
st.sidebar.subheader("⚙️ Fixed Parameters")
s0 = st.sidebar.number_input("Initial Spot Price ($/bbl)", 50.0, 150.0, 80.0)
fixed_price = st.sidebar.number_input("Fixed Swap Price ($/bbl)", 50.0, 150.0, 80.0)
drift_rate = st.sidebar.number_input("Drift Rate", 0.0, 0.1, 0.03, 0.001)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.1, 0.02, 0.001)
num_paths = st.sidebar.slider("Monte Carlo Paths", 1000, 10000, 5000, 500)

# Fixed prices array
FIXED_PRICES = np.full(months, fixed_price)

# Conversion and time parameters
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT
BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1 / BUSINESS_DAYS_PER_YEAR
STRIP_LENGTH_YEARS = months / 12
SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        
        # Generate paths
        def generate_paths_jump_diffusion(S0, mu, sigma, lambda_jump, mu_j, sigma_j, T_sim, dt, num_paths):
            M = int(T_sim / dt)
            S = np.zeros((M + 1, num_paths))
            S[0] = S0
            
            jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            adjusted_mu = mu - lambda_jump * jump_mean_e 
            
            for t in range(M):
                Z_brownian = np.random.normal(size=num_paths)
                N_jumps = np.random.poisson(lambda_jump * dt, size=num_paths)
                
                max_jumps = np.max(N_jumps) if np.max(N_jumps) > 0 else 1
                log_jump_magnitudes = np.random.normal(mu_j, sigma_j, size=(max_jumps, num_paths))
    
                total_jump_impact = np.zeros(num_paths)
                for i in range(num_paths):
                    if N_jumps[i] > 0:
                        total_jump_impact[i] = np.sum(log_jump_magnitudes[:N_jumps[i], i])
    
                S[t+1] = S[t] * np.exp(
                    (adjusted_mu - 0.5 * sigma**2) * dt 
                    + sigma * np.sqrt(dt) * Z_brownian
                    + total_jump_impact
                )
            return S
        
        asset_paths = generate_paths_jump_diffusion(
            s0, drift_rate, volatility, lambda_jump, mean_jump, std_jump, 
            time_points[-1], DAILY_TIMESTEP, num_paths
        )
        
        # Exposure calculation
        Strip_PV_paths = np.zeros((SIMULATION_STEPS + 1, num_paths))
        
        for t in range(SIMULATION_STEPS + 1):
            current_strip_mtm = np.zeros(num_paths)
            
            for i in range(months):
                tenor_start_step = int(i * DAYS_PER_MONTH)
                tenor_end_step = int((i + 1) * DAYS_PER_MONTH)
                settlement_step = tenor_end_step + 1
        
                if t >= settlement_step:
                    continue
                    
                elif t >= tenor_start_step and t < tenor_end_step:
                    steps_in_tenor_so_far = range(tenor_start_step, t + 1)
                    days_passed = len(steps_in_tenor_so_far)
                    P_realized_avg_paths = np.mean(asset_paths[steps_in_tenor_so_far, :], axis=0)
        
                    days_remaining = tenor_end_step - t - 1
                    if days_remaining > 0:
                        P_t = asset_paths[t, :]
                        T_to_end_tenor = days_remaining * DAILY_TIMESTEP
                        P_expected_future_paths = P_t * np.exp(drift_rate * T_to_end_tenor)
                    else:
                        P_expected_future_paths = np.zeros(num_paths)
        
                    P_Final_Avg = (P_realized_avg_paths * days_passed + P_expected_future_paths * days_remaining) / DAYS_PER_MONTH
                    T_to_settlement = time_points[settlement_step] - time_points[t]
                    Discount_Factor = np.exp(-risk_free_rate * T_to_settlement)
                    PV_i = (P_Final_Avg - FIXED_PRICES[i]) * NOTIONAL_PER_MONTH[i] * Discount_Factor
                    
                else:
                    P_current = asset_paths[t, :]
                    T_to_tenor = time_points[tenor_start_step] - time_points[t]
                    Expected_Future_Price = P_current * np.exp(drift_rate * T_to_tenor)
                    T_to_settlement = time_points[settlement_step] - time_points[t]
                    Discount_Factor = np.exp(-risk_free_rate * T_to_settlement)
                    PV_i = (Expected_Future_Price - FIXED_PRICES[i]) * NOTIONAL_PER_MONTH[i] * Discount_Factor
        
                current_strip_mtm += PV_i
        
            Strip_PV_paths[t, :] = current_strip_mtm
        
        # Calculate metrics
        Strip_Exposure_paths = np.maximum(0, Strip_PV_paths)
        PFE_95 = np.percentile(Strip_Exposure_paths, 95, axis=1)
        PFE_99 = np.percentile(Strip_Exposure_paths, 99, axis=1)
        EE = np.mean(Strip_Exposure_paths, axis=1)
        
        Neg_PFE_95 = np.percentile(Strip_PV_paths, 5, axis=1)
        Neg_PFE_99 = np.percentile(Strip_PV_paths, 1, axis=1)
        
        sample_pv_paths = Strip_PV_paths[:, np.random.choice(num_paths, min(10, num_paths), replace=False)]
        
        # Store results in session state
        st.session_state.results = {
            'time_points': time_points,
            'PFE_99': PFE_99, 'PFE_95': PFE_95, 'EE': EE,
            'Neg_PFE_99': Neg_PFE_99, 'Neg_PFE_95': Neg_PFE_95,
            'sample_pv_paths': sample_pv_paths
        }
        
        st.success("✅ Simulation completed!")

# Display results if available
if 'results' in st.session_state:
    results = st.session_state.results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max PFE 99%", f"${np.max(results['PFE_99'])/1e6:.1f}M")
    with col2:
        st.metric("Max PFE 95%", f"${np.max(results['PFE_95'])/1e6:.1f}M")
    with col3:
        st.metric("Max EE", f"${np.max(results['EE'])/1e6:.1f}M")
    with col4:
        st.metric("Max Liability 99%", f"${np.max(np.abs(results['Neg_PFE_99']))/1e6:.1f}M")
    
    # Graph 1: Comprehensive Exposure
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(results['time_points'], results['PFE_99'] / 1e6, label='PFE (99%)', color='red', linewidth=2)
    ax1.plot(results['time_points'], results['PFE_95'] / 1e6, label='PFE (95%)', color='orange', linestyle='--', linewidth=2)
    ax1.plot(results['time_points'], results['EE'] / 1e6, label='EE', color='blue', linewidth=2)
    ax1.plot(results['time_points'], results['Neg_PFE_99'] / 1e6, label='Neg PFE (99%)', color='darkgreen', linewidth=2)
    ax1.plot(results['time_points'], results['Neg_PFE_95'] / 1e6, label='Neg PFE (95%)', color='lightgreen', linestyle='--', linewidth=2)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Two-Sided Exposure Profile (PFE & Liability)')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Exposure/Liability (USD Millions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Graph 2: Sample Paths
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    for i in range(results['sample_pv_paths'].shape[1]):
        ax2.plot(results['time_points'], results['sample_pv_paths'][:, i] / 1e6, 
                color='gray', alpha=0.4, linewidth=0.8)
    ax2.plot(results['time_points'], results['Neg_PFE_99'] / 1e6, label='1% Centile (99% Liability)', 
            color='darkred', linewidth=3)
    ax2.plot(results['time_points'], results['Neg_PFE_95'] / 1e6, label='5% Centile (95% Liability)', 
            color='red', linestyle='--', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Monte Carlo Paths with Liability Centiles')
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Mark-to-Market (USD Millions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # Download results
    df_results = pd.DataFrame({
        'Time_Years': results['time_points'],
        'PFE_99_M': results['PFE_99']/1e6,
        'PFE_95_M': results['PFE_95']/1e6,
        'EE_M': results['EE']/1e6,
        'Neg_PFE_99_M': results['Neg_PFE_99']/1e6,
        'Neg_PFE_95_M': results['Neg_PFE_95']/1e6
    })
    csv = df_results.to_csv(index=False)
    st.download_button("📥 Download Results CSV", csv, "oil_strip_exposure.csv", "text/csv")
