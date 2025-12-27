import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(page_title="Oil Strip PFE Simulator", layout="wide", height=800)
st.title("🛢️ Oil Strip PFE Exposure Simulator")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Simulation Parameters")

# 1. Swap months
months = st.sidebar.slider("Swap Months", 1, 24, 12)

# 2. Monthly volumes
st.sidebar.subheader("📈 Monthly Volumes (MT)")
volumes = []
default_vol = 12000.0
for i in range(months):
    vol = st.sidebar.number_input(
        f"Month {i+1}", 
        min_value=0.0, 
        max_value=50000.0, 
        value=default_vol, 
        key=f"vol_{i}"
    )
    volumes.append(vol)

# 3. Core parameters
col1, col2 = st.sidebar.columns(2)
volatility = col1.slider("Volatility", 0.1, 1.0, 0.3, 0.01)
s0 = col2.number_input("Spot Price ($/bbl)", 50.0, 150.0, 80.0)

fixed_price = st.sidebar.number_input("Fixed Swap Price ($/bbl)", 50.0, 150.0, 80.0)

# 4. Jump-Diffusion (with defaults)
st.sidebar.subheader("⚡ Jump-Diffusion")
use_defaults = st.sidebar.checkbox("Use Defaults", value=True)
if not use_defaults:
    lambda_jump = st.sidebar.number_input("Lambda", 0.0, 5.0, 0.8)
    mean_jump = st.sidebar.number_input("Mean Jump", -1.0, 1.0, 0.0)
    std_jump = st.sidebar.number_input("Jump Std", 0.0, 1.0, 0.20)
else:
    lambda_jump, mean_jump, std_jump = 0.8, 0.0, 0.20

# Other params
drift_rate = st.sidebar.number_input("Drift Rate", 0.0, 0.1, 0.03, 0.001)
risk_free_rate = st.sidebar.number_input("Risk Free Rate", 0.0, 0.1, 0.02, 0.001)
num_paths = st.sidebar.slider("MC Paths", 1000, 5000, 2000, 500)

# Fixed prices and notionals
FIXED_PRICES = np.full(months, fixed_price)
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT

# Time setup
BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1 / BUSINESS_DAYS_PER_YEAR
STRIP_LENGTH_YEARS = months / 12
SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)

# Run simulation
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running Jump-Diffusion Monte Carlo..."):
        progress_bar = st.progress(0)
        
        @st.cache_data
        def generate_paths(S0, mu, sigma, lambda_j, mu_j, sigma_j, T, dt, paths):
            M = int(T / dt)
            S = np.zeros((M + 1, paths))
            S[0] = S0
            
            jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            adjusted_mu = mu - lambda_j * jump_mean_e
            
            for t in range(M):
                if t % 100 == 0:
                    progress_bar.progress(t / M)
                
                Z = np.random.normal(size=paths)
                N_jumps = np.random.poisson(lambda_j * dt, size=paths)
                max_j = max(1, N_jumps.max())
                jumps = np.random.normal(mu_j, sigma_j, (max_j, paths))
                
                jump_impact = np.zeros(paths)
                for i in range(paths):
                    if N_jumps[i] > 0:
                        jump_impact[i] = jumps[:N_jumps[i], i].sum()
                
                S[t+1] = S[t] * np.exp(
                    (adjusted_mu - 0.5 * sigma**2) * dt +
                    sigma * np.sqrt(dt) * Z +
                    jump_impact
                )
            return S
        
        # Generate paths
        asset_paths = generate_paths(
            s0, drift_rate, volatility, lambda_jump, mean_jump, std_jump,
            time_points[-1], DAILY_TIMESTEP, num_paths
        )
        progress_bar.progress(0.3)
        
        # Exposure calculation (optimized)
        Strip_PV_paths = np.zeros((SIMULATION_STEPS + 1, num_paths))
        
        for t in range(SIMULATION_STEPS + 1):
            mtm = np.zeros(num_paths)
            
            for i in range(months):
                tenor_start = int(i * DAYS_PER_MONTH)
                tenor_end = int((i + 1) * DAYS_PER_MONTH)
                settle = tenor_end + 1
                
                if t >= settle:
                    continue
                elif tenor_start <= t < tenor_end:
                    # Averaging period
                    days_passed = t - tenor_start + 1
                    days_remain = DAYS_PER_MONTH - days_passed
                    
                    p_realized = np.mean(asset_paths[tenor_start:t+1], axis=0)
                    if days_remain > 0:
                        p_future = asset_paths[t] * np.exp(drift_rate * days_remain * DAILY_TIMESTEP)
                    else:
                        p_future = 0
                        
                    p_final = (p_realized * days_passed + p_future * days_remain) / DAYS_PER_MONTH
                else:
                    # Forward looking
                    t_to_tenor = time_points[tenor_start] - time_points[t]
                    p_final = asset_paths[t] * np.exp(drift_rate * t_to_tenor)
                
                t_to_settle = time_points[settle] - time_points[t]
                df = np.exp(-risk_free_rate * t_to_settle)
                mtm += (p_final - FIXED_PRICES[i]) * NOTIONAL_PER_MONTH[i] * df
            
            Strip_PV_paths[t] = mtm
            if t % 50 == 0:
                progress_bar.progress(0.3 + 0.7 * t / SIMULATION_STEPS)
        
        # Metrics
        exposure = np.maximum(0, Strip_PV_paths)
        PFE_95 = np.percentile(exposure, 95, axis=1)
        PFE_99 = np.percentile(exposure, 99, axis=1)
        EE = np.mean(exposure, axis=1)
        Neg_PFE_95 = np.percentile(Strip_PV_paths, 5, axis=1)
        Neg_PFE_99 = np.percentile(Strip_PV_paths, 1, axis=1)
        
        sample_paths = Strip_PV_paths[:, np.random.choice(num_paths, 10, replace=False)]
        
        # Store results
        st.session_state.results = {
            'time': time_points, 'PFE_99': PFE_99, 'PFE_95': PFE_95, 'EE': EE,
            'Neg_99': Neg_PFE_99, 'Neg_95': Neg_PFE_95, 'samples': sample_paths
        }
        progress_bar.progress(1.0)
        st.success("✅ Simulation complete!")

# Results
if 'results' in st.session_state:
    results = st.session_state.results
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Max PFE 99%", f"${np.max(results['PFE_99'])/1e6:.1f}M")
    with col2: st.metric("Max PFE 95%", f"${np.max(results['PFE_95'])/1e6:.1f}M")
    with col3: st.metric("Max EE", f"${np.max(results['EE'])/1e6:.1f}M")
    with col4: st.metric("Max Liability 99%", f"${np.max(np.abs(results['Neg_99']))/1e6:.1f}M")
    with col5: st.metric("Total Notional", f"${np.sum(NOTIONAL_PER_MONTH * 30)/1e6:.0f}M")
    
    st.markdown("---")
    
    # Graph 1: Exposure Profile
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(results['time'], results['PFE_99']/1e6, 'r-', lw=2, label='PFE 99%')
    ax1.plot(results['time'], results['PFE_95']/1e6, 'orange', ls='--', lw=2, label='PFE 95%')
    ax1.plot(results['time'], results['EE']/1e6, 'b-', lw=2, label='EE')
    ax1.plot(results['time'], results['Neg_99']/1e6, 'darkgreen', lw=2, label='Liability 99%')
    ax1.plot(results['time'], results['Neg_95']/1e6, 'lightgreen', ls='--', lw=2, label='Liability 95%')
    ax1.axhline(0, color='k', ls='-', alpha=0.3)
    ax1.set_title('Oil Strip Two-Sided Exposure Profile', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Exposure/Liability ($M)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Graph 2: Sample Paths
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    for i in range(results['samples'].shape[1]):
        ax2.plot(results['time'], results['samples'][:,i]/1e6, 'gray', alpha=0.3, lw=0.8)
    ax2.plot(results['time'], results['Neg_99']/1e6, 'darkred', lw=3, label='1% Centile')
    ax2.plot(results['time'], results['Neg_95']/1e6, 'red', ls='--', lw=2, label='5% Centile')
    ax2.axhline(0, color='k', ls='-', alpha=0.3)
    ax2.set_title('Monte Carlo Paths + Liability Centiles', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Mark-to-Market ($M)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    # Download
    df = pd.DataFrame({
        'Time_Years': results['time'],
        'PFE_99_M': results['PFE_99']/1e6,
        'PFE_95_M': results['PFE_95']/1e6,
        'EE_M': results['EE']/1e6,
        'Neg_PFE_99_M': results['Neg_99']/1e6,
        'Neg_PFE_95_M': results['Neg_95']/1e6
    })
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "oil_pfe_results.csv", "text/csv")

st.markdown("---")
st.caption("Powered by Jump-Diffusion Monte Carlo | Optimized for Streamlit Cloud")
