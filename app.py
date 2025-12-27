import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(page_title="Oil Strip PFE Simulator", layout="wide", initial_sidebar_state="expanded")

st.title("🛢️ Oil Strip PFE Exposure Simulator")
st.markdown("**Jump-Diffusion Monte Carlo for Oil Swap Strips**")
st.markdown("---")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar - SIMPLIFIED inputs
st.sidebar.header("📊 Simulation Parameters")

months = st.sidebar.slider("Swap Months", 1, 24, 6)
quick_mode = st.sidebar.checkbox("⚡ Quick Mode (500 paths)", value=True)
num_paths = 500 if quick_mode else st.sidebar.slider("MC Paths", 1000, 3000, 1500, 250)

st.sidebar.subheader("📈 Monthly Volumes (MT)")
volumes = []
default_vol = 12000.0
for i in range(months):
    vol = st.sidebar.number_input(f"M{i+1}", 0.0, 50000.0, default_vol, key=f"vol_{i}")
    volumes.append(vol)

volatility = st.sidebar.slider("Volatility", 0.10, 1.00, 0.30, 0.01)
spot_price = st.sidebar.number_input("Spot ($/bbl)", 50.0, 150.0, 80.0)
fixed_price = st.sidebar.number_input("Fixed Price ($/bbl)", 50.0, 150.0, 80.0)

st.sidebar.subheader("⚡ Jump-Diffusion")
use_defaults = st.sidebar.checkbox("Use Defaults", value=True)
if use_defaults:
    lambda_jump, mean_jump, std_jump = 0.8, 0.0, 0.20
else:
    lambda_jump = st.sidebar.number_input("Lambda", 0.0, 5.0, 0.8)
    mean_jump = st.sidebar.number_input("Mean Jump", -1.0, 1.0, 0.0)
    std_jump = st.sidebar.number_input("Jump Std", 0.0, 1.0, 0.20)

drift_rate = st.sidebar.number_input("Drift", 0.0, 0.10, 0.03)
risk_free = st.sidebar.number_input("Risk Free", 0.0, 0.10, 0.02)

# ✅ DAILY CALCULATION RESTORED
FIXED_PRICES = np.full(months, fixed_price)
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT

BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1 / BUSINESS_DAYS_PER_YEAR  # ✅ DAILY
STRIP_LENGTH_YEARS = months / 12.0
SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)  # ✅ Full daily steps
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)

# ✅ RELIABLE BUTTON + PROGRESS
if st.sidebar.button("🚀 Run Simulation", type="primary", key="run_btn", use_container_width=True):
    st.session_state.results = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🎲 Generating daily price paths...")
    progress_bar.progress(10)
    
    # ✅ DAILY SIMULATION WITH RELIABLE JUMPS
    def run_simulation_daily(months, notionals, fixed_prices, s0, vol, drift, rf, 
                           lambda_j, mu_j, sigma_j, paths, time_pts):
        M = len(time_pts) - 1
        S = np.full((M + 1, paths), s0)
        
        jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        adj_mu = drift - lambda_j * jump_mean_e
        
        # ✅ DAILY PATH GENERATION - Simple reliable jumps
        for t in range(M):
            Z = np.random.normal(size=paths)
            N_jumps = np.random.poisson(lambda_j * DAILY_TIMESTEP, size=paths)
            
            jump_impact = np.zeros(paths)
            for i in range(paths):
                if N_jumps[i] > 0:
                    jumps = np.random.normal(mu_j, sigma_j, N_jumps[i])
                    jump_impact[i] = np.sum(jumps)
            
            S[t+1] = S[t] * np.exp(
                (adj_mu - 0.5 * vol**2) * DAILY_TIMESTEP +
                vol * np.sqrt(DAILY_TIMESTEP) * Z +
                jump_impact
            )
        
        status_text.text("📊 Calculating daily exposures...")
        progress_bar.progress(50)
        
        # ✅ DAILY EXPOSURE CALCULATION
        PV_paths = np.zeros((M + 1, paths))
        
        for t in range(M + 1):
            mtm = np.zeros(paths)
            
            for i in range(months):
                tenor_start = int(i * DAYS_PER_MONTH)
                tenor_end = int((i + 1) * DAYS_PER_MONTH)
                settle = min(tenor_end + 1, M)
                
                if t >= settle:
                    continue
                
                if tenor_start <= t < tenor_end:
                    # Averaging period
                    days_passed = min(t - tenor_start + 1, DAYS_PER_MONTH)
                    days_remain = max(0, DAYS_PER_MONTH - days_passed)
                    start_idx = max(0, tenor_start)
                    end_idx = min(t + 1, tenor_end)
                    p_real = np.mean(S[start_idx:end_idx], axis=0)
                    p_future = S[t] * np.exp(drift * days_remain * DAILY_TIMESTEP) if days_remain > 0 else S[t]
                    p_final = (p_real * days_passed + p_future * days_remain) / DAYS_PER_MONTH
                else:
                    # Forward looking
                    t_to_tenor = time_pts[min(tenor_start, M)] - time_pts[min(t, M)]
                    p_final = S[min(t, M)] * np.exp(drift * t_to_tenor)
                
                t_to_settle = time_pts[min(settle, M)] - time_pts[min(t, M)]
                df = np.exp(-rf * t_to_settle)
                mtm += (p_final - fixed_prices[i]) * notionals[i] * df
            
            PV_paths[t] = mtm
        
        status_text.text("📈 Computing daily statistics...")
        progress_bar.progress(80)
        
        # Metrics
        exposure = np.maximum(0, PV_paths)
        pfe_95 = np.percentile(exposure, 95, axis=1)
        pfe_99 = np.percentile(exposure, 99, axis=1)
        ee = np.mean(exposure, axis=1)
        neg_95 = np.percentile(PV_paths, 5, axis=1)
        neg_99 = np.percentile(PV_paths, 1, axis=1)
        
        progress_bar.progress(100)
        status_text.text("✅ Daily simulation complete!")
        
        return {
            'time': time_pts,
            'pfe_99': pfe_99, 'pfe_95': pfe_95, 'ee': ee,
            'neg_99': neg_99, 'neg_95': neg_95,
            'samples': PV_paths[:, :min(10, paths)]
        }
    
    # Run daily simulation
    results = run_simulation_daily(months, NOTIONAL_PER_MONTH, FIXED_PRICES, spot_price, 
                                 volatility, drift_rate, risk_free, lambda_jump, 
                                 mean_jump, std_jump, num_paths, time_points)
    st.session_state.results = results
    
    time.sleep(1)
    st.rerun()

# Results display
if st.session_state.results is not None:
    results = st.session_state.results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Max PFE 99%", f"${np.max(results['pfe_99'])/1e6:.1f}M")
    with col2: st.metric("Max PFE 95%", f"${np.max(results['pfe_95'])/1e6:.1f}M")
    with col3: st.metric("Max EE", f"${np.max(results['ee'])/1e6:.1f}M")
    with col4: st.metric("Max Liability", f"${np.abs(np.min(results['neg_99']))/1e6:.1f}M")
    
    st.markdown("---")
    
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(results['time'], results['pfe_99']/1e6, 'r-', lw=3, label='PFE 99%')
    ax1.plot(results['time'], results['pfe_95']/1e6, 'orange', ls='--', lw=2.5, label='PFE 95%')
    ax1.plot(results['time'], results['ee']/1e6, 'b-', lw=2.5, label='Expected Exposure')
    ax1.plot(results['time'], results['neg_99']/1e6, 'darkgreen', lw=3, label='Liability 99%')
    ax1.plot(results['time'], results['neg_95']/1e6, 'lightgreen', ls='--', lw=2.5, label='Liability 95%')
    ax1.axhline(0, color='k', ls='-', alpha=0.5)
    ax1.set_title('Oil Strip Two-Sided Exposure Profile (Daily)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Exposure/Liability ($ Millions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    for i in range(results['samples'].shape[1]):
        ax2.plot(results['time'], results['samples'][:,i]/1e6, 'gray', alpha=0.4, lw=1)
    ax2.plot(results['time'], results['neg_99']/1e6, 'darkred', lw=4, label='1% Centile')
    ax2.plot(results['time'], results['neg_95']/1e6, 'red', ls='--', lw=3, label='5% Centile')
    ax2.axhline(0, color='k', ls='-', alpha=0.5)
    ax2.set_title('Monte Carlo Paths + Liability Centiles (Daily)', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    df = pd.DataFrame({
        'Time_Years': results['time'],
        'PFE_99_M': results['pfe_99']/1e6,
        'PFE_95_M': results['pfe_95']/1e6,
        'EE_M': results['ee']/1e6,
        'NegPFE_99_M': results['neg_99']/1e6,
        'NegPFE_95_M': results['neg_95']/1e6
    })
    st.download_button("📥 Download CSV", df.to_csv(index=False), "oil_pfe_results.csv")

else:
    st.info("🚀 **Click 'Run Simulation'** - Quick Mode = 15-30s | Full = 45-90s (Daily calculation)")

st.markdown("---")
st.caption("✅ Daily timesteps restored | Reliable jumps | Full original logic")
