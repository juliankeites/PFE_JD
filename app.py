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

# ✅ Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'run_triggered' not in st.session_state:
    st.session_state.run_triggered = False

# Sidebar
st.sidebar.header("📊 Simulation Parameters")

# ✅ Use session state for months to prevent input conflicts
if 'months' not in st.session_state:
    st.session_state.months = 12

months = st.sidebar.slider("Swap Months", 1, 24, st.session_state.months, key="months_slider")
st.session_state.months = months

st.sidebar.subheader("📈 Monthly Volumes (MT)")
volumes = []
default_vol = 12000.0

# ✅ Clear volume keys when months change
if st.session_state.get('prev_months', 0) != months:
    for i in range(st.session_state.get('prev_months', 0)):
        st.session_state.pop(f"vol_{i}", None)
    st.session_state.prev_months = months

for i in range(months):
    vol = st.sidebar.number_input(
        f"M{i+1}", 
        min_value=0.0, 
        max_value=50000.0, 
        value=default_vol, 
        key=f"vol_{i}"
    )
    volumes.append(vol)

col1, col2 = st.sidebar.columns(2)
volatility = col1.slider("Volatility", 0.10, 1.00, 0.30, 0.01, key="vol_slider")
spot_price = col2.number_input("Spot ($/bbl)", 50.0, 150.0, 80.0, key="spot_slider")

fixed_price = st.sidebar.number_input("Fixed Price ($/bbl)", 50.0, 150.0, 80.0, key="fixed_slider")

st.sidebar.subheader("⚡ Jump-Diffusion")
use_defaults = st.sidebar.checkbox("Use Defaults", value=True, key="defaults_cb")
if use_defaults:
    lambda_jump, mean_jump, std_jump = 0.8, 0.0, 0.20
else:
    lambda_jump = st.sidebar.number_input("Lambda", 0.0, 5.0, 0.8, 0.01, key="lambda_slider")
    mean_jump = st.sidebar.number_input("Mean Jump", -1.0, 1.0, 0.0, 0.01, key="mean_slider")
    std_jump = st.sidebar.number_input("Jump Std", 0.0, 1.0, 0.20, 0.01, key="std_slider")

drift_rate = st.sidebar.number_input("Drift", 0.0, 0.10, 0.03, 0.001, key="drift_slider")
risk_free = st.sidebar.number_input("Risk Free", 0.0, 0.10, 0.02, 0.001, key="risk_slider")
num_paths = st.sidebar.slider("MC Paths", 500, 3000, 1500, 250, key="paths_slider")

# Constants
FIXED_PRICES = np.full(months, fixed_price)
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT

BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1 / BUSINESS_DAYS_PER_YEAR
STRIP_LENGTH_YEARS = months / 12.0
SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)

# ✅ FIXED BUTTON - Unique key prevents conflicts
if st.sidebar.button("🚀 Run Simulation", type="primary", key="run_button", use_container_width=True):
    st.session_state.run_triggered = True
    st.session_state.results = None  # Clear previous
    
    with st.spinner("Running Jump-Diffusion Monte Carlo..."):
        def run_simulation(months, notionals, fixed_prices, s0, vol, drift, rf, lambda_j, mu_j, sigma_j, paths, time_pts):
            M = len(time_pts) - 1
            S = np.full((M + 1, paths), s0)
            
            jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            adj_mu = drift - lambda_j * jump_mean_e
            
            for t in range(M):
                Z = np.random.normal(size=paths)
                N_jumps = np.random.poisson(lambda_j * DAILY_TIMESTEP, size=paths)
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
                        days_passed = min(t - tenor_start + 1, DAYS_PER_MONTH)
                        days_remain = max(0, DAYS_PER_MONTH - days_passed)
                        p_real = np.mean(S[tenor_start:min(t+1, tenor_end)], axis=0)
                        p_future = S[t] * np.exp(drift * days_remain * DAILY_TIMESTEP) if days_remain > 0 else 0
                        p_final = (p_real * days_passed + p_future * days_remain) / DAYS_PER_MONTH
                    else:
                        t_to_tenor = time_pts[min(tenor_start, M)] - time_pts[min(t, M)]
                        p_final = S[min(t, M)] * np.exp(drift * t_to_tenor)
                    
                    t_to_settle = time_pts[min(settle, M)] - time_pts[min(t, M)]
                    df = np.exp(-rf * t_to_settle)
                    mtm += (p_final - fixed_prices[i]) * notionals[i] * df
                
                PV_paths[t] = mtm
            
            exposure = np.maximum(0, PV_paths)
            pfe_95 = np.percentile(exposure, 95, axis=1)
            pfe_99 = np.percentile(exposure, 99, axis=1)
            ee = np.mean(exposure, axis=1)
            neg_95 = np.percentile(PV_paths, 5, axis=1)
            neg_99 = np.percentile(PV_paths, 1, axis=1)
            
            return {
                'time': time_pts,
                'pfe_99': pfe_99, 'pfe_95': pfe_95, 'ee': ee,
                'neg_99': neg_99, 'neg_95': neg_95,
                'samples': PV_paths[:, np.random.choice(paths, min(10, paths), replace=False)]
            }
        
        results = run_simulation(months, NOTIONAL_PER_MONTH, FIXED_PRICES, spot_price, 
                               volatility, drift_rate, risk_free, lambda_jump, 
                               mean_jump, std_jump, num_paths, time_points)
        st.session_state.results = results
    
    st.success("✅ Simulation complete!")
    st.rerun()

# ✅ Results display
if st.session_state.results is not None:
    results = st.session_state.results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Max PFE 99%", f"${np.max(results['pfe_99'])/1e6:.1f}M")
    with col2: st.metric("Max PFE 95%", f"${np.max(results['pfe_95'])/1e6:.1f}M")
    with col3: st.metric("Max EE", f"${np.max(results['ee'])/1e6:.1f}M")
    with col4: st.metric("Max Liability", f"${np.max(np.abs(results['neg_99']))/1e6:.1f}M")
    
    st.markdown("---")
    
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(results['time'], results['pfe_99']/1e6, 'r-', lw=3, label='PFE 99%')
    ax1.plot(results['time'], results['pfe_95']/1e6, 'orange', ls='--', lw=2.5, label='PFE 95%')
    ax1.plot(results['time'], results['ee']/1e6, 'b-', lw=2.5, label='Expected Exposure')
    ax1.plot(results['time'], results['neg_99']/1e6, 'darkgreen', lw=3, label='Liability 99%')
    ax1.plot(results['time'], results['neg_95']/1e6, 'lightgreen', ls='--', lw=2.5, label='Liability 95%')
    ax1.axhline(0, color='k', ls='-', alpha=0.5)
    ax1.set_title('Oil Strip Two-Sided Exposure Profile', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Exposure/Liability ($ Millions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    for i in range(results['samples'].shape[1]):
        ax2.plot(results['time'], results['samples'][:,i]/1e6, 'gray', alpha=0.4, lw=1)
    ax2.plot(results['time'], results['neg_99']/1e6, 'darkred', lw=4, label='1% Centile (99% Liability)')
    ax2.plot(results['time'], results['neg_95']/1e6, 'red', ls='--', lw=3, label='5% Centile (95% Liability)')
    ax2.axhline(0, color='k', ls='-', alpha=0.5)
    ax2.set_title('Monte Carlo Paths + Liability Centiles', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Mark-to-Market ($ Millions)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)
    
    df = pd.DataFrame({
        'Time_Years': results['time'],
        'PFE_99_M': results['pfe_99']/1e6,
        'PFE_95_M': results['pfe_95']/1e6,
        'EE_M': results['ee']/1e6,
        'NegPFE_99_M': results['neg_99']/1e6,
        'NegPFE_95_M': results['neg_95']/1e6
    })
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results", csv, "oil_pfe_results.csv", "text/csv")

else:
    st.info("👆 Click **'Run Simulation'** to generate exposure profiles")

st.markdown("---")
st.caption("💡 First month starts next actual month | Jump-Diffusion Monte Carlo")

