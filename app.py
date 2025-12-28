import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time

# -------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Oil Strip PFE Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Oil Strip PFE Exposure Simulator")
st.markdown("**Jump-Diffusion Monte Carlo for Monthly-Average Oil Swaps**")
st.markdown("---")

# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
if 'results' not in st.session_state:
    st.session_state.results = None
if 'margin_clip' not in st.session_state:
    st.session_state.margin_clip = None

# -------------------------------------------------------------
# Sidebar inputs
# -------------------------------------------------------------
st.sidebar.header("📊 Simulation Parameters")

months = st.sidebar.slider("Swap Months", 1, 24, 12)

quick_mode = st.sidebar.checkbox("⚡ Quick Mode (500 paths)", value=False)
num_paths = 500 if quick_mode else st.sidebar.slider("MC Paths", 1000, 10000, 5000, 500)

st.sidebar.subheader("📈 Monthly Volumes (MT)")
volumes = []
default_vol = 12000.0
for i in range(months):
    vol = st.sidebar.number_input(
        f"M{i+1}",
        min_value=0.0,
        max_value=50000.0,
        value=default_vol,
        key=f"vol_{i}"
    )
    volumes.append(vol)

# Volatility with 3 decimal places
volatility = st.sidebar.number_input("Volatility σ (annual)", 0.001, 5.000, 0.300, format="%.3f")

spot_price = st.sidebar.number_input("Spot ($/bbl)", 1.0, 500.0, 80.0)
fixed_price = st.sidebar.number_input("Fixed Price ($/bbl)", 1.0, 500.0, 80.0)

st.sidebar.subheader("⚡ Jump-Diffusion Parameters")

use_jumps = st.sidebar.checkbox("Enable Jump-Diffusion", value=True)

lambda_jump = st.sidebar.number_input("Lambda (Intensity)", 0.000, 5.000, 0.800, format="%.3f")
mean_jump = st.sidebar.number_input("Mean Jump Size", -1.000, 1.000, 0.000, format="%.3f")
std_jump = st.sidebar.number_input("Jump Std Dev", 0.000, 2.000, 0.200, format="%.3f")

drift_rate = st.sidebar.number_input("Drift Rate μ", -0.200, 0.200, 0.000, format="%.3f")
risk_free = st.sidebar.number_input("Risk-Free Rate r", -0.200, 0.200, 0.00, format="%.3f")

# -------------------------------------------------------------
# Contract / time grid / settlement profile
# -------------------------------------------------------------
FIXED_PRICES = np.full(months, fixed_price)
BBL_PER_MT = 7.88
NOTIONAL_PER_MONTH = np.array(volumes) * BBL_PER_MT  # bbl per month

BUSINESS_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = BUSINESS_DAYS_PER_YEAR / 12
DAILY_TIMESTEP = 1.0 / BUSINESS_DAYS_PER_YEAR
STRIP_LENGTH_YEARS = months / 12.0

SIMULATION_STEPS = int(BUSINESS_DAYS_PER_YEAR * STRIP_LENGTH_YEARS + 1)
time_points = np.linspace(0, STRIP_LENGTH_YEARS + DAILY_TIMESTEP, SIMULATION_STEPS + 1)
time_days = time_points * BUSINESS_DAYS_PER_YEAR  # x-axis in business days

# Settlement window: 5 business days after averaging
SETTLEMENT_WINDOW_DAYS = 5
SETTLEMENT_WINDOW_STEPS = int(SETTLEMENT_WINDOW_DAYS)

# -------------------------------------------------------------
# Core engine: daily jump-diffusion swap PVs
# -------------------------------------------------------------
def simulate_strip(
    months,
    notionals,
    fixed_prices,
    s0,
    vol,
    drift,
    rf,
    lambda_j,
    mu_j,
    sigma_j,
    use_jumps,
    paths,
    time_pts
):
    """
    Simulates daily price paths and strip PVs for a multi-month monthly-average swap
    with a short settlement window and returns PFE statistics and PV paths.
    """
    M = len(time_pts) - 1
    S = np.full((M + 1, paths), s0)

    # Drift adjustment for jump diffusion (if enabled)
    if lambda_j > 0.0 and sigma_j > 0.0 and use_jumps:
        jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        adj_mu = drift - lambda_j * jump_mean_e
    else:
        jump_mean_e = 0.0
        adj_mu = drift

    # 1. Generate daily price paths
    for t in range(M):
        Z = np.random.normal(size=paths)

        jump_impact = np.zeros(paths)
        if lambda_j > 0.0 and sigma_j > 0.0 and use_jumps:
            N_jumps = np.random.poisson(lambda_j * DAILY_TIMESTEP, size=paths)
            for i in range(paths):
                if N_jumps[i] > 0:
                    jumps = np.random.normal(mu_j, sigma_j, N_jumps[i])
                    jump_impact[i] = np.sum(jumps)

        S[t + 1] = S[t] * np.exp(
            (adj_mu - 0.5 * vol**2) * DAILY_TIMESTEP
            + vol * np.sqrt(DAILY_TIMESTEP) * Z
            + jump_impact
        )

    # 2. Strip PV paths
    PV_paths = np.zeros((M + 1, paths))

    for t in range(M + 1):
        mtm = np.zeros(paths)

        for i in range(months):
            tenor_start = int(i * DAYS_PER_MONTH)
            tenor_end = int((i + 1) * DAYS_PER_MONTH)

            settle_start = tenor_end + 1
            settle_end = min(tenor_end + SETTLEMENT_WINDOW_STEPS, M)

            if t > settle_end:
                continue

            # Averaging period
            if tenor_start <= t < tenor_end:
                days_passed = min(t - tenor_start + 1, DAYS_PER_MONTH)
                days_remain = max(0.0, DAYS_PER_MONTH - days_passed)

                start_idx = max(0, tenor_start)
                end_idx = min(t + 1, tenor_end)
                p_real = np.mean(S[start_idx:end_idx], axis=0)

                if days_remain > 0:
                    p_future = S[t] * np.exp(drift * days_remain * DAILY_TIMESTEP)
                else:
                    p_future = S[t]

                p_final = (p_real * days_passed + p_future * days_remain) / DAYS_PER_MONTH

                pay_step = min((settle_start + settle_end) // 2, M)
                t_to_pay = time_pts[pay_step] - time_pts[min(t, M)]
                df = np.exp(-rf * t_to_pay)

                month_pv = (p_final - fixed_prices[i]) * notionals[i] * df

            # Settlement window: roll-off
            elif tenor_end <= t <= settle_end:
                start_idx = max(0, tenor_start)
                end_idx = min(tenor_end, M)
                p_real = np.mean(S[start_idx:end_idx], axis=0)

                pay_step = min((settle_start + settle_end) // 2, M)
                t_to_pay_from_end = time_pts[pay_step] - time_pts[min(tenor_end, M)]
                df_end = np.exp(-rf * t_to_pay_from_end)

                month_pv_at_end = (p_real - fixed_prices[i]) * notionals[i] * df_end

                if settle_end > settle_start:
                    decay = 1.0 - (t - settle_start) / (settle_end - settle_start)
                else:
                    decay = 0.0
                decay = max(decay, 0.0)

                month_pv = month_pv_at_end * decay

            # Pre‑tenor: forward pricing
            else:
                t_to_tenor = time_pts[min(tenor_start, M)] - time_pts[min(t, M)]
                p_final = S[min(t, M)] * np.exp(drift * t_to_tenor)

                pay_step = min((settle_start + settle_end) // 2, M)
                t_to_pay = time_pts[pay_step] - time_pts[min(t, M)]
                df = np.exp(-rf * t_to_pay)

                month_pv = (p_final - fixed_prices[i]) * notionals[i] * df

            mtm += month_pv

        PV_paths[t] = mtm

    # 3. Exposure statistics
    exposure = np.maximum(0.0, PV_paths)
    pfe_95 = np.percentile(exposure, 95, axis=1)
    pfe_99 = np.percentile(exposure, 99, axis=1)
    ee = np.mean(exposure, axis=1)
    neg_95 = np.percentile(PV_paths, 5, axis=1)
    neg_99 = np.percentile(PV_paths, 1, axis=1)

    return {
        "time": time_pts,
        "time_days": time_days,
        "pfe_99": pfe_99,
        "pfe_95": pfe_95,
        "ee": ee,
        "neg_99": neg_99,
        "neg_95": neg_95,
        "PV_paths": PV_paths,
    }

# -------------------------------------------------------------
# Helper: 2‑day margin for a 1‑month 1,000 bbl clip
# -------------------------------------------------------------
def margin_for_1k_bbl_clip(
    s0,
    vol,
    drift,
    rf,
    lambda_j,
    mu_j,
    sigma_j,
    use_jumps,
    paths,
    fixed_price_clip
):
    """
    Run the same model for a single‑month swap with exactly 1,000 bbl notional
    and return the 2‑day PFE99 in USD as the margin per 1,000 bbl.
    """
    # Single-month grid (1 month)
    clip_months = 1
    clip_length_years = 1.0 / 12.0
    steps = int(BUSINESS_DAYS_PER_YEAR * clip_length_years + 1)
    t_grid = np.linspace(0, clip_length_years + DAILY_TIMESTEP, steps + 1)
    t_days = t_grid * BUSINESS_DAYS_PER_YEAR

    # 1,000 bbl in MT
    mt_for_1000_bbl = 1000.0 / BBL_PER_MT
    notionals_clip = np.array([1000.0])  # directly 1000 bbl notional
    fixed_prices_clip = np.array([fixed_price_clip])

    # Re‑use simulate_strip over 1 month
    res_clip = simulate_strip(
        clip_months,
        notionals_clip,
        fixed_prices_clip,
        s0,
        vol,
        drift,
        rf,
        lambda_j,
        mu_j,
        sigma_j,
        use_jumps,
        paths,
        t_grid,
    )

    exposure_clip = np.maximum(0.0, res_clip["PV_paths"])
    pfe99_clip = np.percentile(exposure_clip, 99, axis=1)

    # 2‑business‑day index (t ≈ 2 days)
    day2 = 2
    idx2 = min(day2, len(pfe99_clip) - 1)
    margin_2d_per_1000 = pfe99_clip[idx2]  # USD per 1,000 bbl

    return margin_2d_per_1000

# -------------------------------------------------------------
# Run button
# -------------------------------------------------------------
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True, key="run_btn"):
    st.session_state.results = None
    st.session_state.margin_clip = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🎲 Simulating full strip...")
    progress_bar.progress(10)

    # 1) Full strip simulation
    res = simulate_strip(
        months,
        NOTIONAL_PER_MONTH,
        FIXED_PRICES,
        spot_price,
        volatility,
        drift_rate,
        risk_free,
        lambda_jump,
        mean_jump,
        std_jump,
        use_jumps,
        num_paths,
        time_points,
    )

    st.session_state.results = {
        "time": res["time"],
        "time_days": res["time_days"],
        "pfe_99": res["pfe_99"],
        "pfe_95": res["pfe_95"],
        "ee": res["ee"],
        "neg_99": res["neg_99"],
        "neg_95": res["neg_95"],
        "samples": res["PV_paths"][:, : min(10, num_paths)],
    }

    progress_bar.progress(60)
    status_text.text("📐 Calculating 2‑day margin for 1,000 bbl clip...")

    # 2) Single‑month 1,000 bbl clip margin
    margin_clip = margin_for_1k_bbl_clip(
        spot_price,
        volatility,
        drift_rate,
        risk_free,
        lambda_jump,
        mean_jump,
        std_jump,
        use_jumps,
        num_paths,
        fixed_price,
    )
    st.session_state.margin_clip = margin_clip

    progress_bar.progress(100)
    status_text.text("✅ Simulation complete")
    time.sleep(0.5)
    st.rerun()

# -------------------------------------------------------------
# Results display
# -------------------------------------------------------------
if st.session_state.results is not None:
    results = st.session_state.results
    margin_clip = st.session_state.margin_clip if st.session_state.margin_clip is not None else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Max PFE 99%", f"${np.max(results['pfe_99']) / 1e6:.3f}M")
    with col2:
        st.metric("Max PFE 95%", f"${np.max(results['pfe_95']) / 1e6:.3f}M")
    with col3:
        st.metric("Max EE", f"${np.max(results['ee']) / 1e6:.3f}M")
    with col4:
        st.metric("Max Liability (99%)", f"${np.abs(np.min(results['neg_99'])) / 1e6:.3f}M")
    with col5:
        st.metric("2‑Day Margin / 1k bbl (PFE99 at 2d)", f"${margin_clip:,.2f}")

    st.markdown("---")

    # -------- Graph 1: Exposure profile --------
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(results["time_days"], results["pfe_99"] / 1e6, "r-", lw=3, label="PFE 99%")
    ax1.plot(results["time_days"], results["pfe_95"] / 1e6, "orange", ls="--", lw=2.5, label="PFE 95%")
    ax1.plot(results["time_days"], results["ee"] / 1e6, "b-", lw=2.5, label="Expected Exposure")
    ax1.plot(results["time_days"], results["neg_99"] / 1e6, "darkgreen", lw=3, label="Liability 99%")
    ax1.plot(results["time_days"], results["neg_95"] / 1e6, "lightgreen", ls="--", lw=2.5, label="Liability 95%")
    ax1.axhline(0, color="k", ls="-", alpha=0.5)

    all_y = np.concatenate(
        [
            results["pfe_99"] / 1e6,
            results["pfe_95"] / 1e6,
            results["ee"] / 1e6,
            results["neg_99"] / 1e6,
            results["neg_95"] / 1e6,
        ]
    )
    y_min, y_max = np.min(all_y), np.max(all_y)
    y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.001
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    ax1.set_title("Oil Strip Two-Sided Exposure Profile (Daily)", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xlabel("Business Days")
    ax1.set_ylabel("Exposure / Liability ($ Millions)")
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

    # -------- Graph 2: MC paths + liability centiles --------
    fig2, ax2 = plt.subplots(figsize=(16, 6))

    for i in range(results["samples"].shape[1]):
        ax2.plot(results["time_days"], results["samples"][:, i] / 1e6, color="gray", alpha=0.4, lw=1)

    ax2.plot(results["time_days"], results["neg_99"] / 1e6, "darkred", lw=4, label="1% Centile (99% Liability)")
    ax2.plot(results["time_days"], results["neg_95"] / 1e6, "red", ls="--", lw=3, label="5% Centile (95% Liability)")
    ax2.axhline(0, color="k", ls="-", alpha=0.5)

    all_paths = np.concatenate(
        [
            results["samples"].flatten() / 1e6,
            results["neg_99"] / 1e6,
            results["neg_95"] / 1e6,
        ]
    )
    py_min, py_max = np.min(all_paths), np.max(all_paths)
    py_pad = (py_max - py_min) * 0.05 if py_max > py_min else 0.001
    ax2.set_ylim(py_min - py_pad, py_max + py_pad)

    ax2.set_title("Monte Carlo Paths + Liability Centiles (Daily)", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Business Days")
    ax2.set_ylabel("Mark-to-Market ($ Millions)")
    ax2.ticklabel_format(style="plain", axis="y")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

    df = pd.DataFrame(
        {
            "Business_Days": results["time_days"],
            "Time_Years": results["time"],
            "PFE_99_M": results["pfe_99"] / 1e6,
            "PFE_95_M": results["pfe_95"] / 1e6,
            "EE_M": results["ee"] / 1e6,
            "NegPFE_99_M": results["neg_99"] / 1e6,
            "NegPFE_95_M": results["neg_95"] / 1e6,
        }
    )
    st.download_button("📥 Download CSV", df.to_csv(index=False), "oil_pfe_results.csv")

else:
    st.info("🚀 Set parameters and click **Run Simulation** to generate strip PFE and 2‑day margin per 1,000 bbl.")
