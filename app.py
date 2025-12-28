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

drift_rate = st.sidebar.number_input("Drift Rate μ", -0.200, 0.200, 0.030, format="%.3f")
risk_free = st.sidebar.number_input("Risk-Free Rate r", -0.200, 0.200, 0.020, format="%.3f")

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
# Monte Carlo engine
# -------------------------------------------------------------
def run_simulation_daily(
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
    M = len(time_pts) - 1
    S = np.full((M + 1, paths), s0)

    # Drift adjustment (only if jumps enabled and non-zero)
    if lambda_j > 0.0 and sigma_j > 0.0 and use_jumps:
        jump_mean_e = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        adj_mu = drift - lambda_j * jump_mean_e
    else:
        jump_mean_e = 0.0
        adj_mu = drift

    # -------------------------------
    # 1. Generate daily price paths
    # -------------------------------
    for t in range(M):
        if t % 50 == 0:
            st.session_state._progress = min(10 + int(30 * t / M), 40)

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

    # -------------------------------
    # 2. Pathwise PV of strip (daily)
    # -------------------------------
    PV_paths = np.zeros((M + 1, paths))

    for t in range(M + 1):
        if t % 50 == 0:
            st.session_state._progress = min(50 + int(40 * t / M), 90)

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

    # -------------------------------
    # 3. Exposure statistics
    # -------------------------------
    exposure = np.maximum(0.0, PV_paths)
    pfe_95 = np.percentile(exposure, 95, axis=1)
    pfe_99 = np.percentile(exposure, 99, axis=1)
    ee = np.mean(exposure, axis=1)
    neg_95 = np.percentile(PV_paths, 5, axis=1)
    neg_99 = np.percentile(PV_paths, 1, axis=1)

    # -------------------------------
    # 4. 2‑day margin equivalents per 1,000 bbl
    # -------------------------------
    horizon_steps = 2
    max_step = M - horizon_steps

    pv_today = PV_paths[: max_step + 1, :]
    pv_future = PV_paths[horizon_steps : M + 1, :]

    pnl_2d = pv_future - pv_today

    long_loss_2d = -pnl_2d
    long_margin_2d = np.percentile(long_loss_2d, 99, axis=1)

    short_loss_2d = pnl_2d
    short_margin_2d = np.percentile(short_loss_2d, 99, axis=1)

    strip_notional_bbl = np.sum(notionals)
    per_1000_scale = 1000.0 / strip_notional_bbl if strip_notional_bbl > 0 else 0.0

    long_margin_2d_per_1000 = long_margin_2d * per_1000_scale
    short_margin_2d_per_1000 = short_margin_2d * per_1000_scale

    return {
        "time": time_pts,
        "time_days": time_days,
        "pfe_99": pfe_99,
        "pfe_95": pfe_95,
        "ee": ee,
        "neg_99": neg_99,
        "neg_95": neg_95,
        "samples": PV_paths[:, : min(10, paths)],
        "long_margin_2d_per_1000": long_margin_2d_per_1000,
        "short_margin_2d_per_1000": short_margin_2d_per_1000,
    }

# -------------------------------------------------------------
# Run button
# -------------------------------------------------------------
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True, key="run_btn"):
    st.session_state.results = None
    st.session_state._progress = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🎲 Running daily Jump-Diffusion Monte Carlo...")
    progress_bar.progress(5)

    results = run_simulation_daily(
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
    st.session_state.results = results

    progress_bar.progress(100)
    status_text.text("✅ Simulation complete")
    time.sleep(0.5)
    st.rerun()

# -------------------------------------------------------------
# Results display
# -------------------------------------------------------------
if st.session_state.results is not None:
    results = st.session_state.results

    long_2d = results["long_margin_2d_per_1000"]
    short_2d = results["short_margin_2d_per_1000"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Max PFE 99%", f"${np.max(results['pfe_99']) / 1e6:.3f}M")
    with col2:
        st.metric("Max PFE 95%", f"${np.max(results['pfe_95']) / 1e6:.3f}M")
    with col3:
        st.metric("Max EE", f"${np.max(results['ee']) / 1e6:.3f}M")
    with col4:
        st.metric("Max Liability (99%)", f"${np.abs(np.min(results['neg_99'])) / 1e6:.3f}M")
    with col5:
        st.metric("2‑Day Long Margin / 1k bbl", f"${np.max(long_2d):.2f}")
    with col6:
        st.metric("2‑Day Short Margin / 1k bbl", f"${np.max(short_2d):.2f}")

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

    # CSV download
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
    st.info("🚀 Set parameters and click **Run Simulation** to generate daily PFE, MC paths and 2‑day margin per 1,000 bbl.")
