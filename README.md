When modeling Potential Future Exposure (PFE) for oil swaps using a jump-diffusion model, the choice of **$\lambda$ (Jump Intensity)** and **Jump Std Dev** depends on whether you are aiming for a conservative stress-test framework or a historical calibration.

Standard market-practice parameters typically fall into these ranges:

### 1. Recommended Parameter Benchmarks

| Parameter | Standard / Moderate View | Stress / Conservative View (PFE Focus) | What it means |
| --- | --- | --- | --- |
| **$\lambda$ (Lambda - Intensity)** | **1.0 to 3.0** | **3.0 to 5.0** | Average number of sharp supply/geopolitical shocks or price spikes per year. (Setting it to 3.0 means roughly 3 jumps/year). |
| **Jump Std Dev** | **0.20 to 0.35** | **0.40 to 0.60** | The magnitude volatility of those sudden shocks. Higher values model severe, fat-tailed tail events (e.g., sudden OPEC cuts or Middle East supply disruptions). |
| **Mean Jump Size** | **0.00** | **0.00** | Symmetrical jumps around zero (unless you have a strong directional bias like an expected upward/downward secular trend). |

---

### 2. How to Choose for Your PFE Model

* **For Regulatory / Standard PFE (Moderate):** Use a **$\lambda$ around 1.0 to 2.0** and a **Jump Std Dev around 0.25**. This captures occasional market dislocations without heavily over-inflating your 99% PFE tail.
* **For Stress-Testing / Credit Risk Limits (Conservative):** The settings currently visible in your simulator tool—**$\lambda = 3.0$** and **Jump Std Dev = 0.40**—are well-suited for a conservative credit risk setup. Oil markets are prone to sudden intraday or overnight geopolitical spikes, and a higher jump standard deviation ensures that your 99% PFE properly accounts for extreme tail-risk events where counterparties could face massive margin calls.

*Tip:* If you want to ground these numbers precisely in empirical data, run a maximum likelihood estimation (MLE) on your specific asset's historical daily log-returns by filtering out returns that exceed $3\sigma$ as discrete jump events.
