
<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue.svg"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://github.com/emilio027/Analysis-of-Financial-Instruments-in-periods-of-Hyper-Inflation/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/emilio027/Analysis-of-Financial-Instruments-in-periods-of-Hyper-Inflation/ci.yml?label=CI"></a>
  <a href="https://github.com/emilio027/Analysis-of-Financial-Instruments-in-periods-of-Hyper-Inflation/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/emilio027/Analysis-of-Financial-Instruments-in-periods-of-Hyper-Inflation"></a>
  <a href="#"><img alt="Code style" src="https://img.shields.io/badge/style-ruff-informational"></a>
</p>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilio027/Analysis-of-Financial-Instruments-in-periods-of-Hyper-Inflation/blob/main/notebooks/quickstart.ipynb)


![Preview](docs/img/preview.png)

    # Financial Instruments in Hyper-Inflation — Real Returns & Stress Windows

    ## Executive Summary
    Evidence-based analysis of assets (equities, commodities, FX, alternatives) under inflation spikes.
Computes real return metrics, stress windows, and comparative drawdowns with bootstrap CIs and
transparent assumptions.

    **ATS Keywords:** Python, SQL, Power BI, Tableau, Pandas, NumPy, scikit-learn, ETL, data pipeline, automation, business intelligence, KPI dashboard, predictive modeling, time series forecasting, feature engineering, stakeholder management, AWS, GitHub Actions, Streamlit, Prophet, SARIMAX, SHAP, risk analytics, calibration, cross-validation, A/B testing

    ## Skills & Tools
    - Python
- Pandas/NumPy
- Matplotlib/Plotly
- Event studies
- Bootstrap confidence intervals

    ## Deliverables
    - Inflation-episode identification and real-returns calculation
- Comparative plots across assets and regimes
- Report with allocation sketches and risk caveats

    ## Key Metrics / Evaluation
    - Real returns
- Volatility
- Max drawdown
- Sharpe (real)

    ## How to Run
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    make data
    make report
    ```
    *Law Firm demo:* `streamlit run app.py`

    ## Impact Highlights (from my work history)
    - Saved $3M by automating workflows and migrating Excel processes to SAP HANA at NRG
- Resolved data issues saving $500k annually at CenterPoint Energy
- Improved stakeholder transparency by 15% via SQL + Power BI/Tableau dashboards at Robin Hood
- Scaled an AI automation agency from $750 to $28k weekly revenue as Founder/CEO

    ## Repo Structure
    ```
    src/  notebooks/  data/{raw,processed}  models/  scripts/  tests/  docs/img/  reports/
    ```
