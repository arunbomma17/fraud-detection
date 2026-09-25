"""Streamlit Web Application: AI-Based E-Commerce Fraud Detection System.

Multi-page production dashboard featuring:
- 🏠 Dashboard: Real-time KPIs, transaction volume, and risk distribution
- 🔍 Fraud Detection: Interactive single-transaction analyzer with presets and CSV batch scanner
- 📊 Analytics: Deep-dive exploratory visualizations and latent feature correlations
- 📈 Model Performance: Rigorous comparison metrics (PR-AUC, F1, ROC-AUC), confusion matrices, and ROC/PR curves
- 🧾 Transaction History: Searchable, filterable audit ledger backed by SQLite
- ℹ️ About Project: System architecture, anti-leakage methodology, and business impact
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="AI E-Commerce Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, high-contrast, professional visual presentation
CUSTOM_CSS = """
<style>
    /* Global container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Result Banner Styling */
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .risk-banner-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid #F59E0B;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    .badge-low { background-color: #10B981; color: #FFFFFF; }
    .badge-med { background-color: #F59E0B; color: #FFFFFF; }
    .badge-high { background-color: #EF4444; color: #FFFFFF; }

    /* Tables and general elements */
    .dataframe {
        font-family: inherit !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Safe imports from core project modules
try:
    from src.config import (
        AMOUNT_COL,
        CLASSIFICATION_THRESHOLD,
        DATABASE_PATH,
        EVALUATION_RESULTS_PATH,
        FIGURES_DIR,
        MODEL_PATH,
        RAW_FEATURE_COLS,
        RISK_THRESHOLDS,
        TIME_COL,
        V_COLS,
    )
    from src.database import db
    from src.predict import determine_risk_level, predict_transaction
except Exception as e:
    st.error(f"Error importing core project modules: {e}")
    st.stop()

# Ensure database tables exist
try:
    db.init_db()
except Exception as e:
    st.sidebar.warning(f"Database initialization warning: {e}")


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_eval_results() -> Optional[Dict[str, Any]]:
    """Loads evaluation results JSON cache."""
    if EVALUATION_RESULTS_PATH.exists():
        try:
            with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_header(title: str, subtitle: str, icon: str = "🛡️") -> None:
    """Renders standardized high-impact page header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #F8FAFC;">
                {icon} {title}
            </h1>
            <p style="margin: 4px 0 0 0; font-size: 1.05rem; color: #94A3B8;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 2.5rem;">🛡️</div>
            <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #38BDF8;">FRAUDSHIELD AI</h2>
            <p style="font-size: 0.8rem; color: #64748B;">Enterprise Risk Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Fraud Detection",
            "📊 Analytics",
            "📈 Model Performance",
            "🧾 Transaction History",
            "ℹ️ About Project",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### System Health")

    model_ready = MODEL_PATH.exists()
    db_ready = DATABASE_PATH.exists()

    st.markdown(
        f"""
        * **ML Engine**: {'🟢 Online' if model_ready else '🔴 Not Trained'}
        * **Audit Database**: {'🟢 Connected' if db_ready else '🟡 Offline'}
        * **Decision Threshold**: `0.50`
        * **Risk Policy**: `0-30% | 30-70% | 70-100%`
        """
    )

    if not model_ready:
        st.warning("Model file not found. Run `python -m src.train` to train models.")


# -----------------------------------------------------------------------------
# PAGE 1: 🏠 DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏠 Dashboard":
    render_header("Executive Fraud Operations Dashboard", "Real-time surveillance of monitored transaction streams and threat KPIs")

    stats = db.get_transaction_stats()

    # Top KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Audited Transactions</div>
                <div class="metric-value">{stats['total_transactions']:,}</div>
                <div class="metric-sub">Total logged events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Fraudulent Detected</div>
                <div class="metric-value" style="color: #EF4444;">{stats['fraud_transactions']:,}</div>
                <div class="metric-sub">{stats['fraud_rate']}% of volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Genuine Cleared</div>
                <div class="metric-value" style="color: #10B981;">{stats['genuine_transactions']:,}</div>
                <div class="metric-sub">Low friction approvals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Threat Probability</div>
                <div class="metric-value" style="color: #F59E0B;">{stats['avg_fraud_probability']}%</div>
                <div class="metric-sub">Population risk index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Monitored Volume</div>
                <div class="metric-value" style="color: #38BDF8;">₹{stats['total_amount_audited']:,.2f}</div>
                <div class="metric-sub">Protected financial assets</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization row
    chart_col1, chart_col2 = st.columns([1.2, 1])

    with chart_col1:
        st.markdown("### 📊 Threat Severity Breakdown")
        risk_data = pd.DataFrame({
            "Risk Tier": ["Low Risk (0-30%)", "Medium Risk (30-70%)", "High Risk (70-100%)"],
            "Count": [stats["low_risk_count"], stats["medium_risk_count"], stats["high_risk_count"]],
        })
        st.bar_chart(risk_data.set_index("Risk Tier"), color="#3B82F6", height=280)

    with chart_col2:
        st.markdown("### ⚡ Quick Status Summary")
        if stats["total_transactions"] == 0:
            st.info("No live transactions audited yet. Navigate to '🔍 Fraud Detection' to analyze your first transaction or load sample presets.")
        else:
            summary_df = pd.DataFrame([
                {"Category": "Total Scanned", "Value": str(stats["total_transactions"])},
                {"Category": "Fraud Attacks Blocked", "Value": str(stats["fraud_transactions"])},
                {"Category": "Clean Purchases Approved", "Value": str(stats["genuine_transactions"])},
                {"Category": "High-Risk Escalations", "Value": str(stats["high_risk_count"])},
            ])
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🕒 Recent Live Transactions Feed")
    recent = db.get_recent_transactions(limit=10)
    if recent:
        df_recent = pd.DataFrame(recent)
        st.dataframe(
            df_recent[["id", "created_at", "transaction_amount", "prediction", "fraud_probability", "risk_level"]],
            column_config={
                "transaction_amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                "fraud_probability": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Transaction history is currently empty.")


# -----------------------------------------------------------------------------
# PAGE 2: 🔍 FRAUD DETECTION (TRANSACTION ANALYZER)
# -----------------------------------------------------------------------------
elif page == "🔍 Fraud Detection":
    render_header("Interactive Transaction Threat Analyzer", "Test single transactions or upload bulk batches for instant ML risk scoring")

    tab1, tab2 = st.tabs(["🧪 Single Transaction Inspector", "📁 Batch CSV Scanner"])

    with tab1:
        st.markdown("#### Scenario Simulation Presets")
        preset_cols = st.columns(4)

        # Initialize session state feature values if not present
        if "sim_amount" not in st.session_state:
            st.session_state.sim_amount = 2499.00
        if "sim_time" not in st.session_state:
            st.session_state.sim_time = 14200.0
        if "sim_v_values" not in st.session_state:
            st.session_state.sim_v_values = {col: 0.0 for col in V_COLS}

        with preset_cols[0]:
            if st.button("🟢 Typical Genuine Purchase (₹2,499)", use_container_width=True):
                st.session_state.sim_amount = 2499.00
                st.session_state.sim_time = 15000.0
                st.session_state.sim_v_values = {col: 0.05 for col in V_COLS}
                st.session_state.sim_v_values["V14"] = 0.2
                st.session_state.sim_v_values["V12"] = 0.1
                st.session_state.sim_v_values["V4"] = -0.1
                st.rerun()

        with preset_cols[1]:
            if st.button("🔴 Critical Fraud Attack (₹75,000)", use_container_width=True):
                st.session_state.sim_amount = 75000.00
                st.session_state.sim_time = 78500.0
                st.session_state.sim_v_values = {col: 0.0 for col in V_COLS}
                st.session_state.sim_v_values["V14"] = -6.50
                st.session_state.sim_v_values["V12"] = -4.80
                st.session_state.sim_v_values["V17"] = -4.20
                st.session_state.sim_v_values["V4"] = 4.30
                st.session_state.sim_v_values["V11"] = 3.60
                st.rerun()

        with preset_cols[2]:
            if st.button("🟠 Suspicious Card-Testing (₹1.00)", use_container_width=True):
                st.session_state.sim_amount = 1.00
                st.session_state.sim_time = 3200.0
                st.session_state.sim_v_values = {col: 0.0 for col in V_COLS}
                st.session_state.sim_v_values["V14"] = -3.20
                st.session_state.sim_v_values["V4"] = 2.90
                st.rerun()

        with preset_cols[3]:
            if st.button("🔄 Reset to Neutral (₹1,500)", use_container_width=True):
                st.session_state.sim_amount = 1500.00
                st.session_state.sim_time = 0.0
                st.session_state.sim_v_values = {col: 0.0 for col in V_COLS}
                st.rerun()

        st.markdown("---")

        with st.form("transaction_form"):
            st.markdown("### Transaction Parameters")
            form_col1, form_col2 = st.columns(2)

            with form_col1:
                input_amount = st.number_input(
                    "Transaction Amount (₹ INR)",
                    min_value=0.01,
                    max_value=5000000.0,
                    value=float(st.session_state.sim_amount),
                    step=50.0,
                    format="%.2f",
                    help="Transaction monetary value in Indian Rupees (INR)",
                )

            with form_col2:
                input_time = st.number_input(
                    "Transaction Elapsed Time (seconds)",
                    min_value=0.0,
                    max_value=172800.0,
                    value=float(st.session_state.sim_time),
                    step=300.0,
                    help="Seconds elapsed relative to benchmark zero point (up to 48 hours / 172,800s)",
                )

            with st.expander("🛠️ Advanced PCA Latent Vectors (V1 – V28)", expanded=False):
                st.caption("Principal Component Analysis latent feature space (anonymized user behavior dimensions).")
                v_cols_grid = st.columns(4)
                v_inputs = {}
                for idx, v_name in enumerate(V_COLS):
                    col_idx = idx % 4
                    with v_cols_grid[col_idx]:
                        default_val = float(st.session_state.sim_v_values.get(v_name, 0.0))
                        v_inputs[v_name] = st.number_input(
                            v_name,
                            value=default_val,
                            format="%.3f",
                            key=f"input_{v_name}",
                        )

            submit_btn = st.form_submit_button("⚡ ANALYZE TRANSACTION", use_container_width=True, type="primary")

        if submit_btn:
            # Build transaction dictionary
            payload = {
                AMOUNT_COL: input_amount,
                TIME_COL: input_time,
                **v_inputs,
            }

            with st.spinner("Processing transaction through feature engineering and calibrated model..."):
                try:
                    result = predict_transaction(payload, log_to_db=True)
                except Exception as exc:
                    st.error(f"Inference error: {exc}")
                    result = None

            if result:
                st.markdown("---")
                pred = result["prediction"]
                prob = result["fraud_probability"]
                risk = result["risk_level"]
                pct_str = result["fraud_percentage"]

                if risk == "LOW":
                    banner_class = "risk-banner-low"
                    badge_class = "badge-low"
                    icon = "✅"
                    headline = "GENUINE TRANSACTION APPROVED"
                elif risk == "MEDIUM":
                    banner_class = "risk-banner-med"
                    badge_class = "badge-med"
                    icon = "⚠️"
                    headline = "SUSPICIOUS PATTERN (STEP-UP AUTHENTICATION RECOMMENDED)"
                else:
                    banner_class = "risk-banner-high"
                    badge_class = "badge-high"
                    icon = "🚨"
                    headline = "HIGH-RISK FRAUD DETECTED (TRANSACTION BLOCKED)"

                st.markdown(
                    f"""
                    <div class="{banner_class}">
                        <div style="font-size: 2.2rem; margin-bottom: 8px;">{icon}</div>
                        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800;">{headline}</h2>
                        <div style="margin: 12px 0;">
                            <span class="risk-badge {badge_class}">RISK LEVEL: {risk}</span>
                        </div>
                        <div style="font-size: 2.5rem; font-weight: 800; margin: 8px 0;">
                            {pct_str}
                        </div>
                        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">
                            Model: {result['model_name']} | Decision Threshold: {result['threshold']} | Audit ID: #{result['audit_id']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.markdown("#### 🔬 Threat Risk Gauge")
                    st.progress(prob, text=f"Fraud Probability: {pct_str}")
                    st.markdown(
                        f"""
                        * **Genuine Confidence**: `{result['genuine_probability'] * 100:.2f}%`
                        * **Fraud Confidence**: `{pct_str}`
                        * **Transaction Value**: `₹{result['transaction_amount']:,.2f}`
                        * **Logged to Database**: `ID #{result['audit_id']}`
                        """
                    )

                with res_col2:
                    st.markdown("#### 🔎 Behavioral Risk Indicators")
                    if result.get("risk_factors"):
                        for rf in result["risk_factors"]:
                            st.markdown(f"- 🔸 {rf}")
                    else:
                        st.write("No severe anomalies triggered.")

    with tab2:
        st.markdown("### 📁 Bulk Transaction CSV Scanner")
        st.markdown("Upload a CSV file containing transactions (`Time`, `Amount`, and optional `V1-V28`) to generate risk scores for the entire batch.")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {len(batch_df):,} transactions successfully!")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Fraud Analysis", type="primary"):
                    with st.spinner("Analyzing batch..."):
                        from src.predict import get_predictor
                        predictor = get_predictor()
                        scored_df = predictor.predict_batch(batch_df, log_to_db=False)

                        b_fraud_count = int((scored_df["prediction"] == "Fraud").sum())
                        b_total = len(scored_df)
                        b_rate = (b_fraud_count / b_total) * 100

                        st.markdown("---")
                        b_col1, b_col2, b_col3 = st.columns(3)
                        b_col1.metric("Total Batch Records", f"{b_total:,}")
                        b_col2.metric("Fraud Detected", f"{b_fraud_count:,}", delta=f"{b_rate:.2f}% rate", delta_color="inverse")
                        b_col3.metric("Clean Records", f"{b_total - b_fraud_count:,}")

                        st.dataframe(
                            scored_df[["prediction", "risk_level", "fraud_probability"] + [c for c in scored_df.columns if c not in ["prediction", "risk_level", "fraud_probability"]]],
                            use_container_width=True,
                        )

                        csv_data = scored_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Scored Batch CSV",
                            data=csv_data,
                            file_name="fraud_scored_transactions.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Error reading or processing CSV: {e}")


# -----------------------------------------------------------------------------
# PAGE 3: 📊 ANALYTICS
# -----------------------------------------------------------------------------
elif page == "📊 Analytics":
    render_header("Exploratory Data & Anomaly Analytics", "Statistical distributions, correlation structures, and PCA latent patterns")

    fig_dir = FIGURES_DIR

    tab_dist, tab_corr, tab_latent = st.tabs(["📈 Distributions", "🔥 Feature Correlations", "🧬 Latent Vectors"])

    with tab_dist:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### Severe Class Imbalance Profile")
            class_fig = fig_dir / "class_distribution.png"
            if class_fig.exists():
                st.image(str(class_fig), use_container_width=True)
            else:
                st.info("Class distribution plot not generated yet.")

        with col_d2:
            st.markdown("#### Transaction Amount Distribution")
            amount_fig = fig_dir / "amount_distribution.png"
            if amount_fig.exists():
                st.image(str(amount_fig), use_container_width=True)
            else:
                st.info("Amount distribution plot not generated yet.")

        st.markdown("#### Diurnal Time Rhythm (48-Hour Cycle)")
        time_fig = fig_dir / "time_distribution.png"
        if time_fig.exists():
            st.image(str(time_fig), use_container_width=True)

    with tab_corr:
        st.markdown("#### Correlation of Features with Fraud (Class)")
        st.caption("Key latent features manifest strong positive and negative correlations with fraudulent activity.")
        corr_fig = fig_dir / "correlation_heatmap.png"
        if corr_fig.exists():
            st.image(str(corr_fig), use_container_width=True)
        else:
            st.info("Correlation figure not generated yet.")

    with tab_latent:
        st.markdown("#### Critical Latent Feature Distributions (V14, V12, V17, V4)")
        st.caption("Distribution boxplots contrasting genuine vs fraudulent transaction behaviors.")
        feat_fig = fig_dir / "feature_distributions.png"
        if feat_fig.exists():
            st.image(str(feat_fig), use_container_width=True)
        else:
            st.info("Feature distributions figure not generated yet.")


# -----------------------------------------------------------------------------
# PAGE 4: 📈 MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "📈 Model Performance":
    render_header("Machine Learning Model Evaluation & Benchmarks", "Programmatic validation strictly evaluated on holdout test partition")

    eval_data = load_eval_results()

    if eval_data:
        st.markdown(
            f"""
            <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
                <span style="font-weight: 700; color: #38BDF8;">Selected Best Model:</span>
                <span style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC;"> {eval_data.get('best_model_name', 'Trained Estimator')}</span>
                <span style="color: #94A3B8; margin-left: 12px;">Optimized for PR-AUC & Recall under severe class imbalance</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🏆 Algorithm Comparison Benchmark")
        comp_records = eval_data.get("model_comparison", [])
        if comp_records:
            comp_df = pd.DataFrame(comp_records)
            st.dataframe(
                comp_df,
                column_config={
                    "Precision": st.column_config.NumberColumn(format="%.4f"),
                    "Recall": st.column_config.NumberColumn(format="%.4f"),
                    "F1 Score": st.column_config.NumberColumn(format="%.4f"),
                    "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
                    "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                    "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                },
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Best Model Holdout Test Metrics")
        metrics = eval_data.get("metrics", {})
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

        m_col1.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        m_col2.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        m_col3.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        m_col4.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        m_col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        m_col6.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Diagnostic Evaluation Curves")
        c1, c2, c3 = st.columns(3)

        fig_paths = eval_data.get("figures", {})
        with c1:
            st.markdown("#### Confusion Matrix")
            cm_p = fig_paths.get("confusion_matrix")
            if cm_p and Path(cm_p).exists():
                st.image(cm_p, use_container_width=True)

        with c2:
            st.markdown("#### Precision-Recall Curve")
            pr_p = fig_paths.get("precision_recall_curve")
            if pr_p and Path(pr_p).exists():
                st.image(pr_p, use_container_width=True)

        with c3:
            st.markdown("#### ROC Curve")
            roc_p = fig_paths.get("roc_curve")
            if roc_p and Path(roc_p).exists():
                st.image(roc_p, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Model Comparison Chart")
        comp_p = fig_paths.get("model_comparison")
        if comp_p and Path(comp_p).exists():
            st.image(comp_p, use_container_width=True)

        st.markdown("---")
        st.markdown("### 💡 Why Precision-Recall & PR-AUC Matter Over Accuracy")
        st.markdown(
            """
            > **The Accuracy Paradox in Fraud Detection:**  
            > In a fraud dataset where 99.83% of transactions are genuine and only 0.17% are fraudulent, a naive classifier that predicts *everything* as genuine achieves an impressive **99.83% Accuracy**, yet catches **0% of fraud** (Recall = 0.0).  
            >  
            > In production fraud engineering:
            > - **PR-AUC (Average Precision)** focuses strictly on the minority fraudulent class without being inflated by true negatives.
            > - **Recall** ensures financial loss is minimized by capturing illicit transactions.
            > - **Precision** protects customer satisfaction by keeping false alarm friction low.
            """
        )
    else:
        st.warning("Evaluation results file not found. Please train models by running `python -m src.train`.")


# -----------------------------------------------------------------------------
# PAGE 5: 🧾 TRANSACTION HISTORY
# -----------------------------------------------------------------------------
elif page == "🧾 Transaction History":
    render_header("Audited Transaction Ledger", "Historical inspection, risk level filtering, and CSV export via SQLite")

    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns([1, 1, 1, 1])

    with filt_col1:
        pred_filter = st.selectbox("Filter Prediction", ["All", "Fraud", "Genuine"], index=0)

    with filt_col2:
        risk_filter = st.selectbox("Filter Risk Level", ["All", "LOW", "MEDIUM", "HIGH"], index=0)

    with filt_col3:
        min_amount = st.number_input("Min Amount (₹)", min_value=0.0, value=0.0, step=100.0)

    with filt_col4:
        limit = st.selectbox("Max Records", [50, 100, 250, 500], index=1)

    records = db.get_recent_transactions(limit=limit, risk_filter=risk_filter, prediction_filter=pred_filter)

    if records:
        df_hist = pd.DataFrame(records)
        if min_amount > 0.0:
            df_hist = df_hist[df_hist["transaction_amount"] >= min_amount]

        st.markdown(f"**Showing {len(df_hist)} matching audit records:**")

        st.dataframe(
            df_hist[["id", "created_at", "transaction_amount", "prediction", "fraud_probability", "risk_level"]],
            column_config={
                "id": st.column_config.NumberColumn("ID", format="#%d"),
                "transaction_amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                "fraud_probability": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
        )

        csv_export = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Filtered History to CSV",
            data=csv_export,
            file_name="transaction_audit_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No transaction records match the current filter criteria.")


# -----------------------------------------------------------------------------
# PAGE 6: ℹ️ ABOUT PROJECT
# -----------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    render_header("Architecture & Engineering Methodology", "Comprehensive technical design documentation and production practices")

    st.markdown(
        """
        ### 🎯 Problem Statement
        In modern electronic commerce, fraudulent transactions account for tens of billions of dollars in global annual losses. 
        Detecting fraudulent transactions in real-time requires balancing two critical engineering objectives:
        1. **High Recall**: Intercepting malicious actors to prevent financial loss and chargeback fees.
        2. **High Precision**: Avoiding false-positive friction that frustrates legitimate cardholders.

        ---

        ### 🏗️ System Architecture
        """
    )

    st.markdown(
        """
        ```
        +-------------------------------------------------------------+
        |                 RAW TRANSACTION INGESTION                   |
        |             (Time, Amount, V1-V28 PCA Features)             |
        +------------------------------+------------------------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |               LEAKAGE-FREE PREPROCESSING                    |
        |    - Stratified Train-Test Splitting (80/20)                |
        |    - RobustScaler fitted strictly on training data          |
        |    - Engineered features (log_amount, hour_of_day, ratios)  |
        +------------------------------+------------------------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |             CLASS IMBALANCE HANDLING (SMOTE)                |
        |   - Synthetic Minority Over-sampling on Train split only     |
        |   - Algorithmic Class Weight Calibration (scale_pos_weight) |
        +------------------------------+------------------------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |             MODEL BENCHMARKING & TUNING                     |
        |   - Logistic Regression vs Random Forest vs XGBoost         |
        |   - Stratified K-Fold CV optimizing PR-AUC & F1-Score       |
        +------------------------------+------------------------------+
                                       |
                                       v
        +-------------------------------------------------------------+
        |                  PRODUCTION DEPLOYMENT                      |
        |   - Streamlit Multi-Page Interactive Web Application        |
        |   - SQLite Audit Storage & Historical Transaction Ledger    |
        +-------------------------------------------------------------+
        ```
        """
    )

    st.markdown(
        """
        ### 🛡️ Core Data Science & ML Principles Applied
        1. **Zero Data Leakage**: Scalers and feature transformers are fitted exclusively on the training split.
        2. **SMOTE Discipline**: Synthetic oversampling is never applied to the holdout test set.
        3. **Metric Alignment**: Optimization is anchored on **PR-AUC (Average Precision)** and **F1-Score** rather than raw accuracy.
        4. **Audit Trail**: Every analyzed transaction is recorded in SQLite for real-time monitoring and compliance.
        5. **Calibrated Explainability**: Predictions provide heuristic flags highlighting the specific latent dimensions triggering alerts.

        ---

        ### 👨‍💻 Developer & Portfolio
        * **Project**: AI-Based E-Commerce Fraud Detection System
        * **Technology Stack**: Python, Scikit-learn, XGBoost, imbalanced-learn, Streamlit, SQLite, SQLAlchemy, Pytest
        * **Repository**: Production Machine Learning & Data Science Portfolio
        """
    )
