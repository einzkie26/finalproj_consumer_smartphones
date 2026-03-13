import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Optional: SHAP for accurate local feature contribution
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Purchase Intent Prediction", layout="wide")

# ---- Theme-aware colors for charts ----
def get_chart_theme():
    try:
        theme_base = st.get_option("theme.base")
    except Exception:
        theme_base = "light"

    is_dark = theme_base == "dark"

    return {
        "is_dark": is_dark,
        "text": "#EAEAEA" if is_dark else "#111827",
        "muted_text": "#C7CDD4" if is_dark else "#4B5563",
        "grid": "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.12)",
        "line": "rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.25)",
        "paper_bg": "rgba(0,0,0,0)",
        "plot_bg": "rgba(0,0,0,0)",
        "gauge_red": "rgba(239,68,68,0.20)" if is_dark else "#ffcccc",
        "gauge_yellow": "rgba(245,158,11,0.20)" if is_dark else "#fff4cc",
        "gauge_green": "rgba(34,197,94,0.20)" if is_dark else "#ccffcc",
        "placeholder_bar": "#98A2B3" if is_dark else "#d0d5dd",
        "positive": "#2ca02c",
        "negative": "#d62728",
        "fallback": "#1f77b4",
        "neutral_bar": "#6B7280" if not is_dark else "#9CA3AF",
    }

# ---- CSS: Clean UI ----
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    [data-testid="InputInstructions"],
    button[title="Clear value"],
    div[data-testid="InputEndAdornment"],
    .stNumberInput div[data-testid="InputEndAdornment"] {
        display: none !important;
    }

    [data-testid="stContainer"] {
        background-color: var(--secondary-background-color);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .status-box {
        padding: 12px 16px;
        border-radius: 12px;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .status-success {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .status-neutral {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Model/Scaler
@st.cache_resource
def load_model_assets():
    import os
    
    model_path = "fair_electronics_model.pkl"
    scaler_path = "refined_scaler.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"❌ **File Not Found:** `{model_path}`")
        st.info("📁 Please ensure the model file exists in the current directory.")
        st.code(f"Current directory: {os.getcwd()}")
        st.write("**Expected file location:**")
        st.code(os.path.abspath(model_path))
        return None, None
    
    if not os.path.exists(scaler_path):
        st.error(f"❌ **File Not Found:** `{scaler_path}`")
        st.info("📁 Please ensure the scaler file exists in the current directory.")
        st.code(f"Current directory: {os.getcwd()}")
        st.write("**Expected file location:**")
        st.code(os.path.abspath(scaler_path))
        return None, None
    
    # Try loading model
    try:
        m = joblib.load(model_path)
        st.success(f"✅ Model loaded successfully: `{model_path}`")
    except Exception as e:
        st.error(f"❌ **Error Loading File:** `{model_path}`")
        st.error(f"**Error Type:** `{type(e).__name__}`")
        st.error(f"**Error Message:** {str(e)}")
        
        with st.expander("🔍 Detailed Error Information"):
            st.write("**Full Error Traceback:**")
            import traceback
            st.code(traceback.format_exc())
            
            st.write("\n**Possible Causes:**")
            st.write("- File was created with incompatible scikit-learn/xgboost version")
            st.write("- File is corrupted or incomplete")
            st.write("- Wrong file format")
            st.write("- Missing dependencies")
            
            st.write("\n**Solutions:**")
            st.write("1. Check your scikit-learn version: `pip show scikit-learn`")
            st.write("2. Check your xgboost version: `pip show xgboost`")
            st.write("3. Recreate the model file with your current environment")
            st.write("4. Ensure all dependencies are installed")
        
        return None, None
    
    # Try loading scaler
    try:
        s = joblib.load(scaler_path)
        st.success(f"✅ Scaler loaded successfully: `{scaler_path}`")
    except Exception as e:
        st.error(f"❌ **Error Loading File:** `{scaler_path}`")
        st.error(f"**Error Type:** `{type(e).__name__}`")
        st.error(f"**Error Message:** {str(e)}")
        
        with st.expander("🔍 Detailed Error Information"):
            st.write("**Full Error Traceback:**")
            import traceback
            st.code(traceback.format_exc())
            
            st.write("\n**Possible Causes:**")
            st.write("- File was created with incompatible scikit-learn version")
            st.write("- File is corrupted or incomplete")
            st.write("- Wrong file format")
            
            st.write("\n**Solutions:**")
            st.write("1. Check your scikit-learn version: `pip show scikit-learn`")
            st.write("2. Recreate the scaler file with your current environment")
            st.write("3. Ensure scikit-learn is properly installed")
        
        return None, None
    
    return m, s

model, scaler = load_model_assets()
if model is None or scaler is None:
    st.stop()

st.title("Purchase Intent Prediction")

# ---- INPUT SECTION ----
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### User Info")
        age = st.number_input(
            "Customer Age",
            min_value=10,
            max_value=100,
            placeholder="e.g. 30",
            value=None
        )
        gender = st.selectbox(
            "Customer Gender",
            ["Female", "Male"],
            placeholder="Select a gender",
            index=None
        )

    with col2:
        st.markdown("#### Product")
        category = st.selectbox(
            "Category",
            ["Headphones", "Laptops", "Smart Watches", "Smartphones", "Tablets"],
            placeholder="Select a category",
            index=None
        )
        brand = st.selectbox(
            "Brand",
            ["Apple", "HP", "Other Brands", "Samsung", "Sony"],
            placeholder="Select a brand",
            index=None
        )

    with col3:
        st.markdown("#### Pricing")
        price = st.number_input(
            "Product Price ($)",
            min_value=0.0,
            placeholder="$ 00.00",
            value=None
        )

    with col4:
        st.markdown("#### Behavior")
        frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            placeholder="e.g. 5",
            value=None
        )
        st.write("")
        calculate_clicked = st.button("Calculate Prediction", type="primary", use_container_width=True)

# Exact feature order expected by model (NO GENDER)
model_order = [
    'ProductPrice', 'CustomerAge', 'PurchaseFrequency',
    'ProductCategory_Headphones', 'ProductCategory_Laptops',
    'ProductCategory_Smart Watches', 'ProductCategory_Smartphones',
    'ProductCategory_Tablets', 'ProductBrand_Apple', 'ProductBrand_HP',
    'ProductBrand_Other Brands', 'ProductBrand_Samsung', 'ProductBrand_Sony',
    'IsMostSoldBrand', 'Price Per Frequency'
]

# Friendly names for chart labels
friendly_names = {
    'ProductPrice': 'Product Price',
    'CustomerAge': 'Customer Age',
    'PurchaseFrequency': 'Purchase Frequency',
    'ProductCategory_Headphones': 'Category: Headphones',
    'ProductCategory_Laptops': 'Category: Laptops',
    'ProductCategory_Smart Watches': 'Category: Smart Watches',
    'ProductCategory_Smartphones': 'Category: Smartphones',
    'ProductCategory_Tablets': 'Category: Tablets',
    'ProductBrand_Apple': 'Brand: Apple',
    'ProductBrand_HP': 'Brand: HP',
    'ProductBrand_Other Brands': 'Brand: Other Brands',
    'ProductBrand_Samsung': 'Brand: Samsung',
    'ProductBrand_Sony': 'Brand: Sony',
    'IsMostSoldBrand': 'Most Sold Brand',
    'Price Per Frequency': 'Price per Frequency'
}

def create_gauge(prob=None):
    theme = get_chart_theme()
    display_value = 0 if prob is None else prob * 100

    if prob is None:
        bar_color = theme["neutral_bar"]
    else:
        bar_color = "#22C55E" if prob > 0.5 else "#EF4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_value,
            number={
                "suffix": "%",
                "font": {"size": 34}
            },
            title={
                "text": "Purchase Probability",
                "font": {"size": 22}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1
                },
                "bar": {"color": bar_color},
                "bgcolor": theme["plot_bg"],
                "borderwidth": 1,
                "bordercolor": theme["line"],
                "steps": [
                    {"range": [0, 40], "color": theme["gauge_red"]},
                    {"range": [40, 70], "color": theme["gauge_yellow"]},
                    {"range": [70, 100], "color": theme["gauge_green"]}
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": display_value
                }
            }
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

def create_placeholder_feature_chart():
    theme = get_chart_theme()

    placeholder_df = pd.DataFrame({
        "Feature": ["No data yet"],
        "Contribution": [0]
    })

    fig = go.Figure(
        go.Bar(
            x=placeholder_df["Contribution"],
            y=placeholder_df["Feature"],
            orientation="h",
            marker_color=theme["placeholder_bar"],
            text=["0.0000"],
            textposition="auto"
        )
    )

    fig.update_layout(
        title={
            "text": "Feature Contribution to This Prediction",
            "font": {"size": 20}
        },
        xaxis_title="Contribution",
        yaxis_title=None,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            zeroline=True,
            showline=True
        ),
        yaxis=dict(
            showline=False
        )
    )
    return fig

prob = None
feature_fig = create_placeholder_feature_chart()
prediction_ready = False

all_inputs_filled = all([
    age is not None,
    gender is not None,
    category is not None,
    brand is not None,
    price is not None,
    frequency is not None
])

if calculate_clicked:
    if not all_inputs_filled:
        st.warning("Please complete all fields before calculating the prediction.")
    else:
        # Build input row to match scaler and model expectations
        # Scaler expects these columns:
        scaler_cols = [
            'ProductPrice', 'CustomerAge', 'CustomerGender', 'PurchaseFrequency',
            'ProductCategory_Headphones', 'ProductCategory_Laptops', 'ProductCategory_Smart Watches',
            'ProductCategory_Smartphones', 'ProductCategory_Tablets',
            'ProductBrand_Apple', 'ProductBrand_HP', 'ProductBrand_Other Brands',
            'ProductBrand_Samsung', 'ProductBrand_Sony', 'IsMostSoldBrand', 'Price Per Frequency'
        ]
        model_cols = [
            'CustomerAge', 'ProductCategory', 'ProductPrice', 'PurchaseFrequency', 'ProductBrand', 'CustomerSatisfaction'
        ]

        # Prepare scaler input row
        scaler_row = {col: 0 for col in scaler_cols}
        scaler_row['ProductPrice'] = price
        scaler_row['CustomerAge'] = age
        scaler_row['CustomerGender'] = 1 if gender == 'Male' else 0  # Encode gender as 1/0
        scaler_row['PurchaseFrequency'] = frequency
        scaler_row['IsMostSoldBrand'] = 1 if brand in ["Apple", "Samsung"] else 0
        scaler_row['Price Per Frequency'] = price / frequency if frequency > 0 else 0
        scaler_row[f'ProductCategory_{category}'] = 1
        scaler_row[f'ProductBrand_{brand}'] = 1

        scaler_input_df = pd.DataFrame([scaler_row], columns=scaler_cols)

        try:
            scaler_input_df[scaler_cols] = scaler.transform(scaler_input_df[scaler_cols])
        except Exception as e:
            st.error(f"❌ Scaler Error: {type(e).__name__}")
            st.error(f"Details: {str(e)}")
            st.stop()

        # Prepare model input row
        model_row = {
            'CustomerAge': age,
            'ProductCategory': category,
            'ProductPrice': price,
            'PurchaseFrequency': frequency,
            'ProductBrand': brand,
            'CustomerSatisfaction': 3  # Default value, adjust as needed
        }
        model_input_df = pd.DataFrame([model_row], columns=model_cols)

        try:
            prob = model.predict_proba(model_input_df)[0][1]
            prediction_ready = True

            # Feature importance/SHAP (optional, may not work with categorical model)
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(model_input_df)

                    if isinstance(shap_values, list):
                        shap_vals = shap_values[1][0]
                    else:
                        shap_vals = shap_values[0]

                    shap_df = pd.DataFrame({
                        "Feature": [col for col in model_input_df.columns],
                        "Contribution": shap_vals
                    })

                    shap_df = shap_df[shap_df["Contribution"] != 0].copy()
                    shap_df["AbsContribution"] = shap_df["Contribution"].abs()
                    shap_df = shap_df.sort_values("AbsContribution", ascending=True).tail(10)

                    if shap_df.empty:
                        feature_fig = create_placeholder_feature_chart()
                    else:
                        theme = get_chart_theme()
                        colors = [
                            theme["positive"] if val > 0 else theme["negative"]
                            for val in shap_df["Contribution"]
                        ]

                        feature_fig = go.Figure(
                            go.Bar(
                                x=shap_df["Contribution"],
                                y=shap_df["Feature"],
                                orientation="h",
                                marker_color=colors,
                                text=[f"{v:.4f}" for v in shap_df["Contribution"]],
                                textposition="auto"
                            )
                        )

                        feature_fig.update_layout(
                            title={
                                "text": "Feature Contribution to This Prediction",
                                "font": {"size": 20}
                            },
                            xaxis_title="Contribution",
                            yaxis_title=None,
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis=dict(
                                zeroline=True,
                                showline=True
                            ),
                            yaxis=dict(
                                showline=False
                            )
                        )

                except Exception:
                    feature_fig = create_placeholder_feature_chart()
            else:
                feature_fig = create_placeholder_feature_chart()

        except Exception as e:
            st.error(f"❌ Prediction Error: {type(e).__name__}")
            st.error(f"Details: {str(e)}")
            
            with st.expander("🔍 Troubleshooting Information"):
                st.write("**Input DataFrame Info:**")
                st.write(f"- Shape: {model_input_df.shape}")
                st.write(f"- Columns: {list(model_input_df.columns)}")
                
                st.write("\n**Model Info:**")
                if hasattr(model, 'n_features_in_'):
                    st.write(f"- Expected features: {model.n_features_in_}")
                if hasattr(model, 'feature_names_in_'):
                    st.write(f"- Expected feature names: {list(model.feature_names_in_)}")
                
                st.write("\n**Scaler Info:**")
                if hasattr(scaler, 'n_features_in_'):
                    st.write(f"- Expected features: {scaler.n_features_in_}")
                if hasattr(scaler, 'feature_names_in_'):
                    st.write(f"- Expected feature names: {list(scaler.feature_names_in_)}")
                
                st.write("\n**Solution:**")
                st.write("Check your input features and model training pipeline for consistency.")

gauge_fig = create_gauge(prob)

st.divider()

if prediction_ready and prob is not None:
    st.write(f"### Purchase Probability: {prob:.2%}")

    if prob > 0.5:
        st.markdown(
            f'<div class="status-box status-success">🟢 Likely to Buy ({prob:.1%})</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="status-box status-danger">🔴 Unlikely to Buy ({prob:.1%})</div>',
            unsafe_allow_html=True
        )
else:
    st.write("### Purchase Probability:")
    st.markdown(
        '<div class="status-box status-neutral">⚪ No Prediction Yet</div>',
        unsafe_allow_html=True
    )

with st.container(border=True):
    st.markdown("### Prediction Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(
            gauge_fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with chart_col2:
        st.plotly_chart(
            feature_fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )