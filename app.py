import streamlit as st

import pandas as pd

import joblib


# =========================================
# Page Configuration
# =========================================

st.set_page_config(
    page_title="E-commerce Demand Forecasting",
    page_icon="📦",
    layout="wide"
)


# =========================================
# Load Model
# =========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/demand_model.pkl"
    )

    return model


model = load_model()


# =========================================
# Load Dataset
# =========================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "dataset/processed_sales_data.csv"
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    return data


data = load_data()


# =========================================
# Title
# =========================================

st.title(
    "📦 E-commerce Demand Forecasting"
)


st.write(
    "Predict future product demand "
    "and get inventory recommendations."
)


# =========================================
# Sidebar
# =========================================

st.sidebar.header(
    "Product Information"
)


price = st.sidebar.number_input(
    "Product Price",
    min_value=1.0,
    value=800.0
)


promotion = st.sidebar.selectbox(
    "Promotion",
    ["No", "Yes"]
)


if promotion == "Yes":

    promotion_value = 1

else:

    promotion_value = 0


current_inventory = st.sidebar.number_input(
    "Current Inventory",
    min_value=0,
    value=25
)


# =========================================
# Forecast Button
# =========================================

if st.sidebar.button(
    "Predict Demand"
):

    # -------------------------------------
    # Get latest row
    # -------------------------------------

    latest = data.iloc[-1]


    # -------------------------------------
    # Prepare input
    # -------------------------------------

    input_data = pd.DataFrame({

        "price": [price],

        "promotion": [
            promotion_value
        ],

        "year": [
            latest["year"]
        ],

        "month": [
            latest["month"]
        ],

        "day": [
            latest["day"]
        ],

        "day_of_week": [
            latest["day_of_week"]
        ],

        "is_weekend": [
            latest["is_weekend"]
        ],

        "sales_previous_day": [
            latest["sales_previous_day"]
        ],

        "sales_previous_7_days": [
            latest["sales_previous_7_days"]
        ],

        "sales_7_day_average": [
            latest["sales_7_day_average"]
        ]

    })


    # -------------------------------------
    # Prediction
    # -------------------------------------

    prediction = model.predict(
        input_data
    )


    predicted_demand = max(
        0,
        round(prediction[0])
    )


    # =====================================
    # Display Prediction
    # =====================================

    st.subheader(
        "📊 Demand Prediction"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Demand",
            f"{predicted_demand} units"
        )


    with col2:

        st.metric(
            "Current Inventory",
            f"{current_inventory} units"
        )


    with col3:

        difference = (
            current_inventory
            - predicted_demand
        )

        st.metric(
            "Inventory Difference",
            f"{difference} units"
        )


    # =====================================
    # Inventory Recommendation
    # =====================================

    st.subheader(
        "📦 Inventory Recommendation"
    )


    if current_inventory < predicted_demand:

        restock = (
            predicted_demand
            - current_inventory
        )


        st.error(
            f"⚠️ Low Inventory!\n\n"
            f"Recommended restock: "
            f"{restock} units"
        )


    else:

        remaining = (
            current_inventory
            - predicted_demand
        )


        st.success(
            f"✅ Inventory is sufficient.\n\n"
            f"Expected remaining stock: "
            f"{remaining} units"
        )


# =========================================
# Sales History
# =========================================

st.subheader(
    "📈 Historical Sales"
)


chart_data = data[
    ["date", "sales"]
].set_index(
    "date"
)


st.line_chart(
    chart_data
)


# =========================================
# Recent Data
# =========================================

st.subheader(
    "📋 Recent Sales Data"
)


st.dataframe(
    data.tail(10),
    use_container_width=True
)