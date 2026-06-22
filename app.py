import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Support Ticket Analytics",
    layout="wide"
)

st.title("🎫 Customer Support Ticket Analytics Dashboard")

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    priority_model = load_model(
        "models/priority_lstm.keras"
    )

    category_model = load_model(
        "models/priority_category_lstm.keras"
    )

    rf_model = joblib.load(
        "models/resolution_time_rf.pkl"
    )

    with open(
        "models/priority_tokenizer.pkl",
        "rb"
    ) as f:
        priority_tokenizer = pickle.load(f)

    with open(
        "models/priority_encoder.pkl",
        "rb"
    ) as f:
        priority_encoder = pickle.load(f)

    with open(
        "models/category_tokenizer.pkl",
        "rb"
    ) as f:
        category_tokenizer = pickle.load(f)

    with open(
        "models/category_encoder.pkl",
        "rb"
    ) as f:
        category_encoder = pickle.load(f)

    return (
        priority_model,
        category_model,
        rf_model,
        priority_tokenizer,
        priority_encoder,
        category_tokenizer,
        category_encoder
    )


(
    priority_model,
    category_model,
    rf_model,
    priority_tokenizer,
    priority_encoder,
    category_tokenizer,
    category_encoder
) = load_models()

# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.header("📌 Project Overview")

st.write("""
This project analyzes customer support tickets using
Machine Learning and Deep Learning techniques.

### Features
- Priority Classification
- Category Classification
- Resolution Time Prediction

### Models Used
- LSTM Neural Network
- Random Forest
- Linear Regression
""")

st.info("""
Sample Inputs:

• Payment deducted twice and refund not received

• Unable to login after password reset

• Subscription cancelled but still charged

• App crashes while uploading files
""")

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🚨 Priority Prediction",
        "📂 Category Prediction",
        "⏳ Resolution Time Prediction"
    ]
)

# =====================================================
# PRIORITY PREDICTION
# =====================================================

with tab1:

    st.subheader("Priority Classification")

    priority_text = st.text_area(
        "Enter Ticket Description",
        value="Payment deducted twice and refund not received.",
        height=150
    )

    if st.button("Predict Priority"):

        seq = priority_tokenizer.texts_to_sequences(
            [priority_text]
        )

        pad = pad_sequences(
            seq,
            maxlen=100,
            padding="post"
        )

        pred = priority_model.predict(
            pad,
            verbose=0
        )

        pred_class = np.argmax(pred)

        result = priority_encoder.inverse_transform(
            [pred_class]
        )[0]

        st.success(
            f"Predicted Priority: {result}"
        )

# =====================================================
# CATEGORY PREDICTION
# =====================================================

with tab2:

    st.subheader("Category Classification")

    category_text = st.text_area(
        "Enter Ticket Description",
        value="Unable to login into account after password reset.",
        height=150
    )

    if st.button("Predict Category"):

        seq = category_tokenizer.texts_to_sequences(
            [category_text]
        )

        pad = pad_sequences(
            seq,
            maxlen=100,
            padding="post"
        )

        pred = category_model.predict(
            pad,
            verbose=0
        )

        pred_class = np.argmax(pred)

        result = category_encoder.inverse_transform(
            [pred_class]
        )[0]

        st.success(
            f"Predicted Category: {result}"
        )

# =====================================================
# RESOLUTION TIME PREDICTION
# =====================================================

with tab3:

    st.subheader("Resolution Time Prediction")

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30
    )

    customer_tenure_months = st.number_input(
        "Customer Tenure Months",
        min_value=0,
        max_value=120,
        value=24
    )

    previous_tickets = st.number_input(
        "Previous Tickets",
        min_value=0,
        max_value=100,
        value=5
    )

    customer_satisfaction_score = st.number_input(
        "Customer Satisfaction Score",
        min_value=1,
        max_value=5,
        value=4
    )

    first_response_time_hours = st.number_input(
        "First Response Time Hours",
        min_value=1.0,
        max_value=100.0,
        value=10.0
    )

    issue_complexity_score = st.number_input(
        "Issue Complexity Score",
        min_value=1,
        max_value=10,
        value=6
    )

    issue_length = st.number_input(
        "Issue Length",
        min_value=1,
        max_value=1000,
        value=120
    )

    resolution_length = st.number_input(
        "Resolution Length",
        min_value=1,
        max_value=1000,
        value=150
    )

    ticket_duration_days = st.number_input(
        "Ticket Duration Days",
        min_value=0,
        max_value=365,
        value=3
    )

    if st.button("Predict Resolution Time"):

        sample_df = pd.DataFrame(
            [[
                customer_age,
                customer_tenure_months,
                previous_tickets,
                customer_satisfaction_score,
                first_response_time_hours,
                issue_complexity_score,
                issue_length,
                resolution_length,
                ticket_duration_days
            ]],
            columns=rf_model.feature_names_in_
        )

        prediction = rf_model.predict(
            sample_df
        )[0]

        st.success(
            f"Estimated Resolution Time: {round(prediction, 2)} Hours"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Customer Support Ticket Analytics Project | GUVI Final Project"
)