import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("churnguard_pipeline.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background:#F5F7FA;
}

h1,h2,h3{
    color:#0F4C81;
}

[data-testid="stSidebar"]{
    background:#0F4C81;
}

[data-testid="stSidebar"] *{
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:linear-gradient(90deg,#0F4C81,#2193b0);
    color:white;
    border:none;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#2193b0,#0F4C81);
    color:white;
}

div[data-testid="stForm"]{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.15);
}

.result-card{
    padding:20px;
    border-radius:15px;
    color:white;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.markdown("""
    <div style="
    text-align:center;
    padding:15px;
    ">
    <h1>📊 ChurnGuard AI</h1>
    <p>Customer Churn Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🤖 Model")
    st.success("Logistic Regression")

    st.subheader("🎯 Accuracy")
    st.info("80.55%")

    st.subheader("👩‍💻 Developed By")
    st.write("Prathika Nethaji")

    st.markdown("---")

    st.caption("Built using Streamlit & Scikit-Learn")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style="
background:linear-gradient(90deg,#0F4C81,#2193b0);
padding:25px;
border-radius:15px;
color:white;
text-align:center;
">

<h1>📊 Customer Churn Prediction</h1>

<p>
Predict whether a telecom customer is likely to churn.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📄 Account Details")

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
    "💰 Monthly Charges (₹)",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
    "💰 Total Charges (₹)",
            min_value=0.0,
            value=1000.0
        )

    with col2:

        st.subheader("🌐 Internet Services")

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    predict = st.form_submit_button("🚀 Predict Churn")

    # -----------------------------
# Prediction
# -----------------------------
# -----------------------------
# Prediction
# -----------------------------
if predict:

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "Contract": [contract],
        "InternetService": [internet],
        "OnlineSecurity": [security],
        "TechSupport": [support],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    stay_prob = probability[list(model.classes_).index("No")]
    churn_prob = probability[list(model.classes_).index("Yes")]

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    if prediction == "Yes":

        st.markdown("""
        <div style="
        background:#ffebee;
        padding:25px;
        border-radius:15px;
        border-left:8px solid #d32f2f;
        ">
        <h2 style="color:#d32f2f;">⚠ High Risk of Churn</h2>
        <p>This customer is likely to leave the company.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:#e8f5e9;
        padding:25px;
        border-radius:15px;
        border-left:8px solid #2e7d32;
        ">
        <h2 style="color:#2e7d32;">✅ Customer is Likely to Stay</h2>
        <p>This customer has a low risk of churn.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("📊 Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:

        st.write("🟢 Stay Probability")
        st.progress(float(stay_prob))
        st.success(f"{stay_prob*100:.2f}%")

    with col2:

        st.write("🔴 Churn Probability")
        st.progress(float(churn_prob))
        st.error(f"{churn_prob*100:.2f}%")

    st.write("")

    st.subheader("🤖 Recommendation")

    if prediction == "Yes":

        st.warning("""
• Offer a personalized discount.

• Contact the customer immediately.

• Recommend a long-term contract.

• Provide premium technical support.
""")

    else:

        st.success("""
• Customer is satisfied with current services.

• Maintain service quality.

• Offer loyalty rewards.

• Recommend value-added services.
""")