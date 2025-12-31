# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import google.generativeai as genai

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ===================== MOBILE VIEW TOGGLE =====================
if "mobile_view" not in st.session_state:
    st.session_state.mobile_view = False

col_a, col_b = st.columns([8, 2])
with col_b:
    st.session_state.mobile_view = st.toggle("📱 Mobile View")

# ===================== GEMINI CLIENT =====================

# Configure the API key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return joblib.load("car_price_lr_correct.pkl")

model = load_model()


# ===================== HELPER FUNCTIONS =====================
def calculate_future_prices(base_input, year, model):
    years, prices = [], []
    for i in range(6):
        temp = base_input.copy()
        temp["Year"] = year + i
        price = model.predict(pd.DataFrame([temp]))[0]
        years.append(year + i)
        prices.append(round(price, 2))
    return years, prices


def explain_with_gemini(input_data, predicted_price, years, prices):
    prompt = f"""
You are a car valuation expert.
Give ONLY 4–5 short lines.

Car Details: {input_data}
Predicted Price: {predicted_price} Lakhs
Future Prices: {list(zip(years, prices))}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return "\n".join(response.text.strip().split("\n")[:5])


def save_depreciation_graph(years, prices):
    path = "depreciation_graph.png"
    plt.figure(figsize=(5, 3))
    plt.plot(years, prices, marker="o")
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def generate_pdf(car_data, price, explanation, graph_path):
    pdf_path = "car_price_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Car Price Prediction Report</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    for k, v in car_data.items():
        content.append(Paragraph(f"<b>{k}</b>: {v}", styles["Normal"]))

    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Predicted Price:</b> ₹ {price} Lakhs", styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Image(graph_path, width=400, height=250))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>AI Explanation:</b><br/>{explanation}", styles["Normal"]))

    doc.build(content)
    return pdf_path


# ===================== SIDEBAR (DESKTOP ONLY) =====================
if not st.session_state.mobile_view:
    st.sidebar.title("⚙️ Features")
    feature = st.sidebar.radio(
        "Select Feature",
        [
            "Price Prediction",
            "Depreciation Graph",
            "Best Time to Sell",
            "AI Explanation",
            "Download PDF Report"
        ]
    )
else:
    feature = None

# ===================== TITLE =====================
if st.session_state.mobile_view:
    st.markdown(
        """
        <h4 style='text-align:center; margin-bottom:10px;'>
            🚗 Car Price Prediction System
        </h4>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("🚗 Car Price Prediction System")

# ===================== MAIN INPUT =====================


company = st.text_input("Company", "Maruti")
year = st.number_input("Year", 1995, 2025, 2014)
present_price = st.number_input("Present Price (Lakhs)", 0.0, 100.0, 5.59)
kms_driven = st.number_input("KMs Driven", 0, 500000, 27000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 2, 3])

c1, c2, c3 = st.columns([3, 2, 3])
with c2:
    predict_btn = st.button("🔮 Predict Car Price")


# ===================== DATA =====================
base_input = {
    "company": company,
    "Year": year,
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Fuel_Type": fuel_type,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Owner": owner
}

input_df = pd.DataFrame([base_input])
predicted_price = round(model.predict(input_df)[0], 2)
future_years, future_prices = calculate_future_prices(base_input, year, model)


# ===================== MOBILE FLOW =====================
if st.session_state.mobile_view:
    if predict_btn:
        st.success(f"💰 ₹ {predicted_price} Lakhs")

        st.subheader("📉 Depreciation Trend")
        plt.figure(figsize=(4, 2.5))
        plt.plot(future_years, future_prices, marker="o")
        plt.grid(True)
        st.pyplot(plt, use_container_width=False)

        idx = future_prices.index(min(future_prices))
        st.info(f"📌 Sell before {future_years[idx]}")

        with st.spinner("🧠 Gemini is analyzing..."):
            explanation = explain_with_gemini(
                base_input, predicted_price, future_years, future_prices
            )
        st.write(explanation)

        graph_path = save_depreciation_graph(future_years, future_prices)
        pdf_path = generate_pdf(base_input, predicted_price, explanation, graph_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report", f)

    st.stop()


# ===================== DESKTOP FEATURES =====================
if feature == "Price Prediction":
    st.success(f"₹ {predicted_price} Lakhs")

elif feature == "Depreciation Graph":
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        plt.figure(figsize=(5, 3))
        plt.plot(future_years, future_prices, marker="o")
        plt.grid(True)
        st.pyplot(plt, use_container_width=False)

elif feature == "Best Time to Sell":
    idx = future_prices.index(min(future_prices))
    st.info(f"Sell before {future_years[idx]}")

elif feature == "AI Explanation":
    explanation = explain_with_gemini(
        base_input, predicted_price, future_years, future_prices
    )
    st.write(explanation)

elif feature == "Download PDF Report":
    explanation = explain_with_gemini(
        base_input, predicted_price, future_years, future_prices
    )
    graph_path = save_depreciation_graph(future_years, future_prices)
    pdf_path = generate_pdf(base_input, predicted_price, explanation, graph_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Report", f)
