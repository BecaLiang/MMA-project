import streamlit as st
from prediction import predict_fight

st.set_page_config(page_title="UFC Fight Predictor", layout="centered")
st.title("🥊 UFC Fight Outcome Predictor")
st.markdown("Select two fighters to simulate a matchup and see win probabilities from top ML models.")

# Fighter input fields
fighter_a = st.text_input("Enter Fighter A Name", placeholder="e.g., Jon Jones")
fighter_b = st.text_input("Enter Fighter B Name", placeholder="e.g., Stipe Miocic")

# Predict button
if st.button("Predict Fight Outcome"):
    with st.spinner("Predicting fight outcome using top ML models..."):
        try:
            prediction_df = predict_fight(fighter_a, fighter_b)
            st.success(f"Prediction Complete: {fighter_a} vs {fighter_b}")
            st.dataframe(prediction_df.style.highlight_max(axis=1))
        except Exception as e:
            st.error("An unexpected error occurred. Here is the traceback:")
            st.code(traceback.format_exc(), language='python')  # show full traceback in app
