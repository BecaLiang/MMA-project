import os
import json
import traceback
import pandas as pd
import streamlit as st
import joblib

from preprocess import run_preprocessing

st.set_page_config(page_title="UFC Fight Predictor", layout="centered")
st.title("UFC Fight Outcome Predictor")
st.markdown("Select two fighters to simulate a matchup and see win probabilities using top ML models.")

# ====================
# Step 1: Preprocess Data
# ====================
try:
    final_df = run_preprocessing()
except Exception as e:
    st.error("Failed to preprocess data. Please check preprocess.py.")
    st.code(traceback.format_exc())
    st.stop()

# ====================
# Step 2: Load Top 3 Models (hidden)
# ====================
try:
    with open("models/top_models.json", "r") as f:
        top_model_names = json.load(f)
    top_models_df = pd.DataFrame({"Model": top_model_names})
except Exception as e:
    st.error("Failed to load saved models. Run model training first.")
    st.code(traceback.format_exc())
    st.stop()

# ====================
# Step 3: Fighter Selection
# ====================
fighter_list = sorted(final_df['FIGHTER'].dropna().unique())
fighter_a = st.selectbox("Select Fighter A", options=fighter_list, index=0)
fighter_b = st.selectbox("Select Fighter B", options=fighter_list, index=1)

# ====================
# Step 4: Prediction Logic
# ====================
def predict_fight(fighter_a, fighter_b, final_df, top_models_df):
    feature_cols = [col for col in final_df.columns if col not in [
        'Player_A', 'Player_B', 'FIGHTER', 'ROUND', 'Label', 'TIME', 'TIME FORMAT', 'REFEREE', 'METHOD'
    ]]

    fights_A = final_df[final_df['FIGHTER'] == fighter_a]
    fights_B = final_df[final_df['FIGHTER'] == fighter_b]

    if fights_A.empty or fights_B.empty:
        raise ValueError(f"One or both fighters not found: {fighter_a}, {fighter_b}")

    matchup_samples = []
    for _, row_A in fights_A.iterrows():
        for _, row_B in fights_B.iterrows():
            diff = row_A[feature_cols].values - row_B[feature_cols].values
            matchup_samples.append(diff)

    X_match = pd.DataFrame(matchup_samples, columns=feature_cols)
    predictions = {}

    for _, row in top_models_df.iterrows():
        model_name = row['Model']
        try:
            model = joblib.load(f"models/{model_name}.pkl")
            selector_path = f"models/{model_name}_selector.pkl"
            scaler_path = f"models/{model_name}_scaler.pkl"

            # Feature selection
            X_transformed = X_match
            if os.path.exists(selector_path):
                selector = joblib.load(selector_path)
                X_transformed = selector.transform(X_transformed)

            # Scaling
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                X_transformed = scaler.transform(X_transformed)

            # Predict probability
            probs = model.predict_proba(X_transformed)[:, 1]
            avg_prob = probs.mean()

            predictions[model_name] = {
                'prob_A_wins': round(avg_prob, 4),
                'prob_B_wins': round(1 - avg_prob, 4),
                'predicted_winner': fighter_a if avg_prob >= 0.5 else fighter_b
            }

        except Exception as e:
            predictions[model_name] = {"error": str(e)}

    return pd.DataFrame(predictions).T

# ====================
# Step 5: Prediction Button
# ====================
if st.button("Predict Fight Outcome"):
    with st.spinner("🤖 Simulating fight..."):
        try:
            prediction_df = predict_fight(fighter_a, fighter_b, final_df, top_models_df)
            st.success(f"✅ Prediction Complete: {fighter_a} vs {fighter_b}")
            st.markdown("### 🧠 Prediction Results")

            # === NEW: Split numeric vs non-numeric
            numeric_cols = prediction_df.select_dtypes(include='number').columns.tolist()
            non_numeric_cols = [col for col in prediction_df.columns if col not in numeric_cols]

            # === NEW: Apply styling ONLY to numeric columns
            if numeric_cols:
                styled = prediction_df.style.highlight_max(axis=1, subset=numeric_cols)
                st.dataframe(styled)
            else:
                st.dataframe(prediction_df)

            # === Optional: show predicted winners separately
            if "predicted_winner" in prediction_df.columns:
                st.markdown("### Model Predictions")
                for model, row in prediction_df.iterrows():
                    if "predicted_winner" in row:
                        st.markdown(f"**{model}** predicts: `{row['predicted_winner']}`")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(traceback.format_exc())