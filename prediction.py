import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ============================
# Prediction function using fighter names on final_df
# ============================

def encode_method_column(df):
    if 'METHOD' in df.columns and df['METHOD'].dtype == 'object':
        df['METHOD'] = df['METHOD'].astype('category').cat.codes
    return df

def prepare_features(df):
    exclude_cols = ['Player_A', 'Player_B', 'FIGHTER', 'ROUND', 'Label', 'TIME', 'TIME FORMAT', 'REFEREE']
    features = [col for col in df.columns if col not in exclude_cols]
    return df, features, exclude_cols

def predict_fight(fighter_A, fighter_B, model_dir="models"):
    from preprocess import final_df  # Import here to ensure final_df is always available

    df = encode_method_column(final_df.copy())
    df, feature_cols, exclude_cols = prepare_features(df)

    # Extract fight rows
    fights_A = df[df['FIGHTER'] == fighter_A]
    fights_B = df[df['FIGHTER'] == fighter_B]

    if fights_A.empty or fights_B.empty:
        raise ValueError("One or both fighters not found in dataset.")

    # Generate all A-B feature diffs
    matchup_samples = []
    for _, row_A in fights_A.iterrows():
        for _, row_B in fights_B.iterrows():
            diff = row_A[feature_cols].values - row_B[feature_cols].values
            matchup_samples.append(diff)

    X_match = pd.DataFrame(matchup_samples, columns=feature_cols)

    predictions = {}

    # === XGBoost ===
    try:
        xgb_model = joblib.load(f"{model_dir}/XGBoost.pkl")
        xgb_selector = joblib.load(f"{model_dir}/XGBoost_selector.pkl")
        X_sel = xgb_selector.transform(X_match)
        probs = xgb_model.predict_proba(X_sel)[:, 1]
        avg_prob = probs.mean()
        predictions['XGBoost'] = {
            'prob_A_wins': round(avg_prob, 4),
            'prob_B_wins': round(1 - avg_prob, 4),
            'predicted_winner': fighter_A if avg_prob >= 0.5 else fighter_B
        }
    except Exception as e:
        print("XGBoost error:", e)

    # === Gradient Boosting ===
    try:
        gb_model = joblib.load(f"{model_dir}/Gradient_Boosting.pkl")
        gb_selector = joblib.load(f"{model_dir}/Gradient_Boosting_selector.pkl")
        X_sel = gb_selector.transform(X_match)
        probs = gb_model.predict_proba(X_sel)[:, 1]
        avg_prob = probs.mean()
        predictions['Gradient Boosting'] = {
            'prob_A_wins': round(avg_prob, 4),
            'prob_B_wins': round(1 - avg_prob, 4),
            'predicted_winner': fighter_A if avg_prob >= 0.5 else fighter_B
        }
    except Exception as e:
        print("Gradient Boosting error:", e)

    # === MLP ===
    try:
        mlp_model = joblib.load(f"{model_dir}/Neural_Network_(MLP).pkl")
        mlp_scaler = joblib.load(f"{model_dir}/Neural_Network_(MLP)_scaler.pkl")
        X_scaled = mlp_scaler.transform(X_match)
        probs = mlp_model.predict_proba(X_scaled)[:, 1]
        avg_prob = probs.mean()
        predictions['Neural Network (MLP)'] = {
            'prob_A_wins': round(avg_prob, 4),
            'prob_B_wins': round(1 - avg_prob, 4),
            'predicted_winner': fighter_A if avg_prob >= 0.5 else fighter_B
        }
    except Exception as e:
        print("MLP error:", e)

    print(f"\n=== Fight Prediction: {fighter_A} vs. {fighter_B} ===")
    return pd.DataFrame(predictions).T

# Example usage:
# prediction_df = predict_fight("Yair Rodriguez", "Patricio Pitbull")
# print(prediction_df)
