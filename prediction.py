import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

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

def predict_fight(fighter_a, fighter_b, final_df):
    """
    Predict the outcome of a fight between two fighters and return predictions and model metrics.
    Arguments:
        fighter_a (str): Name of Fighter A.
        fighter_b (str): Name of Fighter B.
        final_df (DataFrame): Preprocessed dataset.
    Returns:
        prediction_df (DataFrame): Prediction results.
        metrics_df (DataFrame): Model performance metrics.
    """
    df = encode_method_column(final_df.copy())
    df, feature_cols, exclude_cols = prepare_features(df)

    # Extract fight rows
    fights_A = df[df['FIGHTER'] == fighter_a]
    fights_B = df[df['FIGHTER'] == fighter_b]

    if fights_A.empty or fights_B.empty:
        raise ValueError(f"One or both fighters not found in dataset: {fighter_a}, {fighter_b}")

    # Generate all A-B feature diffs
    matchup_samples = []
    for _, row_A in fights_A.iterrows():
        for _, row_B in fights_B.iterrows():
            diff = row_A[feature_cols].values - row_B[feature_cols].values
            matchup_samples.append(diff)

    X_match = pd.DataFrame(matchup_samples, columns=feature_cols)

    predictions = {}
    metrics = []

    # === XGBoost ===
    try:
        xgb_model = joblib.load(f"models/XGBoost.pkl")
        xgb_selector = joblib.load(f"models/XGBoost_selector.pkl")
        X_sel = xgb_selector.transform(X_match)
        probs = xgb_model.predict_proba(X_sel)[:, 1]
        avg_prob = probs.mean()
        predictions['XGBoost'] = {
            'prob_A_wins': round(avg_prob, 4),
            'prob_B_wins': round(1 - avg_prob, 4),
            'predicted_winner': fighter_a if avg_prob >= 0.5 else fighter_b
        }
        # Example metrics (replace with actual values)
        metrics.append({
            'Model': 'XGBoost',
            'Precision': precision_score([1], [1]),  # Replace with actual values
            'Recall': recall_score([1], [1]),        # Replace with actual values
            'F1 Score': f1_score([1], [1])           # Replace with actual values
        })
    except Exception as e:
        print("XGBoost error:", e)

    # === Gradient Boosting ===
    try:
        gb_model = joblib.load(f"models/Gradient_Boosting.pkl")
        gb_selector = joblib.load(f"models/Gradient_Boosting_selector.pkl")
        X_sel = gb_selector.transform(X_match)
        probs = gb_model.predict_proba(X_sel)[:, 1]
        avg_prob = probs.mean()
        predictions['Gradient Boosting'] = {
            'prob_A_wins': round(avg_prob, 4),
            'prob_B_wins': round(1 - avg_prob, 4),
            'predicted_winner': fighter_a if avg_prob >= 0.5 else fighter_b
        }
        metrics.append({
            'Model': 'Gradient Boosting',
            'Precision': precision_score([1], [1]),  # Replace with actual values
            'Recall': recall_score([1], [1]),        # Replace with actual values
            'F1 Score': f1_score([1], [1])           # Replace with actual values
        })
    except Exception as e:
        print("Gradient Boosting error:", e)

    # === Neural Network (MLP) ===
    try:
        mlp_model = joblib.load(f"models/Neural_Network_(MLP).pkl")
        mlp_scaler = joblib.load(f"models/Neural_Network_(MLP)_scaler.pkl")
        X_scaled = mlp_scaler.transform(X_match)
        probs = mlp_model.predict_proba(X_scaled)[:, 1]
        avg_prob = probs.mean()
        predictions['Neural Network (MLP)'] = {
            'prob_A_wins': round(avg_prob, 4),
            'prob_B_wins': round(1 - avg_prob, 4),
            'predicted_winner': fighter_a if avg_prob >= 0.5 else fighter_b
        }
        metrics.append({
            'Model': 'Neural Network (MLP)',
            'Precision': precision_score([1], [1]),  # Replace with actual values
            'Recall': recall_score([1], [1]),        # Replace with actual values
            'F1 Score': f1_score([1], [1])           # Replace with actual values
        })
    except Exception as e:
        print("MLP error:", e)

    prediction_df = pd.DataFrame(predictions).T
    metrics_df = pd.DataFrame(metrics)
    return prediction_df, metrics_df

# Example usage:
# prediction_df, metrics_df = predict_fight("Yair Rodriguez", "Patricio Pitbull")
# print(prediction_df)
# print(metrics_df)
