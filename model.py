from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os


# ============================
# Load preprocessed dataframe: final_df
# Ensure 'METHOD' is encoded if categorical
# ============================

def encode_method_column(df):
    if 'METHOD' in df.columns and df['METHOD'].dtype == 'object':
        df['METHOD'] = df['METHOD'].astype('category').cat.codes
    return df

def prepare_data(df, target_col='Label'):
    df = df.dropna(subset=[target_col])
    df = encode_method_column(df)

    exclude_cols = ['Player_A', 'Player_B', 'FIGHTER', 'ROUND', target_col, 'TIME', 'TIME FORMAT', 'REFEREE']
    features = [col for col in df.columns if col not in exclude_cols]
    X = df[features]
    y = df[target_col].astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def run_rfecv(model, X, y, name):
    print(f"\n===== {name} with RFECV Feature Selection =====")
    selector = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
    selector.fit(X, y)
    selected_features = list(X.columns[selector.support_])
    print(f"Selected Features for {name}: {selected_features}")
    return selector



def train_and_evaluate(final_df):
    X_train, X_test, y_train, y_test = prepare_data(final_df)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Neural Network (MLP)": MLPClassifier(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Support Vector Machine": SVC(probability=True)
    }

    os.makedirs("models", exist_ok=True)
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")
        selector = None  # initialize

        if name == "Neural Network (MLP)":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Save model and scaler
            joblib.dump(model, f"models/{name}.pkl")
            joblib.dump(scaler, f"models/{name}_scaler.pkl")

        elif name in ["Naive Bayes", "Support Vector Machine"]:
            if name == "Support Vector Machine":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                # Save model and scaler
                joblib.dump(model, f"models/{name}.pkl")
                joblib.dump(scaler, f"models/{name}_scaler.pkl")
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                joblib.dump(model, f"models/{name}.pkl")

        else:
            # Run RFECV and use selected features
            selector = run_rfecv(model, X_train, y_train, name)
            X_train_sel = selector.transform(X_train)
            X_test_sel = selector.transform(X_test)
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)

            # Save model and selector
            joblib.dump(model, f"models/{name}.pkl")
            joblib.dump(selector, f"models/{name}_selector.pkl")

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        support = len(y_test)

        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Support": support
        })

        print(classification_report(y_test, y_pred))

    return pd.DataFrame(results)


import json

def get_top_models(results_df, top_n=3):
    top_models_df = results_df.sort_values(by="F1 Score", ascending=False).head(top_n)
    print("\n=== Top Models ===")
    print(top_models_df)

    # Save top 3 model names to JSON
    top_model_names = top_models_df["Model"].tolist()
    with open("models/top_models.json", "w") as f:
        json.dump(top_model_names, f)

    return top_models_df


if __name__ == "__main__":
    from preprocess import run_preprocessing

    print("Starting model training...")
    final_df = run_preprocessing()
    results_df = train_and_evaluate(final_df)
    get_top_models(results_df, top_n=3)
    print("All models trained and saved to 'models/' folder.")
