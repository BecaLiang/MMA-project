from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

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

    for name, model in models.items():
        print(f"\n=== {name} ===")

        if name == "Neural Network (MLP)":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        elif name in ["Naive Bayes", "Support Vector Machine"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        else:
            selector = run_rfecv(model, X_train, y_train, name)
            X_train_sel = selector.transform(X_train)
            X_test_sel = selector.transform(X_test)
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)

        print(classification_report(y_test, y_pred))
