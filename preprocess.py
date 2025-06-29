import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from ufcdata import load_or_scrape_data

# -------------------- 1. Clean fight results --------------------
def clean_fight_results(results_df):
    results_df.columns = results_df.columns.str.strip()
    results_df[['Player_A', 'Player_B']] = results_df['BOUT'].str.split(' vs. ', expand=True)
    results_df['outcome_number'] = results_df['OUTCOME'].apply(lambda x: 0 if x == 'W/L' else 1)
    results_df['Winner'] = results_df.apply(
        lambda row: row['Player_A'] if row['outcome_number'] == 0 else row['Player_B'], axis=1
    )
    results_df['Loser'] = results_df.apply(
        lambda row: row['Player_B'] if row['outcome_number'] == 0 else row['Player_A'], axis=1
    )
    return results_df

# -------------------- 2. Merge stats and results --------------------
def clean_fight_stats(stats_df, results_df):
    stats_df.columns = stats_df.columns.str.strip()
    results_df.columns = results_df.columns.str.strip()

    winner_df = results_df[['BOUT', 'Winner']].copy()

    def clean_bout(s):
        return str(s).strip().lower().replace('  ', ' ').replace('\xa0', ' ').replace('\n', '').replace('\r', '')

    stats_df['BOUT_clean'] = stats_df['BOUT'].apply(clean_bout)
    winner_df['BOUT_clean'] = winner_df['BOUT'].apply(clean_bout)
    results_df['BOUT_clean'] = results_df['BOUT'].apply(clean_bout)

    bout_to_winner = winner_df.set_index('BOUT_clean')['Winner'].to_dict()
    stats_df['Winner'] = stats_df['BOUT_clean'].map(bout_to_winner)

    columns_to_fill = ['WEIGHTCLASS', 'METHOD', 'TIME', 'TIME FORMAT', 'REFEREE', 'DETAILS']
    for col in columns_to_fill:
        value_map = results_df.set_index('BOUT_clean')[col].to_dict()
        stats_df[col] = stats_df['BOUT_clean'].map(value_map)

    stats_df[['Player_A', 'Player_B']] = stats_df['BOUT'].str.split(' vs. ', expand=True)
    stats_df.drop(columns=['BOUT_clean', 'BOUT', 'EVENT', 'DETAILS'], inplace=True, errors='ignore')

    cols_to_front = ['Player_A', 'Player_B', 'Winner', 'WEIGHTCLASS', 'TIME', 'TIME FORMAT', 'REFEREE']
    remaining_cols = [col for col in stats_df.columns if col not in cols_to_front]
    stats_df = stats_df[cols_to_front + remaining_cols]

    def determine_label(row):
        try:
            if pd.isna(row['FIGHTER']) or pd.isna(row['Winner']):
                return None
            return 1 if row['FIGHTER'].strip().lower() == row['Winner'].strip().lower() else 0
        except:
            return None

    stats_df['Label'] = stats_df.apply(determine_label, axis=1)
    stats_df.drop(columns=['Winner'], inplace=True)

    return stats_df

# -------------------- 3. Standardize weightclass --------------------
def standardize_weightclass(df):
    valid_weight_classes = [
        "Women's Strawweight", "Women's Bantamweight", "Women's Flyweight", "Women's Featherweight",
        'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight',
        'Heavyweight', 'Light Heavyweight', 'Flyweight', 'Bantamweight',
        'Super Heavyweight', 'Open Weight'
    ]
    def extract(text):
        text = str(text).lower()
        for wc in valid_weight_classes:
            if wc.lower() in text:
                return wc
        return None

    df['WEIGHTCLASS'] = df['WEIGHTCLASS'].apply(extract)
    weightclass_order = {
        "Flyweight": 0, "Bantamweight": 1, "Featherweight": 2, "Lightweight": 3,
        "Welterweight": 4, "Middleweight": 5, "Heavyweight": 6, "Open Weight": 7,
        "Women's Strawweight": 8, "Women's Flyweight": 9, "Women's Bantamweight": 10,
        "Women's Featherweight": 11
    }
    df['weightclass'] = df['WEIGHTCLASS'].map(weightclass_order)
    df.drop(columns='WEIGHTCLASS', inplace=True)
    return df

# -------------------- Step 4: Clean strike columns --------------------
def clean_strike_columns(df):
    strike_fields = {
        'HEAD': 'head_strike',
        'BODY': 'body_strike',
        'DISTANCE': 'distance_strike',
        'CLINCH': 'clinch_strike',
        'LEG': 'leg_strike',
        'GROUND': 'ground_strike',
        'TD': 'takedown'
    }
    for col, prefix in strike_fields.items():
        if col in df.columns:
            split = df[col].str.extract(r'(\d+)\s+of\s+(\d+)', expand=True)
            split.columns = [f'{prefix}_landed', f'{prefix}_attempted']
            split = split.apply(pd.to_numeric, errors='coerce')
            split[f'{prefix}_accuracy'] = (split[f'{prefix}_landed'] / split[f'{prefix}_attempted']).round(4)
            df = pd.concat([df, split[[f'{prefix}_attempted', f'{prefix}_accuracy']]], axis=1)
            df.drop(columns=col, inplace=True, errors='ignore')
        if col == 'TD' and 'TD %' in df.columns:
            df.drop(columns='TD %', inplace=True)
    return df

# -------------------- Step 5: Clean signature stats --------------------
def clean_signature_stats(df):
    if 'SIG.STR.' in df.columns:
        df[['Sig_Strikes_Landed', 'Sig_Strikes_Attempted']] = df['SIG.STR.'].str.split(' of ', expand=True).apply(pd.to_numeric)
        df['Strike_Accuracy'] = (df['Sig_Strikes_Landed'] / df['Sig_Strikes_Attempted']).round(4)
        df.drop(columns=['SIG.STR.', 'SIG.STR. %', 'Sig_Strikes_Landed'], inplace=True, errors='ignore')

    if 'TOTAL STR.' in df.columns:
        df[['Total_Strikes_Landed', 'Total_Strikes_Attempted']] = df['TOTAL STR.'].str.split(' of ', expand=True).apply(pd.to_numeric)
        df['Total_Strike_Accuracy'] = (df['Total_Strikes_Landed'] / df['Total_Strikes_Attempted']).round(4)
        df.drop(columns=['TOTAL STR.', 'Total_Strikes_Landed'], inplace=True, errors='ignore')
    return df

# -------------------- Step 6: Clean metadata --------------------
def clean_metadata(df):
    df['ROUND'] = pd.to_numeric(df['ROUND'].astype(str).str.replace('Round', '', case=False).str.strip(), errors='coerce')

    def parse_time_format(fmt):
        fmt = str(fmt).strip()
        if 'No Time Limit' in fmt:
            return pd.Series([np.nan, np.nan, 0, 1])
        has_ot = int('+ OT' in fmt or '+ 2OT' in fmt)
        numbers = list(map(int, re.findall(r'\d+', fmt)))
        num_rounds = numbers[0] if numbers else np.nan
        round_time = numbers[1] if len(numbers) > 1 else np.nan
        return pd.Series([num_rounds, round_time, has_ot, 0])

    df[['number_of_rounds', 'round_time_minutes', 'has_overtime', 'is_no_time_limit']] = df['TIME FORMAT'].apply(parse_time_format)
    df['TIME'] = df['TIME'].apply(lambda x: time_to_seconds(x))
    df['CTRL'] = df['CTRL'].apply(lambda x: time_to_seconds(x) if pd.notnull(x) else 0)
    df.drop(columns=['TIME FORMAT', 'REFEREE'], inplace=True, errors='ignore')

    method_map = {
        'Decision - Unanimous': 'decision',
        'Decision - Split': 'decision',
        'Decision - Majority': 'decision',
        'KO/TKO': 'tko_ko',
        'TKO - Doctor\'s Stoppage': 'tko_ko',
        'Submission': 'submission',
        'DQ': 'dq',
        'Could Not Continue': 'other',
        'Overturned': 'other',
        'Other': 'other'
    }
    method_code = {'decision': 0, 'tko_ko': 1, 'submission': 2, 'dq': 3, 'other': 4}
    df['METHOD'] = df['METHOD'].astype(str).str.strip().map(method_map).map(method_code)
    df.rename(columns={'METHOD': 'method_code'}, inplace=True)
    return df

def time_to_seconds(t):
    try:
        minutes, seconds = map(int, str(t).strip().split(':'))
        return minutes * 60 + seconds
    except:
        return np.nan

# -------------------- Step 7: Final cleanup --------------------
def final_cleanup(df):
    df.dropna(subset=['Label'], inplace=True)

    cols_to_check = [
        'Sig_Strikes_Attempted', 'Strike_Accuracy',
        'Total_Strikes_Attempted', 'Total_Strike_Accuracy',
        'head_strike_attempted', 'body_strike_attempted',
        'distance_strike_attempted', 'clinch_strike_attempted',
        'takedown_attempted', 'ground_strike_attempted',
        'leg_strike_attempted', 'KD', 'SUB.ATT', 'REV.', 'ROUND'
    ]
    df.dropna(subset=cols_to_check, inplace=True)
    df['CTRL'] = df['CTRL'].fillna(0)
    df['weightclass'] = df['weightclass'].fillna(df['weightclass'].mode()[0])

    accuracy_cols = [col for col in df.columns if 'accuracy' in col]
    df[accuracy_cols] = df[accuracy_cols].fillna(0.0)

    df.dropna(inplace=True)
    return df


# -------------------- Step 8: Aggregate fighter-level features --------------------
def aggregate_fighter_level(df):
    # Accuracy columns — use mean
    accuracy_cols = [
        'head_strike_accuracy', 'body_strike_accuracy', 'distance_strike_accuracy',
        'clinch_strike_accuracy', 'leg_strike_accuracy', 'ground_strike_accuracy',
        'takedown_accuracy', 'Strike_Accuracy', 'Total_Strike_Accuracy'
    ]

    # Sum columns — totals per fighter
    sum_cols = [
        'Sig_Strikes_Attempted', 'Total_Strikes_Attempted',
        'head_strike_attempted', 'body_strike_attempted',
        'distance_strike_attempted', 'clinch_strike_attempted',
        'takedown_attempted', 'ground_strike_attempted', 'leg_strike_attempted',
        'KD', 'SUB.ATT', 'REV.', 'CTRL'
    ]

    # Metadata — one value per fighter per bout
    meta_cols = ['TIME', 'weightclass', 'Label', 'method_code', 'ROUND']

    # Aggregation logic
    agg = {col: 'mean' for col in accuracy_cols}
    agg.update({col: 'sum' for col in sum_cols})
    agg.update({col: 'first' for col in meta_cols})

    # Aggregate per fighter per fight
    df = df.groupby(['Player_A', 'Player_B', 'FIGHTER']).agg(agg).reset_index()

    # Filter out women's weight classes
    df = df[df['weightclass'] < 8].reset_index(drop=True)

    return df

def run_preprocessing():
    """
    Run preprocessing on the loaded datasets.
    Returns:
        final_df (DataFrame): Preprocessed dataset.
    """
    # Load or scrape datasets
    event_details_df, fight_details_df, fight_results_df, fight_stats_df = load_or_scrape_data()

    if fight_results_df is None or fight_stats_df is None:
        raise ValueError("Failed to load or generate datasets.")

    # Step 1: Clean fight results
    fight_results_df = clean_fight_results(fight_results_df)

    # Step 2: Clean fight stats
    fight_stats_df = clean_fight_stats(fight_stats_df, fight_results_df)

    # Step 3: Standardize weightclass
    fight_stats_df = standardize_weightclass(fight_stats_df)

    # Step 4: Clean strike columns
    fight_stats_df = clean_strike_columns(fight_stats_df)

    # Step 5: Clean signature stats
    fight_stats_df = clean_signature_stats(fight_stats_df)

    # Step 6: Clean metadata
    fight_stats_df = clean_metadata(fight_stats_df)

    # Step 7: Final cleanup
    fight_stats_df = final_cleanup(fight_stats_df)

    # Step 8: Aggregate fighter-level features
    final_df = aggregate_fighter_level(fight_stats_df)

    return final_df