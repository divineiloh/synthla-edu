#!/usr/bin/env python3
"""
SYNTHLA-EDU: OULAD Synthetic Data Generation & Evaluation Pipeline v1.0

- Loads and engineers features from the full OULAD dataset.
- Trains multiple synthesizers to generate high-fidelity synthetic data.
- Evaluates data on a dual-task utility benchmark (classification & regression).
- Computes statistical quality and runs a Membership-Inference Attack (MIA) for privacy.
- Generates a comprehensive results report and separate visualizations.
- Fully reproducible and ready for containerization.
"""
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Scikit-learn for machine learning tasks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# SDV and SDMetrics for synthetic data generation and evaluation
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Path and Directory Setup ---
# Use environment variable for Docker compatibility, fallback to local path
ROOT = Path(os.getenv("OULAD_ROOT", "./OULAD_data"))
BASE_DIR = Path(".")
CLEAN_DIR = BASE_DIR / "clean"
RESULTS_DIR = BASE_DIR / "results"
SYNTHETIC_DIR = BASE_DIR / "synthetic"

# Create directories if they don't exist
CLEAN_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
SYNTHETIC_DIR.mkdir(exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "pipeline.log"),
        logging.StreamHandler() # Also print logs to the console
    ]
)

# --- 1. Data Loading & Feature Engineering ---
def load_and_engineer_oulad(root: Path) -> pd.DataFrame:
    """Load all OULAD CSVs, merge, engineer features, and handle missing values."""
    logging.info("Starting data loading and feature engineering...")
    try:
        tables = {f.stem: pd.read_csv(f) for f in root.glob("*.csv")}
        si = tables['studentInfo']
        crs = tables['courses']
        reg = tables['studentRegistration']
        vle = tables['studentVle']
        ass = tables['studentAssessment']
        logging.info("All source CSV files loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"OULAD data file not found. Please check the ROOT path. Error: {e}")
        raise

    # Merge core student information
    master = si.merge(crs, on=["code_module", "code_presentation"], how="left")
    master = master.merge(reg, on=["id_student", "code_module", "code_presentation"], how="left")

    # --- Feature Engineering ---
    logging.info("Engineering new features from VLE and assessment data...")
    # 1. Total VLE clicks and interaction flag
    vle_clicks = vle.groupby('id_student')['sum_click'].sum().reset_index()
    vle_clicks.columns = ['id_student', 'total_vle_clicks']
    master = master.merge(vle_clicks, on='id_student', how='left')
    master['total_vle_clicks'] = master['total_vle_clicks'].fillna(0).astype(int)
    master['has_vle_interaction'] = (master['total_vle_clicks'] > 0).astype(int)

    # 2. Average assessment score
    ass['score_clean'] = pd.to_numeric(ass['score'], errors='coerce')
    avg_score = ass.groupby('id_student')['score_clean'].mean().reset_index()
    avg_score.columns = ['id_student', 'avg_assessment_score']
    master = master.merge(avg_score, on='id_student', how='left')

    # --- Data Cleaning and Final Prep ---
    # Drop rows where the primary outcome is unknown
    master.dropna(subset=['final_result'], inplace=True)
    # Impute missing average score with the mean of the column
    master['avg_assessment_score'].fillna(master['avg_assessment_score'].mean(), inplace=True)

    # Audit and handle high-missingness columns
    imd_na_ratio = master['imd_band'].isna().mean()
    if imd_na_ratio > 0.3:
        master = master.drop(columns=['imd_band'])
        logging.warning(f"Dropped 'imd_band' due to {imd_na_ratio:.1%} missing values (>30%).")
    else:
        master['imd_band'].fillna('Unknown', inplace=True)
        logging.info(f"Retained 'imd_band' and filled {imd_na_ratio:.1%} missing values with 'Unknown'.")

    # Drop original date columns and any other non-essential identifiers
    master = master.drop(columns=['date_registration', 'date_unregistration'])

    out_path = CLEAN_DIR / "oulad_master_engineered.csv"
    master.to_csv(out_path, index=False)
    logging.info(f"Saved engineered master dataset to {out_path} ({len(master)} rows)")
    return master

# --- 2. Prepare Data for Synthesis ---
def prepare_data_for_synthesis(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure only primitive dtypes (int, float, str) remain for SDV compatibility."""
    logging.info("Preparing data for SDV synthesizers...")
    df_prep = df.copy()
    # Exclude student ID from the data fed to the synthesizers
    if 'id_student' in df_prep.columns:
        df_prep = df_prep.drop(columns=['id_student'])

    for col in df_prep.columns:
        if pd.api.types.is_categorical_dtype(df_prep[col]) or pd.api.types.is_object_dtype(df_prep[col]):
            df_prep[col] = df_prep[col].astype(str)
        elif pd.api.types.is_bool_dtype(df_prep[col]):
            df_prep[col] = df_prep[col].astype(int)
        elif pd.api.types.is_datetime64_any_dtype(df_prep[col]):
            df_prep[col] = df_prep[col].astype(str)
    logging.info("Data types converted to primitives for SDV.")
    return df_prep

# --- 3. Fit Generators and Generate Data ---
def fit_and_sample(df: pd.DataFrame, metadata: SingleTableMetadata) -> dict:
    """Train synthesizers, sample synthetic data, and save to disk."""
    logging.info("Fitting synthesizers and sampling data...")
    synthesizers = {
        'GaussianCopula': GaussianCopulaSynthesizer(metadata),
        'CTGAN': CTGANSynthesizer(metadata, epochs=400, batch_size=500, verbose=True)
    }
    synthetic_sets = {}
    for name, synth_model in synthesizers.items():
        logging.info(f"--- Training {name} ---")
        synth_model.fit(df)
        logging.info(f"--- Sampling from {name} ---")
        synthetic_data = synth_model.sample(num_rows=len(df))
        synthetic_sets[name] = synthetic_data
        out_path = SYNTHETIC_DIR / f"synthetic_{name}.csv"
        synthetic_data.to_csv(out_path, index=False)
        logging.info(f"Saved synthetic set '{name}' to {out_path}")
    return synthetic_sets

# --- 4. Evaluation Suite ---

def evaluate_privacy_attacks(real_data: pd.DataFrame, synthetic_sets: dict) -> dict:
    """
    Performs a Membership Inference Attack (MIA) to assess privacy leakage.
    Uses full feature set (numeric + one-hot encoded categorical) for comprehensive testing.
    A score close to 0.5 is ideal (privacy preserved).
    A score close to 1.0 is worst (privacy leaked).
    """
    mia_results = {}
    for name, synth_df in synthetic_sets.items():
        try:
            # Create labels: 0 for real data, 1 for synthetic data
            real_data_copy = real_data.copy()
            synth_df_copy = synth_df.copy()
            real_data_copy['is_synthetic'] = 0
            synth_df_copy['is_synthetic'] = 1
            
            # Combine data
            combined_data = pd.concat([real_data_copy, synth_df_copy], ignore_index=True)
            
            # One-hot encode categorical features for the attacker model
            categorical_features = combined_data.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'final_result' in categorical_features:
                 categorical_features.remove('final_result')
                 
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_cols = pd.DataFrame(
                encoder.fit_transform(combined_data[categorical_features]),
                index=combined_data.index,
                columns=encoder.get_feature_names_out(categorical_features)
            )
            
            X_attack = pd.concat([combined_data.drop(columns=categorical_features), encoded_cols], axis=1)
            X_attack = X_attack.drop(columns=['is_synthetic', 'final_result'])
            y_attack = combined_data['is_synthetic']
            
            # Train a simple classifier to distinguish real vs synthetic
            attacker = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1)
            # Use cross-validation to get a robust AUC score
            scores = cross_val_score(attacker, X_attack, y_attack, cv=3, scoring='roc_auc')
            mia_score = scores.mean()
            
            mia_results[f'{name}_mia_score'] = mia_score
            logging.info(f"MIA score for {name}: {mia_score:.4f}")
            
        except Exception as e:
            logging.error(f"Could not compute MIA score for {name}. Error: {e}")
            mia_results[f'{name}_mia_score'] = np.nan
    return mia_results

def evaluate_suite(real_df: pd.DataFrame, synthetic_sets: dict) -> dict:
    """Run a full evaluation suite: Utility, Quality, and Privacy Attacks."""
    logging.info("Starting full evaluation suite...")
    full_results = {'utility': {}, 'quality': {}, 'privacy_attack': {}}

    # --- Part 1: Utility Evaluation ---
    logging.info("--- Evaluating Machine Learning Utility ---")
    
    real_df_cleaned = real_df.drop(columns=['id_student'])
    categorical_features = real_df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features.remove('final_result')

    # --- Task 1: Dropout Classification ---
    real_df_cleaned['dropped'] = (real_df_cleaned['final_result'] != 'Pass').astype(int)
    X = real_df_cleaned.drop(columns=['final_result', 'dropped', 'avg_assessment_score'])
    y = real_df_cleaned['dropped']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    
    # CRITICAL FIX: Fit encoder only on training data to prevent data leakage
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_features]), index=X_train.index)
    X_train_encoded.columns = encoder.get_feature_names_out(categorical_features)
    X_train_proc = pd.concat([X_train.drop(columns=categorical_features), X_train_encoded], axis=1)
    
    # Transform test data using the SAME encoder (no refitting)
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_features]), index=X_test.index)
    X_test_encoded.columns = encoder.get_feature_names_out(categorical_features)
    X_test_proc = pd.concat([X_test.drop(columns=categorical_features), X_test_encoded], axis=1)
    
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train_proc, y_train)
    full_results['utility']['real_dropout_accuracy'] = accuracy_score(y_test, clf.predict(X_test_proc))
    proba_real = clf.predict_proba(X_test_proc)[:, 1]
    full_results['utility']['real_dropout_auc'] = roc_auc_score(y_test, proba_real)

    for name, synth_df in synthetic_sets.items():
        synth_df_copy = synth_df.copy()
        synth_df_copy['dropped'] = (synth_df_copy['final_result'] != 'Pass').astype(int)
        X_synth = synth_df_copy.drop(columns=['final_result', 'dropped', 'avg_assessment_score'])
        y_synth = synth_df_copy['dropped']
        
        # CRITICAL FIX: Use the SAME encoder for synthetic data (no refitting)
        X_synth_encoded = pd.DataFrame(encoder.transform(X_synth[categorical_features]), index=X_synth.index)
        X_synth_encoded.columns = encoder.get_feature_names_out(categorical_features)
        X_synth_proc = pd.concat([X_synth.drop(columns=categorical_features), X_synth_encoded], axis=1)
        
        clf_synth = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        clf_synth.fit(X_synth_proc, y_synth)
        
        acc = accuracy_score(y_test, clf_synth.predict(X_test_proc))
        proba_synth = clf_synth.predict_proba(X_test_proc)[:, 1]
        auc = roc_auc_score(y_test, proba_synth)
        
        full_results['utility'][f'{name}_dropout_accuracy'] = acc
        full_results['utility'][f'{name}_dropout_auc'] = auc

    # --- Task 2: Grade Regression ---
    X_reg = real_df_cleaned.drop(columns=['final_result', 'dropped', 'avg_assessment_score'])
    y_reg = real_df_cleaned['avg_assessment_score']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.25, random_state=RANDOM_SEED)

    # CRITICAL FIX: Fit encoder only on training data to prevent data leakage
    encoder_reg = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_r_encoded = pd.DataFrame(encoder_reg.fit_transform(X_train_r[categorical_features]), index=X_train_r.index)
    X_train_r_encoded.columns = encoder_reg.get_feature_names_out(categorical_features)
    X_train_r_proc = pd.concat([X_train_r.drop(columns=categorical_features), X_train_r_encoded], axis=1)
    
    # Transform test data using the SAME encoder (no refitting)
    X_test_r_encoded = pd.DataFrame(encoder_reg.transform(X_test_r[categorical_features]), index=X_test_r.index)
    X_test_r_encoded.columns = encoder_reg.get_feature_names_out(categorical_features)
    X_test_r_proc = pd.concat([X_test_r.drop(columns=categorical_features), X_test_r_encoded], axis=1)
    
    reg = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    reg.fit(X_train_r_proc, y_train_r)
    full_results['utility']['real_grade_mae'] = mean_absolute_error(y_test_r, reg.predict(X_test_r_proc))

    for name, synth_df in synthetic_sets.items():
        X_synth_r = synth_df.drop(columns=['final_result', 'avg_assessment_score'])
        y_synth_r = synth_df['avg_assessment_score']

        # CRITICAL FIX: Use the SAME encoder for synthetic data (no refitting)
        X_synth_r_encoded = pd.DataFrame(encoder_reg.transform(X_synth_r[categorical_features]), index=X_synth_r.index)
        X_synth_r_encoded.columns = encoder_reg.get_feature_names_out(categorical_features)
        X_synth_r_proc = pd.concat([X_synth_r.drop(columns=categorical_features), X_synth_r_encoded], axis=1)
        
        reg_synth = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
        reg_synth.fit(X_synth_r_proc, y_synth_r)
        mae = mean_absolute_error(y_test_r, reg_synth.predict(X_test_r_proc))
        full_results['utility'][f'{name}_grade_mae'] = mae
    
    logging.info("Machine Learning Utility evaluation complete.")

    # --- Part 2: Quality & Privacy Evaluation ---
    logging.info("--- Evaluating Statistical Quality and Privacy ---")
    # Use the original real_df (with final_result) for the quality report
    real_data_for_report = prepare_data_for_synthesis(real_df)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data_for_report)

    for name, synth_df in synthetic_sets.items():
        try:
            # Convert SDV metadata to SDMetrics format
            metadata_dict = metadata.to_dict()
            report = QualityReport()
            report.generate(real_data_for_report, synth_df, metadata_dict)
            # Get the overall quality score from the report
            quality_score = report.get_score()
            full_results['quality'][f'{name}_quality_score'] = quality_score
            logging.info(f"{name} - Quality Score: {quality_score:.4f}")
            
            # Try to save visualization if available
            try:
                fig = report.get_visualization('Column Shapes')
                fig.write_image(RESULTS_DIR / f"quality_report_{name}.png")
                logging.info(f"Saved SDMetrics quality report for {name}.")
            except Exception as viz_error:
                logging.warning(f"Could not save visualization for {name}: {viz_error}")
                # Continue execution - visualization failure shouldn't crash the pipeline
                
        except Exception as e:
            logging.error(f"Could not generate quality report for {name}: {e}")
            full_results['quality'][f'{name}_quality_score'] = np.nan

    # --- Part 3: Privacy Attack Simulation (MIA) ---
    logging.info("--- Simulating Membership Inference Attacks (MIA) ---")
    mia_results = evaluate_privacy_attacks(real_data_for_report, synthetic_sets)
    full_results['privacy_attack'] = mia_results

    return full_results

# --- 5. Visualization ---
def create_summary_visualizations(results: dict):
    """Create and save separate summary plots for each evaluation metric."""
    logging.info("Creating and saving separate summary visualizations...")
    utility_res = results['utility']
    quality_res = results['quality']
    privacy_res = results['privacy_attack']
    
    models = ['GaussianCopula', 'CTGAN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Real, GC, CTGAN

    # --- Plot 1: Classification Utility (AUC) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    real_auc = utility_res['real_dropout_auc']
    synth_aucs = [utility_res[f'{m}_dropout_auc'] for m in models]
    ax1.bar(['Real Data'] + models, [real_auc] + synth_aucs, color=colors)
    ax1.axhline(y=real_auc, color='r', linestyle='--', label=f'Real Data AUC ({real_auc:.3f})')
    ax1.set_title('Dropout Prediction Utility (AUC Score)')
    ax1.set_ylabel('Area Under Curve (AUC)')
    ax1.set_ylim(bottom=0.5)
    ax1.legend()
    fig1.tight_layout()
    out_path1 = RESULTS_DIR / "summary_classification_utility.png"
    fig1.savefig(out_path1)
    logging.info(f"Saved classification utility plot to {out_path1}")
    plt.close(fig1)

    # --- Plot 2: Regression Utility (MAE) ---
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
    real_mae = utility_res['real_grade_mae']
    synth_maes = [utility_res[f'{m}_grade_mae'] for m in models]
    ax2.bar(['Real Data'] + models, [real_mae] + synth_maes, color=colors)
    ax2.axhline(y=real_mae, color='r', linestyle='--', label=f'Real Data MAE ({real_mae:.2f})')
    ax2.set_title('Grade Prediction Utility (MAE)')
    ax2.set_ylabel('Mean Absolute Error (lower is better)')
    ax2.legend()
    fig2.tight_layout()
    out_path2 = RESULTS_DIR / "summary_regression_utility.png"
    fig2.savefig(out_path2)
    logging.info(f"Saved regression utility plot to {out_path2}")
    plt.close(fig2)

    # --- Plot 3: Overall Quality Score ---
    fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=100)
    quality_scores = [quality_res[f'{m}_quality_score'] for m in models]
    ax3.bar(models, quality_scores, color=colors[1:]) # Exclude 'Real Data' color
    ax3.set_title('Overall Data Quality (SDMetrics)')
    ax3.set_ylabel('Quality Score (higher is better)')
    ax3.set_ylim(0, 1)
    fig3.tight_layout()
    out_path3 = RESULTS_DIR / "summary_data_quality.png"
    fig3.savefig(out_path3)
    logging.info(f"Saved data quality plot to {out_path3}")
    plt.close(fig3)

    # --- Plot 4: Privacy Leakage (MIA) ---
    fig4, ax4 = plt.subplots(figsize=(8, 6), dpi=100)
    mia_scores = [privacy_res.get(f'{m}_mia_score', np.nan) for m in models]
    ax4.bar(models, mia_scores, color=colors[1:])
    ax4.set_title('Privacy Leakage (MIA Score)')
    ax4.set_ylabel('Distinguishability AUC (lower is better)')
    ax4.set_ylim(0, 1)
    ax4.axhline(y=0.5, color='g', linestyle='--', label='Ideal (Cannot Distinguish)')
    ax4.legend()
    fig4.tight_layout()
    out_path4 = RESULTS_DIR / "summary_privacy_mia.png"
    fig4.savefig(out_path4)
    logging.info(f"Saved MIA privacy plot to {out_path4}")
    plt.close(fig4)


# --- MAIN PIPELINE ---
def main():
    """Main execution function to run the entire pipeline."""
    logging.info("="*50)
    logging.info("SYNTHLA-EDU Pipeline Started")
    logging.info("="*50)
    
    try:
        # 1. Data loading + feature engineering
        master_real_df = load_and_engineer_oulad(ROOT)
        
        # 2. Prepare data for synthesis (no IDs)
        synth_ready_df = prepare_data_for_synthesis(master_real_df)
        
        # 3. Create metadata from the prepared data
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=synth_ready_df)
        
        # 4. Fit generators and sample synthetic sets
        synthetic_sets = fit_and_sample(synth_ready_df, metadata)
        
        # 5. Run the full evaluation suite
        final_results = evaluate_suite(master_real_df, synthetic_sets)
        
        # 6. Save final results to JSON
        results_path = RESULTS_DIR / "final_results.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=4)
        logging.info(f"Saved final combined results to {results_path}")
        
        # 7. Create summary visualizations
        create_summary_visualizations(final_results)
        
        logging.info("="*50)
        logging.info("SYNTHLA-EDU Pipeline Completed Successfully!")
        logging.info("="*50)

    except Exception as e:
        logging.critical(f"Pipeline failed with a critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
