#!/usr/bin/env python3
"""
Quick test script to run only the evaluation phase
(assuming synthetic data is already generated)
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport

# Setup
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(".")
CLEAN_DIR = BASE_DIR / "clean"
RESULTS_DIR = BASE_DIR / "results"
SYNTHETIC_DIR = BASE_DIR / "synthetic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "test_evaluation.log"),
        logging.StreamHandler()
    ]
)



def test_evaluation():
    """Test the evaluation phase with existing synthetic data."""
    logging.info("Starting evaluation test...")
    
    # Load existing data
    real_df = pd.read_csv(CLEAN_DIR / "oulad_master_engineered.csv")
    synthetic_sets = {}
    
    # Load synthetic data
    for synth_file in SYNTHETIC_DIR.glob("synthetic_*.csv"):
        name = synth_file.stem.replace("synthetic_", "")
        synthetic_sets[name] = pd.read_csv(synth_file)
        logging.info(f"Loaded synthetic data: {name} ({len(synthetic_sets[name])} rows)")
    
    if not synthetic_sets:
        logging.error("No synthetic data found!")
        return
    
    # Test ML utility evaluation
    logging.info("Testing ML utility evaluation...")
    real_df_cleaned = real_df.drop(columns=['id_student'], errors='ignore')
    categorical_features = real_df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features.remove('final_result')
    
        # Test classification
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
    
    clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)  # Reduced for speed
    clf.fit(X_train_proc, y_train)
    real_acc = accuracy_score(y_test, clf.predict(X_test_proc))
    real_auc = roc_auc_score(y_test, clf.predict_proba(X_test_proc)[:, 1])
    
    logging.info(f"Real data - Accuracy: {real_acc:.4f}, AUC: {real_auc:.4f}")
    
    # Test with synthetic data
    for name, synth_df in synthetic_sets.items():
        synth_df_copy = synth_df.copy()
        synth_df_copy['dropped'] = (synth_df_copy['final_result'] != 'Pass').astype(int)
        X_synth = synth_df_copy.drop(
        columns=['id_student', 'final_result', 'dropped', 'avg_assessment_score'],
        errors='ignore')
        y_synth = synth_df_copy['dropped']
        
        # CRITICAL FIX: Use the SAME encoder for synthetic data (no refitting)
        X_synth_encoded = pd.DataFrame(encoder.transform(X_synth[categorical_features]), index=X_synth.index)
        X_synth_encoded.columns = encoder.get_feature_names_out(categorical_features)
        X_synth_proc = pd.concat([X_synth.drop(columns=categorical_features), X_synth_encoded], axis=1)
        
        clf_synth = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
        clf_synth.fit(X_synth_proc, y_synth)
        acc = accuracy_score(y_test, clf_synth.predict(X_test_proc))
        auc = roc_auc_score(y_test, clf_synth.predict_proba(X_test_proc)[:, 1])
        
        logging.info(f"{name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # Test SDMetrics quality report
    logging.info("Testing SDMetrics quality report...")
    # Use the original real_df (with final_result) for the quality report
    real_data_for_report = real_df.drop(columns=['id_student'])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data_for_report)
    
    for name, synth_df in synthetic_sets.items():
        try:
            metadata_dict = metadata.to_dict()
            report = QualityReport()
            report.generate(real_data_for_report, synth_df, metadata_dict)
            # Get the overall quality score from the report
            quality_score = report.get_score()
            logging.info(f"{name} - Quality Score: {quality_score:.4f}")
        except Exception as e:
            logging.error(f"Could not generate quality report for {name}: {e}")
    
    logging.info("Evaluation test completed!")

if __name__ == "__main__":
    test_evaluation() 
