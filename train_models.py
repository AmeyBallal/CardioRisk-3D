import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# ==========================================
# CONFIGURATION
# ==========================================
ARTIFACT_DIR = "/kaggle/working/artifacts/"
OUTPUT_DIR = "/kaggle/working/models/"
RANDOM_SEED = 42
TARGET_COLS = ["CAD", "LAD", "LCX", "RCA"]

def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log"), mode='w')
        ]
    )

def load_artifacts():
    logging.info("Loading preprocessed datasets and preprocessor pipeline...")
    X_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_test.csv"))
    preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))
    return X_train, X_test, y_train, y_test, preprocessor

def get_model_grid(pos_weight, class_weight_dict):
    models = {
        'LogisticRegression': {
            'estimator': LogisticRegression(class_weight=class_weight_dict, random_state=RANDOM_SEED, max_iter=2000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2']
            }
        },
        'RandomForest': {
            'estimator': RandomForestClassifier(class_weight=class_weight_dict, random_state=RANDOM_SEED),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'SVM': {
            'estimator': SVC(class_weight=class_weight_dict, probability=True, random_state=RANDOM_SEED),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    if XGBClassifier is not None:
        models['XGBoost'] = {
            'estimator': XGBClassifier(scale_pos_weight=pos_weight, random_state=RANDOM_SEED, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
    else:
        logging.warning("XGBoost not installed, skipping XGBoost in grid search.")
        
    return models

def plot_roc_curve(y_test, y_prob, target_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {target_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_roc.png"))
    plt.close()

def plot_confusion_matrix(y_test, y_pred, target_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix for {target_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_confusion.png"))
    plt.close()
    
def plot_calibration_curve(clf, X_test, y_test, target_name):
    plt.figure(figsize=(8, 6))
    CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, name=target_name)
    plt.title(f'Calibration Curve for {target_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{target_name}_calibration.png"))
    plt.close()

def main():
    setup_logging()
    logging.info("Starting model training pipeline...")
    
    X_train, X_test, y_train, y_test, preprocessor = load_artifacts()
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    metrics_records = []
    
    for target in TARGET_COLS:
        logging.info(f"========== Training models for {target} ==========")
        y_tr = y_train[target]
        y_te = y_test[target]
        
        # Calculate class weights for imbalance
        neg_count = (y_tr == 0).sum()
        pos_count = (y_tr == 1).sum()
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        class_weight_dict = {0: 1.0, 1: pos_weight}
        
        logging.info(f"Class imbalance -> 0: {neg_count}, 1: {pos_count} (Pos Weight: {pos_weight:.2f})")
        
        models_grid = get_model_grid(pos_weight, class_weight_dict)
        
        best_overall_score = -1
        best_overall_model = None
        best_model_name = ""
        
        for name, config in models_grid.items():
            logging.info(f"Running RandomizedSearchCV for {name}...")
            search = RandomizedSearchCV(
                estimator=config['estimator'],
                param_distributions=config['params'],
                n_iter=10,
                scoring='roc_auc',
                cv=cv_strategy,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train, y_tr)
            
            logging.info(f"{name} best CV ROC-AUC: {search.best_score_:.4f}")
            if search.best_score_ > best_overall_score:
                best_overall_score = search.best_score_
                best_overall_model = search.best_estimator_
                best_model_name = name
                
        logging.info(f"🏆 Best model for {target} is {best_model_name} (CV AUC: {best_overall_score:.4f})")
        
        # Calibrate the best model to ensure probabilistic outputs represent true likelihood
        logging.info(f"Calibrating best {target} model...")
        calibrated_clf = CalibratedClassifierCV(estimator=best_overall_model, method='sigmoid', cv=cv_strategy)
        calibrated_clf.fit(X_train, y_tr)
        
        # Evaluate on held-out test set
        y_pred = calibrated_clf.predict(X_test)
        y_prob = calibrated_clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        
        logging.info(f"Test Set Metrics for {target}:")
        logging.info(f"Accuracy:  {acc:.4f}")
        logging.info(f"Precision: {prec:.4f}")
        logging.info(f"Recall:    {rec:.4f}")
        logging.info(f"F1-Score:  {f1:.4f}")
        logging.info(f"ROC-AUC:   {auc:.4f}")
        
        metrics_records.append({
            'Target': target,
            'Best_Algorithm': best_model_name,
            'CV_ROC_AUC': best_overall_score,
            'Test_Accuracy': acc,
            'Test_Precision': prec,
            'Test_Recall': rec,
            'Test_F1': f1,
            'Test_ROC_AUC': auc
        })
        
        # Visualizations
        plot_roc_curve(y_te, y_prob, target)
        plot_confusion_matrix(y_te, y_pred, target)
        plot_calibration_curve(calibrated_clf, X_test, y_te, target)
        
        # Bundle the preprocessor and the calibrated classifier into one single Pipeline
        logging.info(f"Bundling full pipeline for {target}...")
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', calibrated_clf)
        ])
        
        # Save the full bundled pipeline
        pipeline_path = os.path.join(OUTPUT_DIR, f"{target}_pipeline.joblib")
        joblib.dump(final_pipeline, pipeline_path)
        logging.info(f"Saved inference pipeline to {pipeline_path}\n")

    # Save overall metrics summary
    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved complete metrics summary to {metrics_path}")
    
    print("\n✅ Training Complete! Summary of Results:")
    print(metrics_df.to_string())

if __name__ == "__main__":
    main()
