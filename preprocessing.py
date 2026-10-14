import os
import json
import logging
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
# CONFIGURATION
# ==========================================
def _find_dataset_path():
    if os.path.exists("/kaggle/input"):
        import glob
        # Search for any csv or xlsx in the Kaggle input directory
        files = glob.glob("/kaggle/input/**/*.csv", recursive=True) + \
                glob.glob("/kaggle/input/**/*.xlsx", recursive=True) + \
                glob.glob("/kaggle/input/**/*.xls", recursive=True)
        if files:
            return files[0] # Return the first found dataset
    return "Dataset/extention of Z-Alizadeh sani dataset.xlsx" # Fallback local path

DATA_PATH = _find_dataset_path()
OUTPUT_DIR = "/kaggle/working/artifacts/"
USE_WINSORIZE = False
RANDOM_SEED = 42
TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]


def setup_logging():
    """Configure logging to print to console and write to a log file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(OUTPUT_DIR, "preprocessing.log"), mode='w')
        ]
    )


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from CSV or Excel, clean column names, and report basic info."""
    logging.info(f"Loading data from {path}...")
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Must be .csv or .xlsx")
        
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    logging.info(f"Data shape: {df.shape}")
    logging.info(f"Data types:\n{df.dtypes.value_counts()}")
    missing = df.isnull().sum()
    logging.info(f"Missing values by column:\n{missing[missing > 0]}")
    
    # Class balance for all 4 targets
    for t in TARGET_COLS:
        if t in df.columns:
            logging.info(f"Class balance for {t}:\n{df[t].value_counts(dropna=False).to_dict()}")
            
    return df


def process_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map targets to 0/1 binary values."""
    logging.info("Building binary targets...")
    df = df.copy()
    
    def map_target(series: pd.Series, target_name: str) -> pd.Series:
        mapping = {}
        for val in series.dropna().unique():
            s = str(val).strip().lower()
            if s in ['cad', 'stenotic', '1', 'yes', 'y']:
                mapping[val] = 1
            else:
                mapping[val] = 0
        mapped = series.map(mapping)
        logging.info(f"Mapped {target_name} unique values {series.unique().tolist()} -> {mapped.unique().tolist()}")
        return mapped

    y = pd.DataFrame()
    for col in TARGET_COLS:
        if col in df.columns:
            y[col] = map_target(df[col], col)
            # Assert only {0,1} remain
            assert set(y[col].dropna().unique()).issubset({0, 1}), f"Non-binary values found in target {col}"
            
    # Rename Cath to CAD in the target dataframe
    if "Cath" in y.columns:
        y = y.rename(columns={"Cath": "CAD"})
        
    return df, y


def check_consistency(y: pd.DataFrame):
    """Check for logical consistency between overall CAD and vessel-specific labels."""
    if all(col in y.columns for col in ["CAD", "LAD", "LCX", "RCA"]):
        cad_but_no_vessel = ((y["CAD"] == 1) & (y["LAD"] == 0) & (y["LCX"] == 0) & (y["RCA"] == 0)).sum()
        no_cad_but_vessel = ((y["CAD"] == 0) & ((y["LAD"] == 1) | (y["LCX"] == 1) | (y["RCA"] == 1))).sum()
        
        logging.info(f"Consistency Check:")
        logging.info(f"  Rows with CAD=1 but all vessel labels=0: {cad_but_no_vessel}")
        logging.info(f"  Rows with CAD=0 but at least one vessel label=1: {no_cad_but_vessel}")


def clean_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Attempt to safely cast object columns to numeric if they are essentially numbers."""
    X_out = X.copy()
    for col in X_out.columns:
        try:
            X_out[col] = pd.to_numeric(X_out[col])
        except (ValueError, TypeError):
            pass
    return X_out


def handle_duplicates_and_constants(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and drop duplicate rows and constant columns."""
    df_combined = pd.concat([X, y], axis=1)
    duplicates = df_combined.duplicated().sum()
    logging.info(f"Found {duplicates} duplicate rows. Dropping them.")
    df_combined = df_combined.drop_duplicates()
    
    X_new = df_combined[X.columns].copy()
    y_new = df_combined[y.columns].copy()
    
    constants = [col for col in X_new.columns if X_new[col].nunique(dropna=False) <= 1]
    logging.info(f"Found {len(constants)} constant columns: {constants}. Dropping them.")
    X_new = X_new.drop(columns=constants)
    
    return X_new, y_new


def categorize_features(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Automatically categorize features into binary, categorical, and numeric."""
    binary_cols = []
    categorical_cols = []
    numeric_cols = []
    
    for col in X.columns:
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= 2:
            binary_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]) and n_unique > 10:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    logging.info(f"Binary features ({len(binary_cols)}): {binary_cols}")
    logging.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    logging.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    
    return binary_cols, categorical_cols, numeric_cols


def standardize_binary(X: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    """Map string-based binary columns to 0.0 and 1.0 safely."""
    X_out = X.copy()
    for col in binary_cols:
        unique_vals = X_out[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0, '0', '1', '0.0', '1.0'}):
            X_out[col] = pd.to_numeric(X_out[col]).astype(float)
            continue
            
        val_strs = {v: str(v).strip().lower() for v in unique_vals}
        pos_keywords = ['y', 'yes', 'male', 'positive', 'true', '1']
        
        mapping = {}
        for v, s in val_strs.items():
            if s in pos_keywords:
                mapping[v] = 1
            elif s in ['n', 'no', 'female', 'fmale', 'negative', 'false', '0']:
                mapping[v] = 0
            else:
                mapping[v] = 0 # Default fallback
                
        # If no keywords matched, just map alphabetically to 0 and 1
        if len(set(mapping.values())) <= 1 and len(unique_vals) == 2:
            sorted_vals = sorted(list(unique_vals), key=lambda x: str(x))
            mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
            
        X_out[col] = X_out[col].map(mapping).astype(float)
    return X_out


def report_outliers(X: pd.DataFrame, numeric_cols: List[str]):
    """Report outliers using the 1.5 * IQR rule."""
    logging.info("Outlier detection (1.5 * IQR method):")
    for col in numeric_cols:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            logging.info(f"  {col}: {outliers} outliers")


def winsorize_features(X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Cap numeric features at the 5th and 95th percentiles."""
    X_out = X.copy()
    for col in numeric_cols:
        q1 = X_out[col].quantile(0.05)
        q3 = X_out[col].quantile(0.95)
        X_out[col] = X_out[col].clip(lower=q1, upper=q3)
    return X_out


def build_preprocessor(numeric_cols: List[str], binary_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('bin', binary_transformer, binary_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor


def make_plots(df: pd.DataFrame, output_dir: str):
    """Generate and save requested visualizations."""
    logging.info("Generating plots...")
    
    # 1. Target class balance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    targets = ["CAD", "LAD", "LCX", "RCA"]
    for i, t in enumerate(targets):
        if t in df.columns:
            sns.countplot(data=df, x=t, ax=axes[i], hue=df[t], legend=False, palette="viridis")
            axes[i].set_title(f'Class Balance: {t}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_balance.png'))
    plt.show()

    # 2. Missing values
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing.plot(kind='bar', color='salmon')
    else:
        plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=20)
        plt.axis('off')
    plt.title('Missing Values by Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values.png'))
    plt.show()
    
    # 3. Correlation heatmap of numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        plt.figure(figsize=(16, 12))
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title('Correlation Heatmap (Numeric Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.show()
        
    # 4. Age/BMI distributions by CAD
    if "CAD" in df.columns:
        if "Age" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df, x="Age", hue="CAD", kde=True)
            plt.title('Age Distribution by CAD')
            plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
            plt.show()
            
        if "BMI" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df, x="BMI", hue="CAD", kde=True)
            plt.title('BMI Distribution by CAD')
            plt.savefig(os.path.join(output_dir, 'bmi_distribution.png'))
            plt.show()
            
        # 5. Boxplots of Key Clinical Markers by CAD
        key_markers = [col for col in ['BP', 'PR', 'FBS', 'LDL', 'HDL', 'TG'] if col in df.columns]
        if key_markers:
            n_cols = 3
            n_rows = (len(key_markers) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten() if len(key_markers) > 1 else [axes]
            for i, col in enumerate(key_markers):
                sns.boxplot(data=df, x="CAD", y=col, ax=axes[i], hue="CAD", legend=False, palette="Set2")
                axes[i].set_title(f'{col} by CAD')
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'clinical_markers_boxplots.png'))
            plt.show()

        # 6. Stacked Bar Charts for Categorical Risk Factors
        risk_factors = [col for col in ['Sex', 'Current Smoker', 'DM', 'HTN', 'Obesity'] if col in df.columns]
        if risk_factors:
            n_cols = 3
            n_rows = (len(risk_factors) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten() if len(risk_factors) > 1 else [axes]
            for i, factor in enumerate(risk_factors):
                cross_tab = pd.crosstab(df[factor], df['CAD'], normalize='index') * 100
                cross_tab.plot(kind='bar', stacked=True, ax=axes[i], colormap="viridis")
                axes[i].set_title(f'CAD Prevalence by {factor}')
                axes[i].set_ylabel('Percentage (%)')
                axes[i].legend(title='CAD')
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'risk_factors_stacked_bars.png'))
            plt.show()

    # 7. Multi-Vessel Disease Distribution
    vessel_cols = [col for col in ["LAD", "LCX", "RCA"] if col in df.columns]
    if len(vessel_cols) == 3:
        try:
            num_diseased = df[vessel_cols].sum(axis=1)
            plt.figure(figsize=(8, 6))
            sns.countplot(x=num_diseased, hue=num_diseased, legend=False, palette="magma")
            plt.title('Multi-Vessel Disease Distribution (Count of Stenotic Vessels)')
            plt.xlabel('Number of Diseased Vessels (0 to 3)')
            plt.ylabel('Patient Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'multi_vessel_distribution.png'))
            plt.show()
        except Exception as e:
            logging.warning(f"Could not generate multi-vessel plot: {e}")


def main():
    setup_logging()
    logging.info("Starting preprocessing pipeline...")
    
    # 1. Load data
    df = load_data(DATA_PATH)
    
    # 2. Build binary targets
    df, y = process_targets(df)
    
    # 3. Check logical consistency between targets
    check_consistency(y)
    
    # 4. Target Leakage: Remove targets from feature matrix
    X = df.drop(columns=TARGET_COLS, errors='ignore')
    logging.info(f"Targets removed from X. Features remaining: {len(X.columns)}")
    assert not any(c in X.columns for c in TARGET_COLS), "Target leakage detected!"
    
    # Pre-clean string/object types to numeric where directly applicable
    X = clean_dtypes(X)
    
    # 6. Duplicates and Constants
    X, y = handle_duplicates_and_constants(X, y)
    
    # 5. Auto-categorize features
    binary_cols, categorical_cols, numeric_cols = categorize_features(X)
    
    # Drop constant columns from feature lists if they were removed
    binary_cols = [c for c in binary_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    
    # Standardize string binary features to 0/1 numeric
    X = standardize_binary(X, binary_cols)
    
    # Outliers
    report_outliers(X, numeric_cols)
    if USE_WINSORIZE:
        logging.info("Applying winsorization...")
        X = winsorize_features(X, numeric_cols)
        
    # Generate Raw Feature Schema (before pipeline transformations)
    raw_feature_schema = {}
    for col in X.columns:
        schema = {"dtype": str(X[col].dtype)}
        if col in numeric_cols:
            schema["min"] = float(X[col].min()) if not pd.isna(X[col].min()) else None
            schema["max"] = float(X[col].max()) if not pd.isna(X[col].max()) else None
        else:
            allowed_vals = [v for v in X[col].dropna().unique() if not pd.isna(v)]
            schema["allowed_values"] = [
                float(v) if isinstance(v, (np.floating, float, np.integer, int)) else str(v) 
                for v in allowed_vals
            ]
        raw_feature_schema[col] = schema
        
    with open(os.path.join(OUTPUT_DIR, "raw_feature_schema.json"), "w") as f:
        json.dump(raw_feature_schema, f, indent=4)
        
    # Generate Plots
    plot_df = pd.concat([X, y], axis=1)
    make_plots(plot_df, OUTPUT_DIR)
    
    # 8. Splits and StratifiedKFold
    logging.info("Splitting data into 80/20 train/test stratified by CAD...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y["CAD"], random_state=RANDOM_SEED
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_indices = []
    for train_idx, val_idx in skf.split(X_train, y_train["CAD"]):
        cv_indices.append({
            "train": train_idx.tolist(),
            "val": val_idx.tolist()
        })
    with open(os.path.join(OUTPUT_DIR, "cv_indices.json"), "w") as f:
        json.dump(cv_indices, f)
        
    # 7. Build and fit pipeline
    preprocessor = build_preprocessor(numeric_cols, binary_cols, categorical_cols)
    
    # Output Pandas DataFrames instead of Numpy arrays (Requires scikit-learn >= 1.2)
    preprocessor.set_output(transform="pandas")
    
    logging.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    logging.info("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)
    
    # 9. Save outputs
    logging.info("Saving artifacts to output directory...")
    X_train_processed.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test_processed.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
    
    feature_names = X_train_processed.columns.tolist()
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=4)
        
    feature_groups = {
        "binary": binary_cols,
        "categorical": categorical_cols,
        "numeric": numeric_cols
    }
    with open(os.path.join(OUTPUT_DIR, "feature_groups.json"), "w") as f:
        json.dump(feature_groups, f, indent=4)
        
    # Generate Preprocessing Report
    report = f"""# Preprocessing Report

## Dataset Summary
- **Original rows**: {len(df)}
- **Processed rows** (after duplicate removal): {len(X)}
- **Features**: {len(X.columns)}
- **Targets**: {list(y.columns)}

## Feature Groups
- **Binary ({len(binary_cols)})**: {', '.join(binary_cols)}
- **Categorical ({len(categorical_cols)})**: {', '.join(categorical_cols)}
- **Numeric ({len(numeric_cols)})**: {', '.join(numeric_cols)}

## Pipeline Transformations
- **Numeric**: Median Imputation + Standard Scaling
- **Binary**: Most Frequent Imputation (String classes mapped to 0/1)
- **Categorical**: Most Frequent Imputation + One-Hot Encoding (sparse_output=False, ignore unknown)

## Data Leakage Addressed
Targets `{'`, `'.join(TARGET_COLS)}` successfully extracted and excluded from feature set.
"""
    with open(os.path.join(OUTPUT_DIR, "preprocessing_report.md"), "w") as f:
        f.write(report)
        
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
