import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap
except ImportError:
    os.system("pip install shap")
    import shap

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "/kaggle/working/models/"
ARTIFACT_DIR = "/kaggle/working/artifacts/"

def main():
    print("Loading trained classifier and preprocessed test data...")
    # X_test is ALREADY preprocessed data (scaled and one-hot encoded)
    X_test = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_test.csv"))
    pipeline = joblib.load(os.path.join(OUTPUT_DIR, "CAD_pipeline.joblib"))

    # We just need the classifier part since X_test is already processed
    classifier = pipeline.named_steps['classifier']

    print("Calculating SHAP values (this may take a minute)...")
    background = shap.sample(X_test, 50)

    # Explain the probability of predicting Class 1 (CAD Positive)
    predict_fn = lambda x: classifier.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, background)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # Clean up column names for the final plot (e.g., changing 'num__Age' back to 'Age')
    clean_columns = [c.replace('num__', '').replace('bin__', '').replace('cat__', '') for c in X_test.columns]
    X_test_display = X_test.copy()
    X_test_display.columns = clean_columns

    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_display, show=False)
    plt.title("SHAP Feature Importance (CAD Prediction)", fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "CAD_shap_summary.png"), bbox_inches='tight')
    plt.show()

    print("✅ SHAP analysis complete! Image saved to models directory.")

if __name__ == "__main__":
    main()
