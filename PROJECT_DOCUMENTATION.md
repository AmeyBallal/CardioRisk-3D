# CardioRisk: Interactive 3D Cardiac Vasculature Mapping & Calibrated Multi-Target Coronary Artery Disease Risk Prediction

**Multimodal AI Hackathon 2026 — Track A: Cardiovascular Risk Visualization & Prediction**  
**Technical Documentation & System Specification (Deliverable: Max 6 Pages)**  
**Author / Team:** CardioRisk Engineering Team  
**Date:** October 2026 | **Version:** 1.0.0-Production  

---

## Executive Summary & Abstract

Coronary Artery Disease (CAD) is the leading cause of global cardiovascular mortality. Conventional statistical risk scoring frameworks (e.g., Framingham Risk Score, Pooled Cohort ASCVD equations) exhibit two primary clinical deficiencies: (1) they compute aggregate 10-year statistical probabilities that lack vessel-specific anatomical localization, and (2) they present outputs as raw tables, depriving clinicians and surgical teams of intuitive 3D spatial context.

**CardioRisk** addresses these limitations by introducing an end-to-end multimodal clinical intelligence platform that pairs **calibrated multi-target machine learning ensembles** with an **interactive WebGL-based 3D anatomical human heart engine**. Trained on the benchmark catheterization-confirmed Z-Alizadeh Sani dataset ($N = 303$, $54$ physiological parameters), our platform predicts overall CAD presence alongside vessel-specific stenosis across the three primary coronary arteries:
1. **LAD (Left Anterior Descending):** *Anterior Interventricular Groove*
2. **LCX (Left Circumflex):** *Left Atrioventricular Sulcus*
3. **RCA (Right Coronary Artery):** *Right Coronary Sulcus*

Predictions are calibrated via Platt Sigmoid scaling (`CalibratedClassifierCV`) to ensure that output percentages represent true empirical event likelihoods. Using **Three.js (r128)**, risk probabilities are dynamically projected onto an anatomically verified 3D cardiac asset (University of Dundee) via dynamic SVG leader lines, pulsating target rings, and glassmorphism diagnostic badges. The system achieves a hold-out test ROC-AUC of **0.8760** (Accuracy: $88.52\%$, F1: $0.9231$) for overall CAD, **0.7985** for LAD, **0.6989** for LCX, and **0.7171** for RCA, while preserving 100% of the natural anatomical textures of the heart muscle and veins.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  CARDIORISK END-TO-END WORKFLOW                                  │
├─────────────────────┬───────────────────────────┬──────────────────────────┬─────────────────────┤
│ 1. Patient Profile  │ 2. Pipeline Imputation    │ 3. Calibrated ML Engines │ 4. 3D Spatial Canvas│
│ 54 Clinical Markers │ Median / Mode Imputation  │ CAD  (RF: AUC 0.876)     │ Three.js WebGL (60fps)
│ Demographics, ECG,  │ Standard Scaling + OHE    │ LAD  (RF: AUC 0.798)     │ Dynamic SVG Pointers│
│ Labs, Echo, Vitals  │ (preprocessor.joblib)     │ LCX  (RF: AUC 0.699)     │ Locked to Anatomical│
│ (Optional inputs)   │ Zero Target Data Leakage  │ RCA  (L2-LR: AUC 0.717)  │ Coronary Grooves    │
└─────────────────────┴───────────────────────────┴──────────────────────────┴─────────────────────┘
```

---

## 1. Clinical Domain Motivation & Anatomical Context

Coronary arteries supply oxygenated blood to the myocardium. Atherosclerotic plaque accumulation causes arterial narrowing (*stenosis*). In standard interventional cardiology, a stenosis $\ge 50\%$ in a major epicardial coronary vessel is considered hemodynamically significant, impairing regional perfusion and creating an imminent risk of acute coronary syndrome (myocardial infarction).

```
                            EPICARDIAL CORONARY VASCULATURE
                                     ┌─────────────┐
                                     │ Ascending   │
                                     │    Aorta    │
                                     └──────┬──────┘
                       ┌────────────────────┴────────────────────┐
                       ▼                                         ▼
            ┌──────────────────────┐                  ┌──────────────────────┐
            │ Left Coronary Artery │                  │ Right Coronary Artery│
            │     (LCA Trunk)      │                  │        (RCA)         │
            └──────────┬───────────┘                  └──────────┬───────────┘
                 ┌─────┴─────┐                                   │
                 ▼           ▼                                   ▼
          ┌─────────────┐ ┌─────────────┐                 ┌─────────────┐
          │     LAD     │ │     LCX     │                 │ Right Sulcus│
          │ Anterior    │ │ Lateral/    │                 │ RV & Nodal  │
          │ Groove &    │ │ Posterior   │                 │ Conduction  │
          │ Ventricles  │ │ Left Wall   │                 │ Supply      │
          └─────────────┘ └─────────────┘                 └─────────────┘
```

### The Three Target Coronary Arteries
* **LAD (Left Anterior Descending):** Arising from the LCA trunk, the LAD travels down the anterior interventricular sulcus toward the cardiac apex. It supplies $45\text{--}55\%$ of the left ventricular myocardium and the interventricular septum. Occlusion of the LAD is clinically dubbed the "widow-maker" due to high mortality.
* **LCX (Left Circumflex):** Curves to the left within the atrioventricular sulcus, beneath the left auricle, supplying the lateral and posterolateral walls of the left ventricle.
* **RCA (Right Coronary Artery):** Originates from the right aortic sinus and courses through the right coronary sulcus between the right atrium and right ventricle. It perfuses the right ventricle, inferior left ventricular wall, and crucial electrical conduction centers (the Sinoatrial [SA] node in $60\%$ of individuals and Atrioventricular [AV] node in $90\%$).

### Clinical Need for Spatial Localization
Existing computational tools (Framingham, ASCVD) treat cardiovascular risk as a monolithic systemic percentage. In clinical triage, however, an elevated overall risk score provides no guidance on whether an intervention requires revascularization of the anterior septum (LAD), the posterior wall (LCX), or the inferior conduction system (RCA). CardioRisk bridges this gap by marrying multi-target probability estimation with 3D anatomical visualization.

---

## 2. Dataset Description & Preprocessing Pipeline

### 2.1 The Clinical Cohort
The predictive models were developed on the **Z-Alizadeh Sani Coronary Artery Disease Dataset**, collected at the Tehran Heart Center (archived on the UC Irvine Machine Learning Repository). The dataset contains comprehensive medical records for **$N = 303$ consecutive patients** referred for catheterization-based coronary angiography (the clinical ground-truth gold standard).

### 2.2 Feature Categorization (54 Input Features)
The feature space spans six clinical domains:
1. **Demographics (5 features):** Age, Sex, Weight, Length (Height), Body Mass Index (BMI).
2. **Medical History (12 features):** Diabetes Mellitus (DM), Hypertension (HTN), Current Smoker, Ex-Smoker, Family History (FH), Obesity, Chronic Renal Failure (CRF), Cerebrovascular Accident (CVA), Airway Disease, Thyroid Disease, Congestive Heart Failure (CHF), Dyslipidemia (DLP).
3. **Vitals & Clinical Symptoms (13 features):** Systolic Blood Pressure (BP), Pulse Rate (PR), Edema, Weak Peripheral Pulse, Lung Rales, Systolic Murmur, Diastolic Murmur, Typical Chest Pain, Dyspnea, NYHA Function Class, Atypical Angina, Nonanginal Chest Pain, Exertional Angina (LowTH Ang).
4. **Electrocardiography [ECG] (7 features):** Q-Wave presence, ST Elevation, ST Depression, T-Wave Inversion, Left Ventricular Hypertrophy (LVH), Poor R Progression, Bundle Branch Block (BBB: Normal, LBBB, RBBB).
5. **Laboratory Biomarkers (14 features):** Fasting Blood Sugar (FBS), Serum Creatinine (CR), Triglycerides (TG), Low-Density Lipoprotein (LDL), High-Density Lipoprotein (HDL), Blood Urea Nitrogen (BUN), Erythrocyte Sedimentation Rate (ESR), Hemoglobin (HB), Potassium (K), Sodium (Na), White Blood Cell Count (WBC), Lymphocyte Count, Neutrophil Count, Platelet Count (PLT).
6. **Echocardiography (3 features):** Transthoracic Ejection Fraction (EF-TTE), Regional Wall Motion Abnormality (Region RWMA), Valvular Heart Disease (VHD: Normal, Mild, Moderate, Severe).

### 2.3 Strict Target Data Leakage Prevention
A critical pitfall in clinical machine learning is target contamination. In our preprocessing pipeline ([`preprocessing.py`](file:///C:/Users/Amey%20Ballal/OneDrive/Desktop/CardioRisk/preprocessing.py)):
* Ground-truth catheterization labels (`Cath`, `LAD`, `LCX`, `RCA`) were cleanly extracted into a dedicated label matrix $\mathbf{Y} \in \{0, 1\}^{N \times 4}$ and **completely eliminated** from the feature matrix $\mathbf{X}$.
* Binary targets were formalized: $\text{CAD} = 1$ if catheterization indicated $\ge 50\%$ stenosis in $\ge 1$ major vessel; $\text{LAD}, \text{LCX}, \text{RCA} = 1$ if the respective vessel was stenotic.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA PREPROCESSING ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Input Features X (54 columns)                                               │
│    │                                                                        │
│    ├── Numeric (21 cols)      ──> SimpleImputer(median) ──> StandardScaler  │
│    ├── Binary (29 cols)       ──> SimpleImputer(most_frequent)              │
│    └── Categorical (4 cols)   ──> SimpleImputer(most_frequent) ──> OHE      │
│                                                                        │    │
│ Fitted exclusively on X_train (80%, N=242) ──> Bundled in preprocessor.joblib│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Automated Imputation for Real-World Clinical Usability
In real clinical workflows, patients often present with incomplete laboratory panels. Rather than throwing errors or failing, our pipeline embeds median/mode imputation trained on the cohort distribution. Missing laboratory tests are safely imputed using population medians, allowing inference with partial clinical inputs.

---

## 3. Predictive Modeling & Machine Learning Architecture

```
                    MULTI-TARGET ENSEMBLE ARCHITECTURE
                                    X (Patient Data)
                                           │
                                           ▼
                                 ┌───────────────────┐
                                 │   Preprocessor    │
                                 │Pipeline (Scaling, │
                                 │ Imputation, OHE)  │
                                 └─────────┬─────────┘
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
           │   CAD Model     │    │   LAD Model     │    │   RCA Model     │
           │  Random Forest  │    │  Random Forest  │    │  L2-Logistic    │
           │(200 trees, d=15)│    │(200 trees, d=10)│    │   (C = 0.1)     │
           └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                    │                      │                      │
                    ▼                      ▼                      ▼
           ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
           │ Platt Sigmoid   │    │ Platt Sigmoid   │    │ Platt Sigmoid   │
           │  Calibration    │    │  Calibration    │    │  Calibration    │
           └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                    ▼                      ▼                      ▼
              P(CAD) in %            P(LAD) in %            P(RCA) in %
```

### 3.1 Model Search Space & Selection
Model selection was conducted across candidate classifiers:
1. **Random Forest:** Non-linear decision trees with bootstrap aggregation (bagging) and random subspace feature sampling.
2. **Support Vector Machines (SVC):** Linear and RBF kernel boundary estimation.
3. **L2-Regularized Logistic Regression:** Linear risk mapping with Ridge shrinkage.
4. **XGBoost:** Gradient-boosted decision trees with depth and learning rate shrinkage.

Hyperparameter tuning utilized **Stratified 5-Fold Cross-Validation** with `RandomizedSearchCV`, maximizing out-of-fold ROC-AUC:

$$\text{Hyperparameters Trained} = \arg\max_{\theta} \frac{1}{K}\sum_{k=1}^K \text{AUC}_{\text{val}, k}(\theta)$$

### 3.2 Post-Hoc Probability Calibration
In medicine, raw machine learning outputs often represent uncalibrated scores rather than true probabilities. For example, an uncalibrated Random Forest frequently clusters probabilities near $0.2$ or $0.8$. We integrated **Platt Sigmoid Calibration** (`CalibratedClassifierCV(method='sigmoid', cv=5)`):

$$P(Y=1 \mid f(x)) = \frac{1}{1 + \exp\left(A \cdot f(x) + B\right)}$$

where scalar parameters $A$ and $B$ are fit via maximum likelihood on cross-validation folds. This guarantees that when CardioRisk outputs an estimated risk of $60\%$, approximately $60$ of $100$ such patients empirically have stenosis.

### 3.3 Overfitting Control & Generalization Analysis
With $N_{\text{train}} = 242$ rows and $P = 54$ features, the sample-to-feature ratio is approximately **$4 : 1$**, creating a potential high-variance regime. To mitigate overfitting:
* **LAD Variance Control:** Hyperparameter optimization selected `max_depth = 10` and `min_samples_leaf = 4`. Restricting leaf nodes to at least 4 patients prevented trees from isolating idiosyncratic patient profiles.
* **RCA Model Selection:** For the RCA, cross-validation selected regularized `LogisticRegression` with heavy L2 penalty ($C = 0.1$, equivalent to regularization strength $\lambda = 10$). Consequently, the gap between Cross-Validation AUC ($0.7299$) and Hold-Out Test AUC ($0.7171$) is only **$0.0128$ ($1.3\%$)**, confirming virtually zero overfitting.
* **Ensemble Subsampling:** Random Forests sampled $\sqrt{54} \approx 7$ features per split (`max_features = 'sqrt'`), de-correlating constituent estimators.

---

## 4. Quantitative Results & Interpretability

### 4.1 Evaluation Benchmark on Independent Test Set ($N=61$)

| Target Vessel | Final Selected Model | 5-Fold CV AUC | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall CAD** | `RandomForest` ($200$ trees, depth $15$) | **$0.9305$** | **$0.8760$** | **$88.52\%$** | **$87.50\%$** | **$97.67\%$** | **$0.9231$** |
| **LAD** | `RandomForest` ($200$ trees, depth $10$, leaf $\ge 4$) | **$0.8519$** | **$0.7985$** | **$72.13\%$** | **$70.73\%$** | **$85.29\%$** | **$0.7733$** |
| **LCX** | `RandomForest` ($100$ trees, depth $10$, leaf $\ge 2$) | **$0.7647$** | **$0.6989$** | **$72.13\%$** | **$76.47\%$** | **$50.00\%$** | **$0.6047$** |
| **RCA** | `LogisticRegression` (L2 penalty, $C=0.1$) | **$0.7299$** | **$0.7171$** | **$63.93\%$** | **$41.67\%$** | **$25.00\%$** | **$0.3125$** |

### 4.2 Key Findings
1. **CAD Detection:** Demonstrates high diagnostic performance ($87.60\%$ Test AUC and $97.67\%$ Recall), establishing high sensitivity as an initial clinical screening tool.
2. **LAD Detection:** High sensitivity ($85.29\%$ Recall) for the most clinically critical vessel.
3. **LCX & RCA Imbalance:** Demonstrates lower recall on isolated branches due to low positive prevalence in the 303-patient cohort, where stenosis more frequently presents in the LAD.

### 4.3 Clinical Explainability via TreeSHAP
To ensure transparency, global feature attributions were computed using TreeSHAP ($\text{SHapley Additive exPlanations}$):

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

```
Top Predictive Clinical Drivers (SHAP Value Impact):
1. Typical Chest Pain       ████████████████████ (Strongest acute ischemic driver)
2. Region RWMA (Wall Motion)████████████████     (Primary echocardiographic indicator)
3. Ejection Fraction (EF)   ██████████████       (Inverse correlation with disease severity)
4. Patient Age              ████████████         (Baseline demographic risk scaling)
5. Blood Pressure (BP)      ██████████           (Vascular resistance & endothelial stress)
6. Fasting Blood Sugar (FBS)████████             (Metabolic & microvascular marker)
```

---

## 5. Interactive 3D Anatomical Visual Mapping Engine

```
                             3D GRAPHICS PIPELINE
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ Dundee Heart GLB       │      │ Three.js Scene Graph   │      │ Camera View Projection │
│ 38 Sub-meshes          │─────>│ -90° X-Matrix Evaluated│─────>│ P_screen =             │
│ Artery Hierarchy       │      │ Local Anchors Attached │      │ project(P_world, Cam)  │
└────────────────────────┘      └────────────────────────┘      └───────────┬────────────┘
                                                                            │
                                                                            ▼
                                                                ┌────────────────────────┐
                                                                │ Dynamic SVG Overlay    │
                                                                │ - SVG Arrow Lines      │
                                                                │ - Pulsing Target Dots  │
                                                                │ - Floating Badges      │
                                                                └────────────────────────┘
```

### 5.1 3D Mesh Asset Architecture
The visualization utilizes a realistic human cardiac model created by medical artist **Hannah Newey** for the **University of Dundee**. The GLTF model comprises 38 sub-meshes representing the pericardium, myocardium, atria, ventricles, aorta, pulmonary trunk, and epicardial vasculature.

### 5.2 Mathematical Coordinate Resolution
In Sketchfab GLTF exports, `Node 0 (Sketchfab_model)` incorporates an internal coordinate rotation matrix:
$$\mathbf{M}_{\text{root}} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
Transforming raw local points without traversing the scene graph causes $Y$ and $Z$ coordinates to swap, incorrectly pointing arrows $90^\circ$ upward into the great vessels.

**CardioRisk** solves this by parenting dedicated `THREE.Object3D` anchor nodes directly inside the inner mesh node (`SubTool-0-12100944.OBJ`). When computing world coordinates:
$$\mathbf{P}_{\text{world}} = \text{Anchor}.\text{matrixWorld} \cdot [0, 0, 0, 1]^T$$
Three.js automatically applies the full transformation chain:
1. Inner mesh identity matrix.
2. Root node $-90^\circ$ rotation matrix.
3. Model centering and dynamic scale factor ($S = 5.2 / \text{maxDim}$).
4. User OrbitControls dynamic rotation and zoom.

Projecting $\mathbf{P}_{\text{world}}$ through camera intrinsic/extrinsic parameters:
$$\mathbf{P}_{\text{NDC}} = \mathbf{P}_{\text{world}} \times \mathbf{M}_{\text{view}} \times \mathbf{M}_{\text{proj}}$$
$$\text{Target}_X = \left(\frac{\mathbf{P}_{\text{NDC}, x} + 1}{2}\right) \times W_{\text{window}}, \quad \text{Target}_Y = \left(\frac{-\mathbf{P}_{\text{NDC}, y} + 1}{2}\right) \times H_{\text{window}}$$

### 5.3 Texture Preservation & Visual Integrity
Simple vertex-color tinting across shared mesh buffers (`Object_17-20`) causes the entire arterial and venous tree across the heart to turn yellow or red. **CardioRisk completely eliminates vertex color tampering**, preserving 100% of the natural anatomical textures. Risk status is rendered exclusively through:
* **Dynamic SVG Leader Lines:** Connecting badge boundaries to coordinates $(\text{Target}_X, \text{Target}_Y)$ with glowing drop shadows.
* **Pulsing Target Rings:** Positioned directly over the physical coronary vessels.
* **Non-Overlapping Layout:** RCA (`dx: -165, dy: -15`), LCX (`dx: +165, dy: -65`), LAD (`dx: +165, dy: +65`).

---

## 6. System Integration, API & Deployment

### 6.1 Backend API Specification (FastAPI)
The backend exposes a high-throughput REST endpoint (`POST /predict`):

```json
// Request Payload (JSON)
{
  "features": {
    "Age": 62, "Sex": 1.0, "Weight": 82, "Length": 174, "BMI": 27.1,
    "DM": 1.0, "HTN": 1.0, "Current Smoker": 1.0, "BP": 145, "PR": 78,
    "Typical Chest Pain": 1.0, "EF-TTE": 42, "Region RWMA": 2.0,
    "FBS": 138, "CR": 1.1, "TG": 180, "LDL": 145, "HDL": 34
  }
}

// Response Payload (JSON)
{
  "status": "success",
  "risk_scores": {
    "CAD": 68.42,
    "LAD": 61.15,
    "LCX": 38.20,
    "RCA": 29.80
  },
  "message": "Predictions generated successfully."
}
```

### 6.2 Clinical Decision Support Triage Standards

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RISK THRESHOLD     SEVERITY LEVEL             RECOMMENDED CLINICAL ACTION   │
├─────────────────────────────────────────────────────────────────────────────┤
│ < 30% (Green)      Healthy Baseline / Low     Lifestyle counseling & annual │
│                                               preventive monitoring.        │
│                                                                             │
│ 30% - 50% (Yellow) Borderline / Moderate      Closer surveillance, non-     │
│                                               invasive stress ECG / CCTA.   │
│                                                                             │
│ > 50% (Red)        High Stenosis Risk         Urgent cardiology referral &  │
│                                               Coronary Angiography (ICA).   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Containerization & Production Cloud Run Setup
CardioRisk packages into a single portable Docker container serving both the API and WebGL assets:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY cardiorisk_outputs/models/ ./cardiorisk_outputs/models/
COPY frontend/ ./frontend/
WORKDIR /app/backend
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Deployable to **Google Cloud Run** with a single command:
```bash
gcloud run deploy cardiorisk --source . --region us-central1 --allow-unauthenticated --memory 2Gi
```

---

## 7. Limitations, Clinical Safety & Future Work

### 7.1 Limitations
* **Cohort Demographics:** Sourced from a single tertiary heart center (Tehran Heart Center). Multicenter validation across ethnically diverse cohorts is needed to assess dataset shift.
* **Secondary Branch Resolution:** Current models classify major epicardial trunks (LAD, LCX, RCA); smaller diagonal, septal perforator, and obtuse marginal branches are not independently classified.

### 7.2 Clinical Safety Disclaimer
> **MANDATORY CLINICAL NOTICE:** CardioRisk is developed for **clinical decision support, research, and educational visualization only**. It does not constitute formal diagnostic software and must not be used as a standalone substitute for clinical judgment, CT coronary angiography (CCTA), or invasive coronary angiography (ICA).

### 7.3 Future Directions
1. **DICOM / CT Angiography Fusion:** Incorporating patient-specific CT segmentations to render personalized 3D geometries rather than a normalized reference mesh.
2. **Computational Fluid Dynamics (CFD):** Integrating Fractional Flow Reserve (FFR) fluid dynamic simulations onto the 3D model.

---

## References & Academic Citations

1. **Alizadehsani, R., et al.** (2013). *A database for using machine learning and data mining techniques for coronary artery disease diagnosis.* Scientific Data / UCI Machine Learning Repository.
2. **Newey, H., & Reid, L.** (2018). *Cardiac Anatomy: Coronary arteries of the heart.* University of Dundee, Centre for Anatomy and Human Identification (CAHID).
3. **Platt, J. C.** (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.* Advances in Large Margin Classifiers.
4. **Lundberg, S. M., & Lee, S.-I.** (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017).
5. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
6. **Cabello, R., et al.** (2021). *Three.js: JavaScript 3D Library.* https://threejs.org/
