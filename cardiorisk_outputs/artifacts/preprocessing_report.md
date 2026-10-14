# Preprocessing Report

## Dataset Summary
- **Original rows**: 303
- **Processed rows** (after duplicate removal): 303
- **Features**: 54
- **Targets**: ['CAD', 'LAD', 'LCX', 'RCA']

## Feature Groups
- **Binary (29)**: Sex, DM, HTN, Current Smoker, EX-Smoker, FH, Obesity, CRF, CVA, Airway disease, Thyroid Disease, CHF, DLP, Edema, Weak Peripheral Pulse, Lung rales, Systolic Murmur, Diastolic Murmur, Typical Chest Pain, Dyspnea, Atypical, Nonanginal, LowTH Ang, Q Wave, St Elevation, St Depression, Tinversion, LVH, Poor R Progression
- **Categorical (4)**: Function Class, BBB, Region RWMA, VHD
- **Numeric (21)**: Age, Weight, Length, BMI, BP, PR, FBS, CR, TG, LDL, HDL, BUN, ESR, HB, K, Na, WBC, Lymph, Neut, PLT, EF-TTE

## Pipeline Transformations
- **Numeric**: Median Imputation + Standard Scaling
- **Binary**: Most Frequent Imputation (String classes mapped to 0/1)
- **Categorical**: Most Frequent Imputation + One-Hot Encoding (sparse_output=False, ignore unknown)

## Data Leakage Addressed
Targets `Cath`, `LAD`, `LCX`, `RCA` successfully extracted and excluded from feature set.
