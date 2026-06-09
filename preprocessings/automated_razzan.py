import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Gagal memuat data: {e}")
        return None

def preprocess_data(df):
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    cat_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed",
        "Credit_History"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    num_cols = [
        "LoanAmount",
        "Loan_Amount_Term"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Encoding data kategorikal
    le = LabelEncoder()
    encode_cols = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status",
        "Dependents"
    ]
    for col in encode_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    INPUT_PATH = "dataset_raw/loan_data.csv"
    OUTPUT_DIR = "dataset_preprocessing"
    OUTPUT_PATH = f"{OUTPUT_DIR}/loan_data_clean.csv"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_data = load_data(INPUT_PATH)
    if raw_data is not None:
        clean_data = preprocess_data(raw_data)
        save_data(clean_data, OUTPUT_PATH)