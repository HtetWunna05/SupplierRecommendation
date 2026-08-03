import pandas as pd
import os

def run_cleaning():
    print("\n🚀 PROCESSING US SUPPLY CHAIN DATA...")
    file_path = "data/dataco_data.csv"
    
    if not os.path.exists(file_path):
        print("❌ Error: dataco_data.csv not found!")
        return

    try:
        # 1. Load the file
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"📊 Total raw rows found: {len(df)}")

        # 2. FIXED MAPPING (Based on your actual columns)
        # We map specific columns to the names our app expects
        mapping = {
            'Supplier_ID': 'supplier',
            'Product_Category': 'part_number',
            'Delay_Days': 'avg_lead_time',
            'Supplier_Reliability_Score': 'reliability_score'
        }

        # Create the cleaned dataframe
        df_clean = df[list(mapping.keys())].rename(columns=mapping).copy()

        # 3. DATA CLEANING
        # Ensure numbers are actually numbers (important for math!)
        df_clean['avg_lead_time'] = pd.to_numeric(df_clean['avg_lead_time'], errors='coerce').fillna(0)
        df_clean['reliability_score'] = pd.to_numeric(df_clean['reliability_score'], errors='coerce').fillna(90)
        
        # Add the quality score required by database.py
        df_clean['quality_impact_score'] = 100.0

        # 4. THE CALCULATION (Safety Check)
        # Now that avg_lead_time is a number, this won't crash!
        df_clean['consistency_index'] = (df_clean['reliability_score'] / (df_clean['avg_lead_time'] + 1)).round(2)

        # 5. SAVE
        df_clean.to_csv("data/cleaned_master_data.csv", index=False)
        print(f"🏁 SUCCESS: Processed {len(df_clean)} rows.")
        return df_clean

    except Exception as e:
        print(f"❌ A processing error occurred: {e}")
        return None

if __name__ == "__main__":
    run_cleaning()