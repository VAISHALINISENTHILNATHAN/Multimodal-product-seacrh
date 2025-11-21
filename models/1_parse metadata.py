# data_loader.py
import os
import pandas as pd

# -----------------------------
# Paths (absolute)
# -----------------------------
RAW_CSV = r"C:\Users\ADMIN\multimodal product search\styles.csv"
IMAGES_DIR = r"C:\Users\ADMIN\multimodal product search\images"
PROCESSED_CSV = r"C:\Users\ADMIN\multimodal product search\processed\products_clean.csv"

def preprocess_products(raw_csv_path=RAW_CSV, images_dir=IMAGES_DIR, processed_csv_path=PROCESSED_CSV):
    """
    Load styles.csv, extract relevant columns, verify image existence,
    and save cleaned CSV.
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)

    # Load CSV
    df = pd.read_csv(raw_csv_path, on_bad_lines='skip')
    print("Total rows in CSV:", len(df))

    # Extract relevant fields
    if "id" in df.columns and "productDisplayName" in df.columns:
        clean = df[["id","productDisplayName"]].copy()
    else:
        raise ValueError("Expected columns 'id' and 'productDisplayName' not found in CSV.")

    clean = clean.rename(columns={"id":"product_id","productDisplayName":"description"})
    clean["image_path"] = clean["product_id"].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
    clean = clean[clean["image_path"].apply(os.path.exists)]
    print("Total products with images:", len(clean))

    # Save cleaned CSV
    clean.to_csv(processed_csv_path, index=False)
    print(f"Cleaned data saved to: {processed_csv_path}")

    return clean
