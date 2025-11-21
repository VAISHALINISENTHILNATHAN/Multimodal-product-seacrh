# generate_embeddings.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = r"C:\Users\ADMIN\multimodal product search\data\processed\products_clean.csv"
IMAGES_FOLDER = r"C:\Users\ADMIN\multimodal product search\data\raw\images"
EMBED_PATH = r"C:\Users\ADMIN\multimodal product search\embeddings\product_vectors.npy"
ID_PATH = r"C:\Users\ADMIN\multimodal product search\embeddings\product_ids.npy"

os.makedirs(os.path.dirname(EMBED_PATH), exist_ok=True)

# -----------------------------
# CPU-only
# -----------------------------
device = "cpu"

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
df = df.sample(n=2000, random_state=42)  # optional sampling for testing
print("Products to embed:", len(df))

# Make sure image paths exist
df["image_path"] = df["image_path"].apply(lambda x: os.path.join(IMAGES_FOLDER, os.path.basename(x)))

# -----------------------------
# Load CLIP
# -----------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# -----------------------------
# Generate embeddings
# -----------------------------
embeddings = []
product_ids = []
batch_size = 8

for i in tqdm(range(0, len(df), batch_size), ncols=100):
    batch = df.iloc[i:i+batch_size]
    images = []
    texts = []
    valid_ids = []

    # Load batch
    for _, row in batch.iterrows():
        text = str(row["description"])
        img_path = row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB").resize((224,224))
        except:
            img = None
        images.append(img)
        texts.append(text)
        valid_ids.append(row["product_id"])

    if not texts:
        continue

    # Text embeddings
    text_inputs = processor(text=texts, return_tensors="pt", padding=True)
    for k in text_inputs:
        text_inputs[k] = text_inputs[k].to(device)
    with torch.no_grad():
        txt_emb = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb.cpu().numpy()

    # Image embeddings
    valid_images = [img for img in images if img is not None]
    img_embs = []
    if valid_images:
        img_inputs = processor(images=valid_images, return_tensors="pt")
        for k in img_inputs:
            img_inputs[k] = img_inputs[k].to(device)
        with torch.no_grad():
            img_emb = model.get_image_features(img_inputs["pixel_values"])
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            img_embs = img_emb.cpu().numpy()

    # Combine text + image embeddings
    combined_emb = []
    img_idx = 0
    for idx, img in enumerate(images):
        if img is not None and len(img_embs) > 0:
            combined_emb.append(0.5 * txt_emb[idx] + 0.5 * img_embs[img_idx])
            img_idx += 1
        else:
            combined_emb.append(txt_emb[idx])
    embeddings.extend(combined_emb)
    product_ids.extend(valid_ids)

# -----------------------------
# Save embeddings
# -----------------------------
embeddings = np.array(embeddings).astype("float32")
product_ids = np.array(product_ids)
np.save(EMBED_PATH, embeddings)
np.save(ID_PATH, product_ids)

print("✅ Embeddings generated and saved successfully!")
print("Total products:", len(product_ids))
print("Embedding dimension:", embeddings[0].shape)
