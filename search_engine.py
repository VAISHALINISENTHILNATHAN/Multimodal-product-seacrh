# search_engine.py
import os
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# -----------------------------
# Paths
# -----------------------------
EMBED_PATH = r"C:\Users\ADMIN\multimodal product search\embeddings\product_vectors.npy"
ID_PATH = r"C:\Users\ADMIN\multimodal product search\embeddings\product_ids.npy"
IMAGES_FOLDER = r"C:\Users\ADMIN\multimodal product search\data\raw\images"

# -----------------------------
# Load embeddings and IDs
# -----------------------------
embeddings = np.load(EMBED_PATH)
product_ids = np.load(ID_PATH)

print("Embeddings shape:", embeddings.shape)
print("Number of products:", len(product_ids))

# -----------------------------
# Initialize FAISS (cosine similarity)
# -----------------------------
dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print("✅ FAISS index built. Total vectors:", index.ntotal)

# -----------------------------
# Load CLIP model
# -----------------------------
device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# -----------------------------
# Embed query (text/image)
# -----------------------------
def embed_query(text=None, image_path=None):
    with torch.no_grad():
        txt_emb = None
        img_emb = None

        if text:
            text_inputs = processor(text=[text], return_tensors="pt", padding=True)
            for k in text_inputs:
                text_inputs[k] = text_inputs[k].to(device)
            txt_emb = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
            txt_emb = txt_emb.cpu().numpy()

        if image_path:
            try:
                img = Image.open(image_path).convert("RGB").resize((224,224))
                img_inputs = processor(images=[img], return_tensors="pt")
                for k in img_inputs:
                    img_inputs[k] = img_inputs[k].to(device)
                img_emb = model.get_image_features(img_inputs["pixel_values"])
                img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
                img_emb = img_emb.cpu().numpy()
            except:
                img_emb = None

        if txt_emb is not None and img_emb is not None:
            query_emb = 0.5 * txt_emb + 0.5 * img_emb
        elif txt_emb is not None:
            query_emb = txt_emb
        elif img_emb is not None:
            query_emb = img_emb
        else:
            raise ValueError("No text or image provided for query!")

        faiss.normalize_L2(query_emb)
        return query_emb

# -----------------------------
# Search function
# -----------------------------
def search(query_text=None, query_image_path=None, top_k=5):
    query_emb = embed_query(text=query_text, image_path=query_image_path)
    D, I = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "product_id": product_ids[idx],
            "image_path": os.path.join(IMAGES_FOLDER, f"{product_ids[idx]}.jpg"),
            "score": float(score)
        })
    return results

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    print("🔹 Text search example:")
    for r in search(query_text="red t-shirt", top_k=5):
        print(r)

    example_img = os.path.join(IMAGES_FOLDER, "1163.jpg")  # replace with a valid image
    print("\n🔹 Image search example:")
    for r in search(query_image_path=example_img, top_k=5):
        print(r)

    print("\n🔹 Multimodal search example:")
    for r in search(query_text="blue sneakers", query_image_path=example_img, top_k=5):
        print(r)
