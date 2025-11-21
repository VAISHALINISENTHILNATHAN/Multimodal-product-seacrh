#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import os
from search_engine import search

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("🛍 Multimodal Product Search")
st.write("Search products using text, image, or both!")

# -----------------------------
# Inputs
# -----------------------------
query_text = st.text_input("Enter product description")
query_image = st.file_uploader("Upload product image", type=["jpg", "png"])

# Temporary path for uploaded image
img_path = None
if query_image:
    img_path = os.path.join("temp", query_image.name)
    os.makedirs("temp", exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(query_image.getbuffer())

top_k = st.slider("Number of results", 1, 20, 5)

# -----------------------------
# Search button
# -----------------------------
if st.button("Search"):
    if not query_text and not img_path:
        st.warning("Please enter text or upload an image to search.")
    else:
        with st.spinner("Searching..."):
            results = search(query_text=query_text, query_image_path=img_path, top_k=top_k)

        st.success(f"Found {len(results)} results")
        for r in results:
            st.write(f"**Product ID:** {r['product_id']}, **Score:** {r['score']:.4f}")
            if r['image_path'] and os.path.exists(r['image_path']):
                st.image(r['image_path'], width=150)

# -----------------------------
# Cleanup temporary files
# -----------------------------
if img_path and os.path.exists(img_path):
    os.remove(img_path)


# In[ ]:




