🛍️ Multimodal Product Search Engine

A complete ML pipeline for text & image product retrieval using embeddings and FAISS.

🚀 Overview

This project is an end-to-end multimodal product search system built to showcase practical machine learning engineering skills. It processes product metadata, generates embeddings (text + image), stores them in a high-performance FAISS index, and exposes a clean Streamlit UI for real-time product search.

Ideal for:

ML engineering portfolios
Demonstrating vector search systems
Ecommerce search & recommendation prototypes
Real-world multimodal retrieval applications

✨ Features
🔍 Metadata Parsing

Cleans & structures product data
Extracts titles, descriptions, attributes, categories
Prepares unified inputs for embedding models

🧠 Embedding Generation

Supports text, image, or fused embeddings
Uses CLIP / Sentence Transformers (configurable)
Stores embeddings + metadata for efficient retrieval

⚡ FAISS Indexing

Builds a scalable vector similarity search index
Supports L2 or cosine similarity
Fast top-K product retrieval

🎨 Streamlit User Interface

Search using text queries or uploaded product images
Displays ranked results with similarity scores
Lightweight, interactive, and deployment-ready

🧱 Project Structure
├── models/1_parse metadata          # source data
├── models/generate_embeddings       # Embedding generation scripts/models
├── search_engine                    # FAISS index build + query logic
├── streamlit_ui/                    # Streamlit UI for search
├── data/                            # Example dataset / sample products
└── README.md

🛠️ Tech Stack

- Python
- PyTorch / CLIP / Sentence Transformers
- FAISS (Facebook AI Similarity Search)
- Streamlit
- NumPy + Pandas

▶️ Getting Started
1. Install dependencies
pip install -r requirements.txt

2. Parse product metadata
python models/1_metadata.py

3. Generate embeddings
python models/generate_embeddings.py

4. Build the FAISS index
python search_engine.py

5. Launch the Streamlit UI
streamlit run streamlit_ui.py

🎯 Example Use Cases

Ecommerce product search
Content-based recommendation systems
Duplicate item detection
Visual search and multimodal retrieval research
