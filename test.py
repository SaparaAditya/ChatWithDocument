import os

def load_faiss_index():
    allow_dangerous_deserialization = True  # Set to True only if you trust the source
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index_path = "faiss_index"

    if os.path.exists(faiss_index_path):
        if os.path.isdir(faiss_index_path):
            # If the index is a directory, load it using the directory path
            try:
                faiss_index = FAISS.deserialize_index(faiss_index_path)
                faiss_index.add(embeddings)
                return faiss_index
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
                return None
        else:
            # If the index is a file, open it and load it using the file object
            try:
                with open(faiss_index_path, "rb") as f:
                    faiss_index = FAISS.deserialize_index(f)
                    faiss_index.add(embeddings)
                    return faiss_index
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
                return None
    else:
        st.error("FAISS index file not found.")
        return None
