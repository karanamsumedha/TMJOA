import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------------
# LOAD MULTIPLE DOCUMENTS
# -------------------------------
def load_documents(folder_path):
    texts = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                texts.append(f.read())

    return texts


# -------------------------------
# CREATE VECTOR DATABASE
# -------------------------------
def create_vector_db(texts):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    docs = []
    for text in texts:
        docs.extend(splitter.create_documents([text]))

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"   # 🔥 upgraded model
    )

    db = FAISS.from_documents(docs, embeddings)
    return db


# -------------------------------
# RETRIEVE CONTEXT
# -------------------------------
def retrieve_context(db, feature):
    query = f"TMJ osteoarthritis impact of {feature}"

    results = db.similarity_search(query, k=2)

    return [doc.page_content for doc in results]


# -------------------------------
# MAIN TEST
# -------------------------------
if __name__ == "__main__":
    folder_path = "data/knowledge"

    texts = load_documents(folder_path)
    db = create_vector_db(texts)

    query_feature = "BVTV_change"
    contexts = retrieve_context(db, query_feature)

    print("\nRetrieved Context:\n")
    for c in contexts:
        print("-", c)