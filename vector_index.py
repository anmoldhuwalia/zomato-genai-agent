import os, json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def build_vector_db():
    embeddings = OpenAIEmbeddings()
    docs = []
    for fname in os.listdir("data/menus"):
        if fname.endswith(".json"):
            menu = json.load(open(f"data/menus/{fname}"))
            docs.append({"page_content": json.dumps(menu), "metadata":{"source":fname}})
    db = FAISS.from_documents([langchain.schema.Document(**d) for d in docs], embeddings)
    db.save_local("vector_db")