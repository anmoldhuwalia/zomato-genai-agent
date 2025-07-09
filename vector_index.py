import os
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# ✅ Set your API key temporarily here for local run
openai_key = "sk-..."  # REPLACE for local use

def build_vector_db():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    docs = []

    for fname in os.listdir("data/menus"):
        if fname.endswith(".json"):
            with open(os.path.join("data/menus", fname), "r") as f:
                data = json.load(f)
                restaurant = data.get("restaurant_name", fname)
                for item in data.get("menu", []):
                    text = f"{item['item']} - {item['description']} - ₹{item['price']} - {item.get('calories', '')} cal"
                    docs.append(Document(page_content=text, metadata={"restaurant": restaurant}))

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("vector_db")
    print("✅ Vector DB created and saved to ./vector_db/")

if __name__ == "__main__":
    build_vector_db()
