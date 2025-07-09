import streamlit as st
import re
import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# ✅ Load embeddings with secure key
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["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"])

# ✅ Load FAISS vector DB
db = FAISS.load_local("vector_db", embeddings)

# ✅ Setup LangChain retriever chain
retriever = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=st.secrets["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"]),
    chain_type="stuff",
    retriever=db.as_retriever()
)

# ----------------------------
# 🔧 Define mock workflow tools
# ----------------------------

def detect_intent(user_input):
    if match := re.search(r"(cancel|add|track).+order\s*#?(\d+)", user_input.lower()):
        action, order_id = match.groups()
        return action, order_id
    return "faq", None

def cancel_order(order_id):
    return f"✅ Order #{order_id} has been successfully canceled."

def track_order(order_id):
    return f"📦 Order #{order_id} is out for delivery and will arrive soon."

def add_item(order_id, item="Fries"):
    return f"🍟 {item} has been added to Order #{order_id}."

# ✅ LangChain tool wrappers (optional for agent)
tools = [
    Tool.from_function(name="cancel_order", func=cancel_order, description="Cancel a given order by order ID"),
    Tool.from_function(name="track_order", func=track_order, description="Track a given order by order ID"),
    Tool.from_function(name="add_item", func=add_item, description="Add an item to a given order")
]

# ✅ Initialize LLM agent
llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["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"])
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# ----------------------------
# 🎯 Main entry: handle user input
# ----------------------------

def handle_query(user_input):
    intent, order_id = detect_intent(user_input)
    
    if intent == "cancel":
        return cancel_order(order_id)
    elif intent == "track":
        return track_order(order_id)
    elif intent == "add":
        # Extract item (simple pattern match or just assume fries)
        item = "fries"
        return add_item(order_id, item)
    else:
        # FAQ or unknown → use vector retriever
        return retriever.run(user_input)
