import re, json, os
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load vector DB
db = FAISS.load_local("vector_db", OpenAIEmbeddings())

# Tools
def detect_intent(inp):
    if m := re.search(r"(cancel|add|track) order\s*#?(\d+)", inp.lower()):
        action, oid = m.groups()
        return action, oid
    return "faq", None

def cancel_order(order_id): return f"✅ Order #{order_id} canceled!"
def track_order(order_id): return f"🚚 Order #{order_id} is out for delivery."
def add_item(order_id, item): return f"🍟 Added *{item}* to order #{order_id}."

tools = [
    Tool.from_function(func=cancel_order, name="cancel_order", description="Cancels an order"),
    Tool.from_function(func=track_order, name="track_order", description="Tracks an order"),
    Tool.from_function(func=add_item, name="add_item", description="Adds item to order")
]

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")
retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

def handle_query(q):
    intent, oid = detect_intent(q)
    if intent != "faq":
        if intent == "add":
            item = q.split("add")[-1].split("to order")[0].strip()
            return add_item(oid, item)
        return {"cancel":"cancel_order","track":"track_order"}[intent](oid)
    return retriever.run(q)