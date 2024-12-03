import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = pipeline("text-generation", model="distilgpt2")

# Knowledge base
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten causes damage to the small intestine. "
    "The disease is triggered by the body's immune response to gluten, leading to inflammation and damage to the villi. "
    "Celiac disease requires a lifelong gluten-free diet to manage symptoms and prevent complications."
    # Add more sentences as needed...
]

# Function to encode text into embeddings
def encode_text(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.transformer.wte(inputs.input_ids)
    return outputs.mean(dim=1).detach().numpy()

# Encode the knowledge base and create a FAISS index
encoded_kb = encode_text(knowledge_base)
index = faiss.IndexFlatL2(encoded_kb.shape[1])
index.add(np.array(encoded_kb))

# Function to retrieve relevant information based on user query
def retrieve_info(query):
    query_vec = encode_text([query])
    D, I = index.search(query_vec, k=3)
    relevant_info = [knowledge_base[i] for i in I[0]]
    return " ".join(relevant_info)

# Function to generate a response
def generate_response(query):
    context = retrieve_info(query)
    prompt = f"User Query: {query}\nContext: {context}\nAnswer:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"].strip()

# Streamlit Appearance Setup
st.set_page_config(
    page_title="Medical Advisor 🤖",
    page_icon="⚕️",
    layout="wide",
)

# App Header
st.title("Medical Advisor 🤖")
st.markdown("""
Welcome to the **Celiac Disease Assistant**! 🌟  
Ask me anything about celiac disease or related health concerns.  
I use advanced AI and a curated knowledge base to provide accurate responses.
""")

# Sidebar Configuration
st.sidebar.title("Settings ⚙️")
st.sidebar.markdown("Adjust your preferences and explore additional options here!")

# Main Chat Interface
st.subheader("🔍 Ask Your Question")
query = st.text_input("Type your question here:", help="E.g., What are the symptoms of celiac disease?")

if st.button("Get Response 🚀"):
    if query.strip():
        with st.spinner("Thinking... 🤔"):
            try:
                response = generate_response(query)
                st.success("Here's what I found! 🧠")
                st.markdown(f"**{response}**")
            except Exception as e:
                st.error(f"Something went wrong! 😕 Error: {e}")
    else:
        st.warning("Please enter a valid question! 📝")

# Footer
st.markdown("""
---
**Pro Tip:** Use specific queries for the best results!  
**Example:** "What are the symptoms of celiac disease?"
""")
st.markdown("Made with ❤️ using Streamlit and Transformers.")
