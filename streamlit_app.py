import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import torch

# Ensure the device is set correctly for PyTorch (if you're using a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = pipeline("text-generation", model="distilgpt2", device=device)

# Define a simple medical knowledge base (this can be expanded)
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten causes damage to the small intestine. "
    "The disease is triggered by the body's immune response to gluten, leading to inflammation and damage to the villi. "
    "Celiac disease requires a lifelong gluten-free diet to manage symptoms and prevent complications.",
    "Symptoms of celiac disease include diarrhea, weight loss, abdominal pain, and fatigue. It can also lead to skin rashes, bone pain, and mood changes.",
    "A gluten-free diet is essential for managing celiac disease. Avoiding gluten helps heal the damaged small intestine and prevent further health problems.",
    "Gluten is a protein found in wheat, barley, and rye. Individuals with celiac disease must completely avoid foods containing these grains."
]

# Function to encode text into embeddings using a pre-trained model
def encode_text(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Use eos_token_id for padding
    tokenizer.pad_token = tokenizer.eos_token      # Padding will use eos_token
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.transformer.wte(inputs.input_ids.to(device))
    return outputs.mean(dim=1).detach().cpu().numpy()

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

# Function to generate a response based on the query and relevant context
def generate_response(query):
    # Retrieve relevant context from the knowledge base
    context = retrieve_info(query)
    
    # Format the prompt with clear structure and line breaks
    prompt = (
        f"User Query:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    
    # Generate the response with constraints
    response = generator(
        prompt,
        max_length=150,  # Limit length for concise answers
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    # Extract the text and clean it up
    generated_text = response[0]["generated_text"].strip()
    
    # Extract only the "Answer" part and remove redundant lines
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()
    
    # Post-process to remove abrupt endings or unrelated lines
    sentences = generated_text.split(". ")
    final_response = ". ".join(sentence for sentence in sentences if "gluten" in sentence or "celiac" in sentence)
    
    # Ensure the final response ends logically
    if not final_response.endswith("."):
        final_response += "."
    
    return final_response


# Streamlit Appearance Setup
st.set_page_config(
    page_title="Medical Advisor 🤖",
    page_icon="⚕️",
    layout="wide",
)

# App Header
st.title("Cel.AI🤖")
st.subheader("The Gluten Intolerance GPT")
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
                # Generate the AI response based on the user's query
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
st.markdown("Made with ❤️ for all Celiac Patients and health freaks out there.")
