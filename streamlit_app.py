import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import torch

# Set device for PyTorch (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DistilGPT-2 model (lighter version of GPT-2) for text generation
generator = pipeline("text-generation", model="distilgpt2", device=device)

# Short knowledge base for celiac and general health information
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten causes damage to the small intestine. It requires a gluten-free diet.",
    "Symptoms of celiac disease include diarrhea, weight loss, and abdominal pain. It is important to avoid gluten-containing foods.",
    "A gluten-free diet helps heal the small intestine and prevents complications in individuals with celiac disease.",
    "Gluten is a protein found in wheat, barley, and rye. Celiac patients must completely avoid these grains.",
    "Diabetes is a chronic condition where the body struggles to regulate blood sugar levels. Lifestyle changes and medication can help manage it.",
    "Hypertension is high blood pressure, which increases the risk of heart disease and stroke. Managing it involves lifestyle changes and sometimes medication.",
    "Mental health is essential for overall well-being, and conditions like anxiety or depression can significantly impact daily life.",
    "Hydration is crucial for bodily functions like digestion, nutrient absorption, and maintaining healthy skin."
]

# Function to encode text into embeddings using a pre-trained model
def encode_text(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Use eos_token_id for padding
    tokenizer.pad_token = tokenizer.eos_token  # Padding will use eos_token
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=512)
    outputs = model.transformer.wte(inputs.input_ids)  # Correct access to model embedding weights
    return outputs.mean(dim=1).detach().cpu().numpy()

# Encode the knowledge base and create a FAISS index for efficient similarity search
encoded_kb = encode_text(knowledge_base)
index = faiss.IndexFlatL2(encoded_kb.shape[1])
index.add(np.array(encoded_kb))

# Function to retrieve relevant information based on user query
def retrieve_info(query):
    query_vec = encode_text([query])
    D, I = index.search(query_vec, k=3)  # Retrieve top 3 relevant texts
    relevant_info = [knowledge_base[i] for i in I[0]]  # Retrieve information from the knowledge base
    return " ".join(relevant_info)  # Combine the retrieved texts into a single string for context

# Function to generate a response based on the query and relevant context
def generate_response(query):
    # Retrieve relevant context from the knowledge base
    context = retrieve_info(query)
    
    # Format the prompt to emphasize the context is related to the query
    prompt = (
        f"User Query: {query}\n"
        f"Context: {context}\n\n"
        "Answer the query based on the context provided. Be specific and relevant to the topic."
    )
    
    # Generate the response using the pre-trained model
    eos_token_id = tokenizer.convert_tokens_to_ids(".")  # Full stop (.) as EOS token
    response = generator.generate(
        input_ids=tokenizer(prompt, return_tensors="pt").input_ids.to(device),
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,  # Moderate creativity for detailed answers
        repetition_penalty=1.2,  # Avoid repetitive phrases
        pad_token_id=eos_token_id,  # Full stop as pad token
        eos_token_id=eos_token_id  # Ensure the response ends with a full stop
    )
    
    # Decode the response and clean it
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True).strip()
    
    return generated_text

# Streamlit Appearance Setup
st.set_page_config(
    page_title="Baymax here!",
    page_icon="⚕️",
    layout="wide",
)

# App Header
st.title("Baymax🩺")
st.subheader("Your Personal Healthcare Companion")
st.markdown("""
Welcome to **Baymax**, your personal healthcare companion! 🌟  
Ask me anything about health, wellness, or medical concerns.  
I use advanced AI and a curated knowledge base to provide accurate, helpful responses.
""")

# Sidebar Configuration
st.sidebar.image(
    "https://i.pinimg.com/originals/3a/a8/51/3aa851a0f34d6703c7f0ac7ff6a41e8a.png",
    caption="Baymax: Your Personal Healthcare Companion",
    use_column_width=True
)

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
**Example:** "What are the symptoms of diabetes?"
""")
st.markdown("Made with ❤️ for your health and well-being. Love, Baymax")
