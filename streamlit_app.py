import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import torch

# Ensure the device is set correctly for PyTorch (if you're using a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Define a simple medical knowledge base (this can be expanded)
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten causes damage to the small intestine. "
    "The disease is triggered by the body's immune response to gluten, leading to inflammation and damage to the villi. "
    "Celiac disease requires a lifelong gluten-free diet to manage symptoms and prevent complications.",
    "Symptoms of celiac disease include diarrhea, weight loss, abdominal pain, and fatigue. It can also lead to skin rashes, bone pain, and mood changes.",
    "A gluten-free diet is essential for managing celiac disease. Avoiding gluten helps heal the damaged small intestine and prevent further health problems.",
    "Gluten is a protein found in wheat, barley, and rye. Individuals with celiac disease must completely avoid foods containing these grains.",
    # Add more relevant knowledge as needed
]

# Function to encode text into embeddings using a pre-trained model
def encode_text(texts):
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Use eos_token_id for padding
    tokenizer.pad_token = tokenizer.eos_token  # Padding will use eos_token
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = generator.transformer.wte(inputs.input_ids)  # Correct access to model embedding weights
    return outputs.mean(dim=1).detach().cpu().numpy()

# Encode the knowledge base and create a FAISS index
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
    
    eos_token_id = tokenizer.convert_tokens_to_ids(".")  # Full stop (.) as EOS token
    
    # Generate the response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    response = generator.generate(
        input_ids=inputs.input_ids,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,  # Moderate creativity for detailed answers
        repetition_penalty=1.2,  # Avoid repetitive phrases
        pad_token_id=eos_token_id,  # Full stop as pad token
        eos_token_id=eos_token_id,  # Ensure the response ends with a full stop
        truncation=True
    )
    
    # Decode the response and clean it
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True).strip()
    
    # Post-process to remove unwanted content
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:")[-1].strip()
    
    # Filter sentences for relevance and coherence
    sentences = generated_text.split(". ")
    filtered_sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence) > 20 and not sentence.startswith("How does")
    ]  # Keep meaningful sentences and filter out irrelevant ones
    
    final_response = ". ".join(filtered_sentences)
    
    # Ensure the response ends with a full stop
    if not final_response.endswith("."):
        final_response += "."
    
    # Fallback for overly vague or failed responses
    if len(final_response) < 20:
        final_response = "I'm sorry, I couldn't find a suitable answer. Please try rephrasing your question."
    
    return final_response

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
