import torch
import faiss
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Ensure the device is set correctly for PyTorch (if you're using a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = pipeline("text-generation", model="distilgpt2", device=device)

# Define a simple medical knowledge base (this can be expanded)
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten causes damage to the small intestine.",
    "The disease is triggered by the body's immune response to gluten, leading to inflammation and damage to the villi.",
    "Symptoms of celiac disease include diarrhea, weight loss, abdominal pain, and fatigue.",
    # Add more entries as needed
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
    
    # Format the prompt
    prompt = (
        f"User Query: {query}\n"
        f"Context: {context}\n\n"
    )
    
    # Generate the response
    response = generator(
        prompt,
        max_length=160,  # Adjust length for concise answers
        num_return_sequences=1,
        temperature=0.7,  # Moderate creativity for detailed answers
        repetition_penalty=1.2,  # Avoid repetitive phrases
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    # Extract and clean the generated text
    generated_text = response[0]["generated_text"].strip()
    
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
    
    # Ensure the response ends logically
    if not final_response.endswith("."):
        final_response += "."
    
    # Fallback for overly vague or failed responses
    if len(final_response) < 20:
        final_response = "I'm sorry, I couldn't find a suitable answer. Please try rephrasing your question."
    
    return final_response





import streamlit as st
from part1 import generate_response  # Import the response generation logic from Part 1

# Streamlit Appearance Setup
st.set_page_config(
    page_title="Baymax🩺",
    page_icon="⚕️",
    layout="wide",
)

# App Header with Branding
st.title("Baymax🩺")
st.subheader("Your Personal Healthcare Companion")
st.markdown("""
Welcome to **Baymax**, your personal healthcare companion! 🌟  
Ask me anything about health, wellness, or medical concerns.  
I use advanced AI and a curated knowledge base to provide accurate, helpful responses.
""")

# Sidebar Configuration with Image and Caption
st.sidebar.image(
    "https://i.pinimg.com/originals/3a/a8/51/3aa851a0f34d6703c7f0ac7ff6a41e8a.png",
    caption="Baymax: Your Personal Healthcare Companion",
    use_column_width=True
)

# Manage conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Main Chat Interface Section
st.subheader("🔍 Ask Your Question")
query = st.text_input("Type your question here:", help="E.g., What are the symptoms of celiac disease?")

# Button to trigger response generation
if st.button("Get Response 🚀"):
    if query.strip():
        with st.spinner("Thinking... 🤔"):
            try:
                # Generate the AI response based on the user's query
                response = generate_response(query)
                st.success("Here's what I found! 🧠")
                st.markdown(f"**{response}**")

                # Append the conversation history
                st.session_state.history.append(f"User: {query}")
                st.session_state.history.append(f"Baymax: {response}")
            except Exception as e:
                st.error(f"Something went wrong! 😕 Error: {e}")
    else:
        st.warning("Please enter a valid question! 📝")

# Display the conversation history
if st.session_state.history:
    st.write("### Conversation History")
    for message in st.session_state.history:
        st.write(message)

# Footer Section
st.markdown("""
---
**Pro Tip:** Use specific queries for the best results!  
**Example:** "What are the symptoms of diabetes?"
""")
st.markdown("Made with ❤️ for your health and well-being.")
