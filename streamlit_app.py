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
    
    # Format the prompt
    prompt = (
        f"User Query: {query}\n"
        f"Context: {context}\n\n"
        
    )
    
    # Generate the response
    response = generator(
        prompt,
        max_length=150,  # Allow enough length for detailed responses
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

    # Split the generated text into sentences
    sentences = generated_text.split(". ")
    
    # Track word count and detect first paragraph break
    filtered_sentences = []
    word_count = 0
    paragraph_found = False

    for sentence in sentences:
        # Ignore sentences or paragraphs that do not align with the question context
        if "lactic acid" in sentence.lower():  # Exclude irrelevant parts like the second paragraph
            continue
        filtered_sentences.append(sentence.strip())
        word_count += len(sentence.split())

        # Check if a paragraph break occurs (empty line between paragraphs)
        if sentence == "":
            paragraph_found = True
        
        # Stop generation after first paragraph and within the word range
        if word_count >= 100 and word_count <= 150:
            break
        
        if paragraph_found:  # Stop after the first paragraph
            break

    # Join filtered sentences to form a coherent response
    final_response = ". ".join(filtered_sentences)
    
    # Ensure the response ends logically
    if not final_response.endswith("."):
        final_response += "."
    
    # Fallback for overly vague or failed responses
    if len(final_response) < 20:
        final_response = "I'm sorry, I couldn't find a suitable answer. Please try rephrasing your question."
    
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
Welcome to the **Gluten Free Lifestyle Assistant**! 🌟  
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
