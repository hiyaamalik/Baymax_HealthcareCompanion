from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = pipeline("text-generation", model="distilgpt2")

# Define a simple medical knowledge base (this can be expanded)
knowledge_base = [
    "Celiac disease is an autoimmune disorder where the ingestion of gluten (a protein found in wheat, barley, and rye) causes damage to the small intestine.",
    "The disease is triggered by the body's immune response to gluten, which leads to inflammation and damage to the villi in the small intestine.",
    "Celiac disease is genetic, and individuals with a first-degree relative with the condition are at an increased risk.",
    "Diagnosis of celiac disease typically involves blood tests to check for specific antibodies (tTG-IgA and EMA), followed by a biopsy of the small intestine.",
    "Management of celiac disease primarily involves a strict, lifelong gluten-free diet.",
    "Some people may also experience symptoms outside of the digestive system, such as skin rashes (dermatitis herpetiformis), joint pain, and headaches.",
    "Regular physical activity is encouraged for individuals with celiac disease, but it is important to listen to the body and avoid overexertion.",
    "Therapy or counseling may be beneficial for individuals experiencing emotional distress related to their diagnosis.",
    "With proper care, individuals with celiac disease can lead healthy, fulfilling lives, free from the debilitating symptoms of gluten exposure."
]

# Function to encode text into embeddings (using DistilGPT-2 embeddings here for simplicity)
def encode_text(texts):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Add a padding token to the tokenizer if it doesn't already exist
    tokenizer.pad_token = tokenizer.eos_token
    
    # Encode texts with padding
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Get word embeddings from the model
    outputs = model.transformer.wte(inputs.input_ids)  # Get word embeddings
    return outputs.mean(dim=1).detach().numpy()  # Use mean of word embeddings for simplicity

# Encode the knowledge base into vectors
encoded_kb = encode_text(knowledge_base)

# Create a FAISS index for fast retrieval
index = faiss.IndexFlatL2(encoded_kb.shape[1])  # Create FAISS index for vector search
index.add(np.array(encoded_kb))  # Add encoded knowledge base to the FAISS index

# Function to retrieve relevant information based on user query
def retrieve_info(query):
    query_vec = encode_text([query])
    
    # Perform a search in the FAISS index
    D, I = index.search(query_vec, k=3)  # Retrieve top 3 most relevant pieces of knowledge
    
    # Collect the relevant documents
    relevant_info = [knowledge_base[i] for i in I[0]]
    return " ".join(relevant_info)

# Function to generate a response based on retrieved information
def generate_response(query):
    # Retrieve relevant information from the knowledge base
    context = retrieve_info(query)
    
    # Combine the context with the user query to generate a context-aware response
    prompt = f"User Query: {query}\nContext: {context}\nAnswer:"
    
    # Generate response using the DistilGPT-2 model
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"].strip()

# Simple chatbot loop to interact with the user
def chatbot():
    print("Hello! I'm your Medical Advisor. Ask me anything about health.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye! Stay healthy!")
            break
        
        # Get the context-aware response
        response = generate_response(user_input)
        
        print("Bot:", response)

# Start the chatbot
chatbot()
