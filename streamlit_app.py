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
    "Celiac disease is an autoimmune disorder where the ingestion of gluten (a protein found in wheat, barley, and rye) causes damage to the small intestine.",
    "The disease is triggered by the body's immune response to gluten, which leads to inflammation and damage to the villi in the small intestine.",
    "The villi are tiny hair-like structures that line the small intestine and are essential for nutrient absorption.",
    "When the villi are damaged, the body is unable to properly absorb nutrients, leading to malnutrition, vitamin deficiencies, and a variety of other health problems.",
    "Celiac disease is genetic, and individuals with a first-degree relative with the condition are at an increased risk.",
    "It affects both children and adults, although it can sometimes go undiagnosed for years.",
    "Diagnosis of celiac disease typically involves blood tests to check for specific antibodies (tTG-IgA and EMA), followed by a biopsy of the small intestine to assess the extent of damage to the villi.",
    "The biopsy is the gold standard for confirming celiac disease, but the blood tests are highly indicative and often used as the first step in diagnosis.",
    "The main symptom of celiac disease is diarrhea, but other common symptoms include abdominal pain, bloating, gas, weight loss, and fatigue.",
    "In children, celiac disease may cause delayed growth, irritability, and behavioral issues.",
    "Some people may also experience symptoms outside of the digestive system, such as skin rashes (dermatitis herpetiformis), joint pain, and headaches.",
    "In adults, celiac disease may present as anemia, osteopenia (low bone density), or infertility.",
    "In addition to these common symptoms, some individuals may have more subtle symptoms, or they may be asymptomatic, making diagnosis more difficult.",
    "If left untreated, celiac disease can lead to severe complications such as osteoporosis, infertility, neurological disorders, and an increased risk of certain types of cancer, including lymphoma and small bowel cancer.",
    "Management of celiac disease primarily involves a strict, lifelong gluten-free diet.",
    "This is the only known effective treatment for the disease and helps to heal the intestine, prevent further damage, and alleviate symptoms.",
    "A gluten-free diet means avoiding all foods that contain wheat, barley, rye, and any ingredients derived from these grains.",
    "This includes bread, pasta, cereals, baked goods, and processed foods that contain gluten as an additive or thickener.",
    "Cross-contamination is also a concern, so it is important to ensure that foods are prepared in a gluten-free environment and that utensils, surfaces, and appliances are thoroughly cleaned.",
    "In addition to a gluten-free diet, individuals with celiac disease may need to take supplements to address nutritional deficiencies, such as iron, calcium, vitamin D, and folate.",
    "People with celiac disease may also need to be monitored by a healthcare provider for other conditions associated with the disease, including osteoporosis, thyroid disease, and liver issues.",
    "The management of celiac disease involves regular check-ups and monitoring for complications, as well as lifestyle changes to minimize the risk of gluten exposure.",
    "There is currently no cure for celiac disease, but following a gluten-free diet can allow most people with the condition to live a healthy life and prevent complications.",
    "For some individuals, the disease may be diagnosed later in life, and they may face challenges in adjusting to a gluten-free lifestyle.",
    "Celiac disease can also have a significant impact on mental health, as individuals may feel isolated due to dietary restrictions, or they may experience anxiety around food choices.",
    "Social support from family, friends, and celiac disease support groups can be crucial in managing the emotional aspects of the disease.",
    "Additionally, education about celiac disease, labeling laws, and finding gluten-free alternatives are essential for individuals to thrive.",
    "There are several lifestyle strategies for managing celiac disease, including meal planning, cooking at home, and reading food labels carefully.",
    "Many people with celiac disease benefit from working with a dietitian who specializes in gluten-free diets to ensure they are getting the nutrients they need while avoiding gluten.",
    "It's important for individuals with celiac disease to be aware of the risk of hidden gluten in foods, medications, and even cosmetics.",
    "Taking care to avoid any form of gluten exposure is key to managing the condition.",
    "Some individuals may find it difficult to travel or dine out, but with careful planning and communication with restaurant staff, it is often possible to enjoy meals while avoiding gluten.",
    "In terms of recipes, there are many gluten-free options for breakfast, lunch, and dinner.",
    "For breakfast, gluten-free oatmeal, egg-based dishes, and smoothies are good choices.",
    "For lunch and dinner, gluten-free pasta, rice dishes, salads with grilled meats or fish, and soups made with gluten-free broths are all excellent options.",
    "Gluten-free breads, cakes, and cookies can be made using alternative flours such as rice flour, almond flour, or coconut flour.",
    "There are also pre-packaged gluten-free products available, but it is important to read labels to ensure they are truly gluten-free and not cross-contaminated.",
    "In addition to a gluten-free diet, many people with celiac disease find that eating smaller, more frequent meals helps with digestion and reduces symptoms.",
    "Hydration is also important, as diarrhea and malabsorption can lead to dehydration.",
    "Regular physical activity is encouraged for individuals with celiac disease, but it is important to listen to the body and avoid overexertion, particularly when symptoms are active.",
    "Celiac disease can impact mental health, and people with the condition may experience feelings of frustration, isolation, or depression due to the challenges of living with dietary restrictions.",
    "Therapy or counseling may be beneficial for individuals experiencing emotional distress related to their diagnosis.",
    "It's important for people with celiac disease to maintain regular communication with their healthcare providers, follow up with appropriate screenings, and adhere to a gluten-free diet to manage their health and quality of life.",
    "In summary, celiac disease is a lifelong condition that requires careful management, including a strict gluten-free diet, regular monitoring for complications, and support for both physical and mental health.",
    "With proper care, individuals with celiac disease can lead healthy, fulfilling lives, free from the debilitating symptoms of gluten exposure."
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
        max_length=175,  # Allow for enough length to handle the desired range
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

    # Split into sentences and handle paragraph breaks
    sentences = generated_text.split(". ")
    filtered_sentences = []
    word_count = 0
    paragraph_found = False
    for sentence in sentences:
        # Add sentences to the response until the first paragraph break
        filtered_sentences.append(sentence.strip())
        word_count += len(sentence.split())

        # Detect paragraph breaks (two consecutive line breaks)
        if sentence == "":
            paragraph_found = True

        if word_count >= 150 and word_count <= 160:
            break
        
        if paragraph_found:  # If a paragraph break is found, stop the generation
            break

    # Join filtered sentences into a coherent response
    final_response = ". ".join(filtered_sentences)

    # Ensure the response ends logically and has no trailing spaces
    final_response = final_response.strip()
    
    # Fallback for overly vague or failed responses
    if len(final_response.split()) < 20:
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
