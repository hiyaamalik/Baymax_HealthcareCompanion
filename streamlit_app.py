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
    "Gluten is a protein found in wheat, barley, and rye. Individuals with celiac disease must completely avoid foods containing these grains.",
    "Baymax is a versatile AI-powered personal healthcare companion designed to assist with medical queries, wellness tracking, and health insights.",
    "Maintaining hydration is essential for overall health, as water supports digestion, nutrient absorption, and cellular function.",
    "Diabetes is a chronic condition characterized by high blood sugar levels due to insufficient insulin production or the body's inability to use insulin effectively.",
    "Regular exercise can significantly improve cardiovascular health, mental well-being, and weight management.",
    "Hypertension, or high blood pressure, is a condition where the force of blood against the artery walls is consistently too high, increasing the risk of heart disease.",
    "Asthma is a chronic respiratory condition that causes inflammation and narrowing of the airways, leading to difficulty breathing, wheezing, and coughing.",
    "Mental health is a crucial aspect of overall wellness, encompassing emotional, psychological, and social well-being.",
    "Chronic stress can negatively impact both mental and physical health, contributing to conditions such as anxiety, depression, and hypertension.",
    "A balanced diet rich in fruits, vegetables, lean proteins, and whole grains is essential for maintaining optimal health.",
    "Sleep is vital for physical recovery and mental clarity. Adults should aim for 7-9 hours of sleep per night.",
    "Anemia is a condition in which the body lacks enough healthy red blood cells to carry adequate oxygen to tissues, often causing fatigue and weakness.",
    "Allergies occur when the immune system reacts to a foreign substance such as pollen, pet dander, or certain foods.",
    "Vitamin D is essential for bone health and immune function. Sun exposure and fortified foods are common sources.",
    "Migraine is a neurological condition characterized by intense, throbbing headaches often accompanied by nausea, sensitivity to light, and sound.",
    "Arthritis refers to inflammation of the joints, causing pain, stiffness, and decreased mobility.",
    "Regular health checkups and screenings are vital for early detection and management of potential health issues.",
    "Cancer is the uncontrolled growth of abnormal cells in the body, which can invade nearby tissues and spread to other parts of the body.",
    "Cardiovascular diseases, including heart attacks and strokes, are the leading cause of death globally, often preventable through lifestyle changes.",
    "Smoking is one of the leading causes of preventable diseases, including lung cancer, heart disease, and chronic obstructive pulmonary disease (COPD).",
    "Obesity increases the risk of various health conditions, including type 2 diabetes, hypertension, and sleep apnea.",
    "The immune system protects the body against infections and diseases by identifying and neutralizing harmful pathogens.",
    "Vaccines are critical for preventing diseases such as measles, polio, and influenza by training the immune system to recognize pathogens.",
    "Chronic kidney disease is a condition in which the kidneys lose their ability to filter waste from the blood effectively.",
    "Osteoporosis is a condition characterized by weak and brittle bones, increasing the risk of fractures, especially in older adults.",
    "Alzheimer's disease is a progressive neurological disorder that leads to memory loss, cognitive decline, and changes in behavior.",
    "Depression is a common mental health disorder that negatively affects mood, thoughts, and physical well-being.",
    "The digestive system breaks down food into nutrients the body can absorb, involving organs such as the stomach, intestines, and liver.",
    "Skin health is influenced by factors such as diet, hydration, and protection from UV rays. Sunscreen is essential for preventing skin damage.",
    "The respiratory system supplies oxygen to the body and removes carbon dioxide, relying on organs like the lungs and trachea.",
    "The liver is a vital organ responsible for detoxifying the blood, producing bile, and regulating metabolism.",
    "The endocrine system regulates hormones that control growth, metabolism, and reproduction, with the thyroid gland playing a significant role.",
    "The brain is the control center of the body, managing functions like memory, emotions, and motor coordination.",
    "The human microbiome, consisting of trillions of microorganisms, plays a crucial role in digestion, immunity, and overall health.",
    "Anxiety disorders are among the most common mental health conditions, characterized by excessive worry and fear.",
    "Exercise releases endorphins, which act as natural painkillers and mood elevators.",
    "Posture affects musculoskeletal health, and poor posture can lead to back pain and other physical issues.",
    "Dehydration can cause symptoms such as headache, dizziness, and fatigue, highlighting the importance of drinking enough water daily.",
    "Nutritional deficiencies, such as a lack of iron or vitamin B12, can lead to specific health problems like anemia.",
    "Good oral hygiene, including brushing and flossing, helps prevent dental issues such as cavities and gum disease.",
    "Prolonged exposure to loud noise can lead to hearing loss, emphasizing the importance of protecting your ears in noisy environments.",
    "The heart beats approximately 100,000 times per day, pumping oxygenated blood throughout the body.",
    "Good mental health practices, like mindfulness and relaxation techniques, can help manage stress and improve overall well-being."
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
        max_length=130,  # Adjust length for concise answers
        num_return_sequences=1,
        temperature=0.7,  # Moderate creativity for detailed answers
        repetition_penalty=1.2,  # Avoid repetitive phrases
        truncation=True,
        pad_token_id=generator.tokenizer.convert_tokens_to_ids(".")  # Full stop as pad token
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
st.markdown("Made with ❤️ for your health and well-being.")
