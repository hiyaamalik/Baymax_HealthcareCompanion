import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Ensure the device is set correctly for PyTorch (if you're using a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the DistilGPT-2 model (a lighter version of GPT-2)
generator = pipeline("text-generation", model="distilgpt2", device=device)

# Define a simple medical knowledge base (this can be expanded)
knowledge_base = [
    "Celiac disease is an autoimmune disorder that causes the body’s immune system to mistakenly attack its own tissues when gluten is ingested. Gluten, a protein found in wheat, barley, and rye, triggers an immune response that leads to inflammation and damage to the villi of the small intestine. This damage impairs the absorption of essential nutrients, leading to malnutrition and a variety of symptoms such as diarrhea, weight loss, abdominal pain, fatigue, and even skin rashes. If left untreated, celiac disease can cause long-term complications such as osteoporosis, infertility, and even neurological disorders. The only effective treatment for celiac disease is a lifelong gluten-free diet, which allows the intestine to heal and prevents further damage. Individuals with celiac disease must carefully avoid all sources of gluten, including hidden gluten in processed foods, cosmetics, and medicines. Regular checkups and blood tests are essential to monitor the individual's health and ensure the gluten-free diet is being strictly followed.",
    
    "Celiac disease symptoms can vary widely among individuals and can include both gastrointestinal and non-gastrointestinal symptoms. Common gastrointestinal symptoms include diarrhea, bloating, abdominal cramps, and weight loss. However, many people may also experience non-gastrointestinal symptoms such as fatigue, joint pain, headaches, depression, skin rashes, and even infertility. Celiac disease can often be misdiagnosed due to the variety of symptoms, and it may be confused with other conditions like irritable bowel syndrome or lactose intolerance. Diagnosis typically involves blood tests looking for antibodies specific to celiac disease and a biopsy of the small intestine to assess damage to the villi. Left untreated, celiac disease can lead to serious health complications such as osteoporosis, liver disease, and an increased risk of certain types of cancer. It is crucial for individuals with suspected celiac disease to seek early diagnosis and follow a strict gluten-free diet to prevent long-term health issues.",
    
    "A gluten-free diet is essential for managing celiac disease. This diet eliminates all foods containing wheat, barley, rye, and their derivatives, including many processed foods, sauces, and beverages. The primary goal of a gluten-free diet is to allow the small intestine to heal and to prevent further damage from the immune response triggered by gluten. Foods that are naturally gluten-free, such as fruits, vegetables, lean meats, and most dairy products, can form the foundation of the diet. Gluten-free grains like rice, corn, quinoa, and oats (that are not contaminated with gluten) can also be consumed. However, cross-contamination is a significant concern, and individuals with celiac disease must be diligent in avoiding even trace amounts of gluten. For those with celiac disease, careful label reading and food preparation are necessary to ensure safety. Additionally, individuals should be aware of hidden gluten in medications and supplements, as these can also cause adverse reactions.",
    
    "Gluten is a protein found in wheat, barley, and rye that gives dough its elasticity and helps it rise and maintain its shape. For individuals with celiac disease, gluten triggers an immune response that damages the small intestine, particularly the villi, which are responsible for nutrient absorption. In people with celiac disease, consuming gluten leads to inflammation and destruction of these villi, leading to malabsorption of nutrients. This can cause a range of symptoms from gastrointestinal issues like diarrhea, bloating, and cramps, to more serious complications such as anemia, osteoporosis, and even infertility. Gluten is found in a wide range of foods and beverages, including bread, pasta, cereals, and beer. It is also used in processed foods as a stabilizing agent and thickener. For individuals with celiac disease, it is crucial to avoid all forms of gluten and any products that may be contaminated with gluten, including cosmetics and medications that may contain gluten as a binding agent.",
    
    "Baymax is a virtual healthcare companion powered by artificial intelligence. Designed to provide users with information about health-related queries, Baymax can track wellness metrics such as heart rate, sleep patterns, and physical activity. As an AI-powered assistant, Baymax can also provide insights on various health topics, such as nutrition, exercise, and mental well-being. Its goal is to enhance healthcare accessibility by providing personalized and real-time health recommendations. Baymax can be integrated with wearable devices to monitor various health parameters and provide alerts when there are concerns. It also offers helpful advice on lifestyle modifications, such as improving diet, exercise, and sleep, to promote overall well-being. As technology continues to evolve, Baymax can play a significant role in preventive healthcare by empowering individuals to make informed health decisions and proactively manage their wellness.",
    
    "Diabetes is a chronic medical condition that occurs when the body cannot effectively regulate blood sugar levels. Type 1 diabetes is an autoimmune disorder where the immune system attacks and destroys the insulin-producing cells in the pancreas, requiring individuals to take insulin for life. Type 2 diabetes, the most common form, occurs when the body becomes resistant to insulin or the pancreas does not produce enough insulin. Type 2 diabetes is often linked to obesity, lack of physical activity, and poor diet, and it can be managed with lifestyle changes, medication, and insulin therapy in more severe cases. Untreated diabetes can lead to serious complications, including heart disease, kidney damage, nerve damage, and blindness. Monitoring blood sugar levels, following a balanced diet, exercising regularly, and managing stress are crucial for managing diabetes and preventing complications. Early detection and intervention are key to improving outcomes and quality of life for individuals with diabetes.",
    
    "Hypertension, or high blood pressure, is a condition in which the force of blood against the walls of the arteries is consistently too high. This condition can be caused by factors such as poor diet, lack of physical activity, obesity, excessive alcohol consumption, and stress. Hypertension is often referred to as the 'silent killer' because it may not show symptoms until significant damage has occurred to the heart, kidneys, or other organs. Left untreated, hypertension can lead to serious health complications, including stroke, heart disease, kidney failure, and vision loss. Blood pressure is measured in millimeters of mercury (mmHg), and a normal reading is usually around 120/80 mmHg. Lifestyle changes, such as reducing salt intake, maintaining a healthy weight, exercising regularly, and limiting alcohol consumption, can help lower blood pressure. In some cases, medication may be required to control hypertension and prevent complications.",
    
    "Asthma is a chronic respiratory condition that affects the airways, causing them to become inflamed and narrow, making it difficult to breathe. Common symptoms include shortness of breath, wheezing, coughing, and chest tightness, which can range from mild to severe. Asthma attacks are triggered by various factors, including allergens, air pollution, respiratory infections, and physical activity. There are two main types of asthma: allergic and non-allergic asthma. While allergic asthma is triggered by environmental allergens like pollen, dust mites, and pet dander, non-allergic asthma can be triggered by irritants such as smoke, strong odors, and weather changes. Asthma is managed through medications, including inhaled bronchodilators and corticosteroids, which help open the airways and reduce inflammation. It is also essential to identify and avoid triggers to prevent asthma attacks. With proper management, most people with asthma can lead normal, active lives.",
    
    "Mental health encompasses emotional, psychological, and social well-being. It affects how individuals think, feel, and behave, influencing how they handle stress, relate to others, and make decisions. Mental health disorders are common and can affect anyone at any stage of life. Some of the most prevalent mental health disorders include anxiety disorders, depression, bipolar disorder, and schizophrenia. Mental health is influenced by a variety of factors, including genetics, life experiences, and family history. Mental health issues can manifest in various ways, including changes in mood, behavior, and cognitive function. Early intervention and treatment are critical to managing mental health conditions and improving quality of life. Therapy, medication, lifestyle changes, and social support play a key role in managing mental health disorders. Destigmatizing mental health and encouraging open dialogue is essential to improving access to care and promoting mental wellness.",
    
    "Chronic stress is a prolonged and constant feeling of stress that can have detrimental effects on both mental and physical health. While short-term stress can be beneficial in certain situations, chronic stress overwhelms the body's ability to cope and can contribute to the development of various health conditions. Physically, chronic stress can lead to high blood pressure, heart disease, digestive issues, weakened immune function, and even sleep disturbances. Psychologically, chronic stress can exacerbate mental health issues, including anxiety and depression. Effective stress management techniques, such as mindfulness, relaxation exercises, deep breathing, regular physical activity, and maintaining a healthy work-life balance, can help reduce the impact of chronic stress. In some cases, professional counseling or therapy may be necessary to help individuals cope with stress effectively and prevent long-term health complications.",
]




# Using a pre-trained Sentence-BERT model to generate better embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_text(texts):
    embeddings = sentence_model.encode(texts, convert_to_numpy=True)
    return embeddings

# Rebuild the FAISS index
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
    
    # Create a refined prompt
    prompt = f"""
    User's Query: {query}

    Context:
    {context}

    Based on the above context, answer the user's query as accurately and concisely as possible, focusing on the most relevant information. 
    """
    
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

    # Filter sentences for relevance and coherence
    sentences = generated_text.split(". ")
    filtered_sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence) > 20
    ]

    final_response = ". ".join(filtered_sentences)
    
    if not final_response.endswith("."):
        final_response += "."
    
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
