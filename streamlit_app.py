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
    
    # Join sentences until a full stop is encountered
    final_response = ""
    for sentence in filtered_sentences:
        final_response += sentence + ". "
        # Stop when a full stop is encountered
        if sentence.endswith("."):
            break
    
    # Remove any extra spaces and ensure the response ends logically
    final_response = final_response.strip()
    
    # Fallback for overly vague or failed responses
    if len(final_response) < 20:
        final_response = "I'm sorry, I couldn't find a suitable answer. Please try rephrasing your question."
    
    return final_response
