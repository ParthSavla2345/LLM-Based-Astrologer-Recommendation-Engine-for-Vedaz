
from transformers import pipeline
import sys

# Initialize a text generation pipeline with distilgpt2
generator = pipeline("text-generation", model="distilgpt2")

# Prompt for AI astrologer
def ai_astrologer(input_text):
    prompt = f"""
You are an AI astrologer with a mystical and positive tone. Based on the following palm reading or horoscope summary, provide a concise life insight or advice in 2-3 sentences. Keep the advice uplifting, spiritual, and focused on guiding the user.

Palm Reading: {input_text}

Advice:
"""
    # Generate text with tuned parameters for better coherence
    response = generator(
        prompt,
        max_new_tokens=50,
        num_return_sequences=1,
        truncation=True,
        temperature=0.7,  # Lower temperature for focused output
        top_p=0.9,        # Nucleus sampling for coherent text
        do_sample=True    # Enable sampling for creative responses
    )[0]["generated_text"]
    
    # Extract only the advice part after "Advice:"
    try:
        advice = response.split("Advice:")[1].strip()
    except IndexError:
        advice = response.strip()  # Fallback if "Advice:" is not found
    return advice

# Function to get input from user (interactive or command-line)
def get_user_input():
    if len(sys.argv) > 1:
        # If command-line argument is provided
        return " ".join(sys.argv[1:])
    else:
        # Interactive input
        return input("Enter your palm reading or horoscope summary: ")

# Example usage
if __name__ == "__main__":
    # Get input from user
    palm_reading = get_user_input()
    
    # Generate advice
    advice = ai_astrologer(palm_reading)
    print("AI Astrologer Advice:")
    print(advice)