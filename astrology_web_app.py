
from flask import Flask, request, render_template
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Expanded mock astrologer data
astrologers = [
    {"id": 1, "name": "Astra", "tags": ["love", "relationships", "marriage"], "bio": "Specializes in love and relationship guidance, helping you find harmony in your heart.", "experience_years": 12, "rating": 4.8, "specialty_description": "Astra uses cosmic insights to heal emotional wounds and foster deep connections."},
    {"id": 2, "name": "Cosmo", "tags": ["career", "finance"], "bio": "Expert in career and financial astrology, guiding you to prosperity and success.", "experience_years": 15, "rating": 4.9, "specialty_description": "Cosmo aligns your career path with the stars for financial abundance."},
    {"id": 3, "name": "Luna", "tags": ["marriage", "family"], "bio": "Focuses on marriage and family dynamics, fostering unity and understanding.", "experience_years": 10, "rating": 4.7, "specialty_description": "Luna strengthens family bonds through celestial wisdom."},
    {"id": 4, "name": "Stellar", "tags": ["love", "spiritual"], "bio": "Guides on love and spiritual growth, connecting you to your higher self.", "experience_years": 8, "rating": 4.6, "specialty_description": "Stellar channels spiritual energy to enhance love and personal growth."},
    {"id": 5, "name": "Nova", "tags": ["career", "personal growth"], "bio": "Helps with career and personal development, unlocking your true potential.", "experience_years": 9, "rating": 4.8, "specialty_description": "Nova empowers you to achieve professional and personal milestones."},
    {"id": 6, "name": "Sol", "tags": ["health", "wellness"], "bio": "Specializes in health and wellness, promoting balance and vitality.", "experience_years": 11, "rating": 4.5, "specialty_description": "Sol uses astrology to guide you toward physical and mental well-being."},
    {"id": 7, "name": "Celeste", "tags": ["spiritual", "destiny"], "bio": "Explores your spiritual destiny and life purpose.", "experience_years": 14, "rating": 4.9, "specialty_description": "Celeste reveals your soul’s path through cosmic alignment."},
    {"id": 8, "name": "Orion", "tags": ["finance", "investments"], "bio": "Guides financial decisions and investments with stellar foresight.", "experience_years": 13, "rating": 4.7, "specialty_description": "Orion helps you navigate wealth creation with astrological precision."},
    {"id": 9, "name": "Vega", "tags": ["love", "self-discovery"], "bio": "Focuses on love and self-discovery, helping you find inner peace.", "experience_years": 7, "rating": 4.6, "specialty_description": "Vega fosters self-love and romantic harmony through the stars."},
    {"id": 10, "name": "Aurora", "tags": ["family", "personal growth"], "bio": "Supports family harmony and personal growth journeys.", "experience_years": 10, "rating": 4.8, "specialty_description": "Aurora nurtures family connections and personal evolution."}
]

# Initialize models
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_generator = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    print(f"Failed to load models: {e}")
    print("Ensure libraries are installed: pip install flask sentence-transformers transformers")
    sys.exit(1)

# Preprocess user input
def preprocess_input(user_input):
    if not user_input or not user_input.strip():
        return None, "Please share what guidance you seek (e.g., 'I need career advice' or 'Your heart line is deep')."
    user_input = re.sub(r'\s+', ' ', user_input.strip().lower())
    keywords = ['love', 'relationship', 'marriage', 'career', 'finance', 'family', 'spiritual', 'personal growth', 'health', 'wellness', 'destiny', 'investments', 'self-discovery']
    if len(user_input.split()) < 3:
        for keyword in keywords:
            if keyword in user_input:
                return user_input, None
        return user_input + " guidance", "Your query is brief! I've added context to find the best matches."
    return user_input, None

# Recommendation engine
def recommend_astrologers(user_input, astrologers, top_k=3):
    astrologer_texts = [f"{astro['bio']} {astro['specialty_description']} {' '.join(astro['tags'])}" for astro in astrologers]
    try:
        user_embedding = sentence_model.encode([user_input])[0]
        astrologer_embeddings = sentence_model.encode(astrologer_texts)
        similarities = cosine_similarity([user_embedding], astrologer_embeddings)[0]
        weighted_scores = [similarities[i] * 0.8 + (astrologers[i]["rating"] / 5) * 0.2 for i in range(len(similarities))]
        top_indices = np.argsort(weighted_scores)[::-1][:top_k]
        return [
            {
                "name": astrologers[i]["name"],
                "bio": astrologers[i]["bio"],
                "tags": astrologers[i]["tags"],
                "rating": astrologers[i]["rating"],
                "experience_years": astrologers[i]["experience_years"],
                "relevance_score": round(float(weighted_scores[i]), 4)
            }
            for i in top_indices
        ]
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return []

# AI astrologer with top astrologer integration
def ai_astrologer(input_text, top_astrologer):
    prompt = f"""
You are an AI astrologer channeling the expertise of {top_astrologer['name']}, who specializes in {', '.join(top_astrologer['tags'])}. 
Based on the following palm reading or horoscope summary, provide a concise, mystical, and positive life insight or advice in 2-3 sentences. 
Incorporate {top_astrologer['name']}'s expertise to make the advice actionable and relevant.
Input: {input_text}
Advice:
"""
    try:
        logger.debug(f"Generating advice with prompt: {prompt[:100]}...")
        response = text_generator(
            prompt,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )[0]["generated_text"]
        logger.debug(f"Raw response: {response}")
        advice = response.split("Advice:")[1].strip() if "Advice:" in response else response.strip()
        return advice
    except Exception as e:
        logger.error(f"AI astrologer error: {e}")
        return f"The stars align with {top_astrologer['name']}'s guidance in {', '.join(top_astrologer['tags'])}; trust their expertise to lead you forward."

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    rec_result = None
    advice_result = None
    rec_feedback = None
    advice_feedback = None

    if request.method == "POST":
        # Recommendation form
        if "rec_query" in request.form:
            user_input = request.form["rec_query"]
            processed_input, feedback = preprocess_input(user_input)
            if processed_input is None:
                rec_feedback = feedback
            else:
                rec_result = recommend_astrologers(processed_input, astrologers)
                rec_feedback = feedback

        # AI astrologer form
        if "advice_query" in request.form:
            user_input = request.form["advice_query"]
            processed_input, feedback = preprocess_input(user_input)
            if processed_input is None:
                advice_feedback = feedback
            else:
                try:
                    top_astrologer = recommend_astrologers(processed_input, astrologers, top_k=1)[0]
                    advice_result = ai_astrologer(processed_input, top_astrologer)
                    advice_feedback = feedback
                except Exception as e:
                    logger.error(f"Advice processing error: {e}")
                    advice_feedback = "Unable to generate advice. Please try again with a different input."

    return render_template(
        "index.html",
        rec_result=rec_result,
        advice_result=advice_result,
        rec_feedback=rec_feedback,
        advice_feedback=advice_feedback
    )

if __name__ == "__main__":
    app.run(debug=True)
