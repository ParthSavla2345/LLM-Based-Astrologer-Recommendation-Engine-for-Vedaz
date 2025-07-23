
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import re

# Mock astrologer data
astrologers = [
    {"id": 1, "name": "Astra", "tags": ["love", "relationships", "marriage"], "bio": "Specializes in love and relationship guidance, helping you find harmony in your heart."},
    {"id": 2, "name": "Cosmo", "tags": ["career", "finance"], "bio": "Expert in career and financial astrology, guiding you to prosperity and success."},
    {"id": 3, "name": "Luna", "tags": ["marriage", "family"], "bio": "Focuses on marriage and family dynamics, fostering unity and understanding."},
    {"id": 4, "name": "Stellar", "tags": ["love", "spiritual"], "bio": "Guides on love and spiritual growth, connecting you to your higher self."},
    {"id": 5, "name": "Nova", "tags": ["career", "personal growth"], "bio": "Helps with career and personal development, unlocking your true potential."}
]

# Initialize SentenceTransformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Ensure 'sentence-transformers' is installed: pip install sentence-transformers")
    sys.exit(1)

# Function to preprocess user input
def preprocess_input(user_input):
    """Clean and enhance user input for better matching."""
    if not user_input or not user_input.strip():
        return None, "It looks like you didn't share your needs. Could you tell me what guidance you're seeking (e.g., love, career, or family)?"
    
    # Remove extra spaces and normalize
    user_input = re.sub(r'\s+', ' ', user_input.strip().lower())
    
    # Add context for vague inputs
    keywords = ['love', 'relationship', 'marriage', 'career', 'finance', 'family', 'spiritual', 'personal growth']
    if len(user_input.split()) < 3:  # Short input
        for keyword in keywords:
            if keyword in user_input:
                return user_input, None
        return user_input + " guidance", "Your query is quite brief! I've added some context to find the best astrologers for you."
    return user_input, None

# Function to encode texts
def encode_texts(texts):
    return model.encode(texts, convert_to_tensor=False)

# Function to recommend top 3 astrologers
def recommend_astrologers(user_input, astrologers, top_k=3):
    # Combine bio and tags for better representation
    astrologer_texts = [f"{astro['bio']} {' '.join(astro['tags'])}" for astro in astrologers]
    
    # Encode user input and astrologer texts
    user_embedding = encode_texts([user_input])[0]
    astrologer_embeddings = encode_texts(astrologer_texts)
    
    # Calculate cosine similarity
    similarities = cosine_similarity([user_embedding], astrologer_embeddings)[0]
    
    # Get top k astrologers
    top_indices = np.argsort(similarities)[::-1][:top_k]
    recommendations = [
        {
            "name": astrologers[i]["name"],
            "bio": astrologers[i]["bio"],
            "tags": astrologers[i]["tags"],
            "relevance_score": round(float(similarities[i]), 4)
        }
        for i in top_indices
    ]
    
    return recommendations

# Function to get user input
def get_user_input():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    else:
        print("🌟 Welcome to Vedaz's Astrologer Recommendation System! 🌟")
        print("I'm here to connect you with the perfect astrologer.")
        return input("Please share what guidance you seek (e.g., 'I need career advice'): ")

# Main execution
if __name__ == "__main__":
    try:
        # Get and preprocess user input
        user_input, feedback = preprocess_input(get_user_input())
        if user_input is None:
            print(feedback)
            sys.exit(1)
        if feedback:
            print(f"Note: {feedback}")
        
        # Get recommendations
        recommendations = recommend_astrologers(user_input, astrologers)
        
        # Print results in a conversational style
        print(f"\n✨ Based on your request for '{user_input}', here are the top 3 astrologers for you: ✨")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']} (Match Score: {rec['relevance_score']})")
            print(f"   Why they're a fit: {rec['bio']}")
            print(f"   Specialties: {', '.join(rec['tags'])}")
            print("-" * 60)
        
        # Offer to refine the query
        print("\nWould you like to refine your query for more tailored recommendations?")
        retry = input("Type 'yes' to try again or press Enter to exit: ").strip().lower()
        if retry == 'yes':
            print("\nLet's try again!")
            user_input, feedback = preprocess_input(get_user_input())
            if user_input is None:
                print(feedback)
                sys.exit(1)
            if feedback:
                print(f"Note: {feedback}")
            recommendations = recommend_astrologers(user_input, astrologers)
            print(f"\n✨ New recommendations for '{user_input}': ✨")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['name']} (Match Score: {rec['relevance_score']})")
                print(f"   Why they're a fit: {rec['bio']}")
                print(f"   Specialties: {', '.join(rec['tags'])}")
                print("-" * 60)
    
    except Exception as e:
        print(f"😔 Something went wrong: {str(e)}")
        print("Please ensure all libraries are installed: pip install sentence-transformers numpy scikit-learn pandas")
        print("If using Python 3.13, try Python 3.10 or 3.11 for better compatibility.")
