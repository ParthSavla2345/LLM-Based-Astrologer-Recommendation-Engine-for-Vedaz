
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys

# Mock astrologer data
astrologers = [
    {"id": 1, "name": "Astra", "tags": ["love", "relationships", "marriage"], "bio": "Specializes in love and relationship guidance."},
    {"id": 2, "name": "Cosmo", "tags": ["career", "finance"], "bio": "Expert in career and financial astrology."},
    {"id": 3, "name": "Luna", "tags": ["marriage", "family"], "bio": "Focuses on marriage and family dynamics."},
    {"id": 4, "name": "Stellar", "tags": ["love", "spiritual"], "bio": "Guides on love and spiritual growth."},
    {"id": 5, "name": "Nova", "tags": ["career", "personal growth"], "bio": "Helps with career and personal development."}
]

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode astrologer bios and user input
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
        # Use command-line argument if provided
        return " ".join(sys.argv[1:])
    else:
        # Otherwise, prompt for interactive input
        return input("Enter your query (e.g., 'I need career advice'): ")

# Example usage
if __name__ == "__main__":
    try:
        user_input = get_user_input()
        if not user_input.strip():
            print("Error: Please provide a non-empty input.")
            sys.exit(1)
        
        recommendations = recommend_astrologers(user_input, astrologers)
        
        print("\nTop 3 Recommended Astrologers:")
        for rec in recommendations:
            print(f"Name: {rec['name']}")
            print(f"Bio: {rec['bio']}")
            print(f"Tags: {', '.join(rec['tags'])}")
            print(f"Relevance Score: {rec['relevance_score']}")
            print("-" * 50)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Ensure all required libraries are installed and compatible with your Python version.")
        print("Try running: pip install sentence-transformers numpy scikit-learn pandas")

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys

# Mock astrologer data
astrologers = [
    {"id": 1, "name": "Astra", "tags": ["love", "relationships", "marriage"], "bio": "Specializes in love and relationship guidance."},
    {"id": 2, "name": "Cosmo", "tags": ["career", "finance"], "bio": "Expert in career and financial astrology."},
    {"id": 3, "name": "Luna", "tags": ["marriage", "family"], "bio": "Focuses on marriage and family dynamics."},
    {"id": 4, "name": "Stellar", "tags": ["love", "spiritual"], "bio": "Guides on love and spiritual growth."},
    {"id": 5, "name": "Nova", "tags": ["career", "personal growth"], "bio": "Helps with career and personal development."}
]

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode astrologer bios and user input
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
        # Use command-line argument if provided
        return " ".join(sys.argv[1:])
    else:
        # Otherwise, prompt for interactive input
        return input("Enter your query (e.g., 'I need career advice'): ")

# Example usage
if __name__ == "__main__":
    try:
        user_input = get_user_input()
        if not user_input.strip():
            print("Error: Please provide a non-empty input.")
            sys.exit(1)
        
        recommendations = recommend_astrologers(user_input, astrologers)
        
        print("\nTop 3 Recommended Astrologers:")
        for rec in recommendations:
            print(f"Name: {rec['name']}")
            print(f"Bio: {rec['bio']}")
            print(f"Tags: {', '.join(rec['tags'])}")
            print(f"Relevance Score: {rec['relevance_score']}")
            print("-" * 50)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Ensure all required libraries are installed and compatible with your Python version.")
        print("Try running: pip install sentence-transformers numpy scikit-learn pandas")
