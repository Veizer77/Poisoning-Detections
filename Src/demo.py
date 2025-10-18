"""
Quick Demo Script - Run this to test everything
"""
import pandas as pd
import sys
import os

# Simple path solution
sys.path.append('..')  # Go up one level
sys.path.append('.')   # Current directory
sys.path.append('./src')  # src directory

try:
    from src.integration import SecureRAGSystem
except ImportError:
    try:
        from engine import SecureRAGSystem
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure SecureRAGSystem is available")
        exit()

def main():
    print("RAG System Demo - Movie Recommendation with Poison Detection")
    print("=" * 60)
    
    # Load data
    try:
        movies = pd.read_csv('data/ml-32m/movies.csv')
        print(f"Loaded MovieLens dataset: {len(movies)} movies")
    except FileNotFoundError:
        print("Please download MovieLens dataset first")
        return
    
    # Initialize system
    system = SecureRAGSystem()
    system.initialize_system(movies)
    
    # Interactive demo
    while True:
        print("\n" + "="*40)
        print("Choose an option:")
        print("1. Get secure recommendations")
        print("2. Run attack-defense simulation") 
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            query = input("Enter your movie query: ").strip()
            if query:
                recommendations = system.secure_recommend(query, top_k=5)
                print(f"\n🔒 Secure recommendations for: '{query}'")
                for idx, movie in recommendations.iterrows():
                    print(f"   🎬 {movie['title']} | {movie['genres']} | Score: {movie['similarity_score']:.3f}")
        
        elif choice == '2':
            print("\nRunning attack-defense simulation...")
            results = system.simulate_attack_and_defense(movies, poison_ratio=0.1)
            print("\nSimulation Results:")
            print(results.to_string(index=False))
        
        elif choice == '3':
            print("Thank you for using the RAG System Demo!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()