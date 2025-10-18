import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import torch
import random


class RAGRecommender:
    """
    Simple RAG System for Movie Recommendations
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.knowledge_base = None
        self.embeddings = None
        self.is_built = False
        
    def build_knowledge_base(self, movies_df, save_path='knowledge_base.pkl'):
        """
        Build vector database from movie data
        """
        print("Building RAG Knowledge Base...")
        
        # Create descriptive texts for each movie
        descriptions = []
        for _, row in movies_df.iterrows():
            desc = f"Movie: {row['title']}. Genres: {row['genres']}."
            descriptions.append(desc)
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.model.encode(descriptions, show_progress_bar=True)
        
        # Build knowledge base
        self.knowledge_base = movies_df.copy()
        self.knowledge_base['description'] = descriptions
        self.knowledge_base['embedding'] = list(self.embeddings)
        
        self.is_built = True
        print(f"Knowledge base built with {len(self.knowledge_base)} items")
        
        # Save for later use
        with open(save_path, 'wb') as f:
            pickle.dump({
                'knowledge_base': self.knowledge_base,
                'embeddings': self.embeddings
            }, f)
        print(f"Knowledge base saved to {save_path}")
        
        return self.knowledge_base
    
    def load_knowledge_base(self, load_path='knowledge_base.pkl'):
        """
        Load pre-built knowledge base
        """
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            self.knowledge_base = data['knowledge_base']
            self.embeddings = data['embeddings']
            self.is_built = True
            print(f"Knowledge base loaded from {load_path}")
            return True
        else:
            print("No saved knowledge base found. Please build first.")
            return False
    
    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k most relevant movies based on query
        """
        if not self.is_built:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = self.knowledge_base.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results
    
    def recommend(self, query, top_k=10, rerank=True):
        """
        Full recommendation pipeline with optional reranking simulation
        """
        # Step 1: Retrieval
        candidates = self.retrieve(query, top_k=top_k * 2)  # Get more for reranking
        
        if rerank:
            # Step 2: Simple reranking simulation (replace with actual LLM in future)
            candidates = self._simulate_llm_reranking(candidates, query, top_k)
        
        return candidates.head(top_k)
    
    def _simulate_llm_reranking(self, candidates, query, top_k):
        """
        Simulate LLM reranking with simple heuristics
        In real scenario, this would call an actual LLM API
        """
        # Simple heuristic: boost scores for exact genre matches
        query_lower = query.lower()
        
        for idx, row in candidates.iterrows():
            boost = 1.0
            
            # Boost if query contains genre words
            genres = str(row['genres']).lower()
            if any(genre in query_lower for genre in ['action', 'comedy', 'drama', 'horror', 'romance']):
                if any(genre in genres for genre in ['action', 'comedy', 'drama', 'horror', 'romance']):
                    boost *= 1.2
            
            # Boost for exact title matches
            title = str(row['title']).lower()
            if any(word in title for word in query_lower.split()):
                boost *= 1.1
            
            candidates.loc[idx, 'similarity_score'] *= boost
        
        # Rerank based on boosted scores
        candidates = candidates.sort_values('similarity_score', ascending=False)
        return candidates.head(top_k)
    
    def get_movie_by_id(self, movie_id):
        """
        Get movie details by ID
        """
        if not self.is_built:
            return None
        return self.knowledge_base[self.knowledge_base['movieId'] == movie_id]

# Test the RAG system
if __name__ == "__main__":
    # Load data
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # Initialize and build RAG system
    rag = RAGRecommender()
    
    # Build or load knowledge base
    if not rag.load_knowledge_base():
        rag.build_knowledge_base(movies)
    
    # Test queries
    test_queries = [
        "action movies with comedy",
        "emotional drama films",
        "scary horror movies",
        "romantic comedy"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        recommendations = rag.recommend(query, top_k=3)
        
        for _, movie in recommendations.iterrows():
            print(f"  {movie['title']} | {movie['genres']} | Score: {movie['similarity_score']:.3f}")

class CharacteristicVectorExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def extract_characteristics(self, texts):
        """
        Extract characteristic vectors for a list of texts
        """
        characteristics = []
        
        for text in texts:
            # Get embeddings
            embedding = self.model.encode(text, show_progress_bar=False)
            
            # For simplicity, we use the embedding as characteristic vector
            # In advanced version, we'll extract from each layer
            char_vector = {
                'embedding': embedding,
                'mean': np.mean(embedding),
                'std': np.std(embedding)
            }
            characteristics.append(char_vector)
            
        return characteristics
    
    def batch_extract(self, texts, batch_size=32):
        """Extract characteristics in batches"""
        all_characteristics = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_chars = self.extract_characteristics(batch)
            all_characteristics.extend(batch_chars)
            
        return all_characteristics

# Test the extractor
if __name__ == "__main__":
    extractor = CharacteristicVectorExtractor()
    test_texts = ["This is a great movie", "Amazing storyline"]
    characteristics = extractor.extract_characteristics(test_texts)
    print("Characteristic vectors extracted successfully!")
    print(f"Vector dimension: {len(characteristics[0]['embedding'])}")


class PoisonSimulator:
    def __init__(self):
        self.emotional_words = {
            'positive': ['amazing', 'brilliant', 'masterpiece', 'fantastic', 'excellent'],
            'negative': ['boring', 'disappointing', 'terrible', 'awful', 'poor']
        }
    
    def emotional_attack(self, description, attack_type='positive'):
        """Simple emotional poisoning attack"""
        words = description.split()
        
        # Replace 10% of words with emotional words
        num_replace = max(1, len(words) // 10)
        
        for _ in range(num_replace):
            if len(words) > 3:  # Ensure we have enough words
                replace_idx = random.randint(1, len(words)-2)  # Avoid first/last word
                words[replace_idx] = random.choice(self.emotional_words[attack_type])
        
        return ' '.join(words)
    
    def generate_poisoned_dataset(self, movies_df, poison_ratio=0.1):
        """Generate poisoned version of dataset"""
        clean_movies = movies_df.copy()
        poisoned_movies = movies_df.copy()
        
        # Select random movies to poison
        n_poison = int(len(movies_df) * poison_ratio)
        poison_indices = random.sample(range(len(movies_df)), n_poison)
        
        print(f"Poisoning {n_poison} movies...")
        
        for idx in poison_indices:
            original_desc = poisoned_movies.iloc[idx]['title'] + " " + str(poisoned_movies.iloc[idx].get('genres', ''))
            poisoned_desc = self.emotional_attack(original_desc, 'positive')
            
            # Create poisoned version
            poisoned_movies.iloc[idx] = poisoned_movies.iloc[idx].copy()
            poisoned_movies.at[idx, 'poisoned_description'] = poisoned_desc
            poisoned_movies.at[idx, 'is_poisoned'] = True
        
        # Mark clean movies
        clean_movies['is_poisoned'] = False
        clean_movies['poisoned_description'] = clean_movies['title'] + " " + clean_movies['genres'].astype(str)
        
        return clean_movies, poisoned_movies

# Test poisoning
if __name__ == "__main__":
    movies = pd.read_csv('data/ml-32m/movies.csv')
    simulator = PoisonSimulator()
    clean_df, poisoned_df = simulator.generate_poisoned_dataset(movies, poison_ratio=0.1)
    
    print("Clean dataset samples:")
    print(clean_df[['title', 'is_poisoned']].head())
    print("\nPoisoned dataset samples:")
    print(poisoned_df[poisoned_df['is_poisoned'] == True][['title', 'poisoned_description']].head())


class PoisonSimulator:
    def __init__(self):
        self.poison_keywords = ['free', 'download', 'watch now', 'click here', 'limited time', 
                               'exclusive', 'secret', 'hidden', 'special offer', 'buy now']
    
    def generate_poisoned_dataset(self, movies, poison_ratio=0.2):
        """Generate poisoned dataset by injecting malicious keywords"""
        clean_df = movies.copy()
        clean_df['is_poisoned'] = False
        clean_df['poisoned_description'] = clean_df['title']  # Using title as placeholder
        
        # Create poisoned samples
        n_poison = int(len(movies) * poison_ratio)
        poison_indices = random.sample(range(len(movies)), n_poison)
        
        poisoned_df = movies.iloc[poison_indices].copy()
        poisoned_df['is_poisoned'] = True
        
        # Add poison keywords to descriptions
        poisoned_descriptions = []
        for idx, row in poisoned_df.iterrows():
            base_desc = row['title']
            keyword = random.choice(self.poison_keywords)
            poisoned_desc = f"{base_desc} - {keyword}!"
            poisoned_descriptions.append(poisoned_desc)
        
        poisoned_df['poisoned_description'] = poisoned_descriptions
        
        return clean_df, poisoned_df

class CharacteristicVectorExtractor:
    def __init__(self, vector_size=50):
        self.vector_size = vector_size
    
    def simple_embedding(self, text):
        """Create simple text embedding based on character frequencies"""
        # Simple hash-based embedding for demonstration
        text = text.lower()
        embedding = np.zeros(self.vector_size)
        
        for i, char in enumerate(text[:self.vector_size]):
            if i < len(text):
                embedding[i] = ord(char) % 100 / 100.0  # Normalize
        
        # Fill remaining positions if text is shorter than vector_size
        if len(text) < self.vector_size:
            for i in range(len(text), self.vector_size):
                embedding[i] = (i * 13) % 100 / 100.0  # Pseudo-random fill
        
        return embedding
    
    def extract(self, description):
        """Extract characteristic vector from description"""
        embedding = self.simple_embedding(description)
        return {
            'embedding': embedding,
            'length': len(description),
            'has_special_chars': any(c in description for c in '!@#$%^&*()')
        }
    
    def batch_extract(self, descriptions):
        """Extract characteristics for multiple descriptions"""
        return [self.extract(desc) for desc in descriptions]

class SimplePoisonDetector:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.centroids = {}
        self.is_fitted = False
    
    def compute_centroids(self, characteristics, genres):
        """Compute centroids for each genre"""
        genre_vectors = {}
        
        for char_vec, genre in zip(characteristics, genres):
            if genre not in genre_vectors:
                genre_vectors[genre] = []
            genre_vectors[genre].append(char_vec['embedding'])
        
        # Compute centroid for each genre
        for genre, vectors in genre_vectors.items():
            self.centroids[genre] = np.mean(vectors, axis=0)
        
        self.is_fitted = True
        print(f"Computed centroids for {len(self.centroids)} genres")
    
    def detect_poison(self, characteristics, genres):
        """Detect poisoned items based on distance to centroid"""
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call compute_centroids first.")
        
        predictions = []
        distances = []
        
        for char_vec, genre in zip(characteristics, genres):
            if genre not in self.centroids:
                # Unknown genre, mark as suspicious
                predictions.append(True)
                distances.append(1.0)
                continue
            
            embedding = char_vec['embedding']
            centroid = self.centroids[genre]
            
            # Compute cosine distance
            similarity = cosine_similarity([embedding], [centroid])[0][0]
            distance = 1 - similarity
            
            distances.append(distance)
            predictions.append(distance > self.threshold)
        
        return predictions, distances

# Test detector
if __name__ == "__main__":
    # Create sample data if file doesn't exist
    try:
        movies = pd.read_csv('data/ml-32m/movies.csv')
    except:
        print("CSV file not found, creating sample data...")
        # Create sample movie data
        movies_data = {
            'movieId': range(1, 101),
            'title': [f'Movie {i}' for i in range(1, 101)],
            'genres': ['Action'] * 25 + ['Comedy'] * 25 + ['Drama'] * 25 + ['Horror'] * 25
        }
        movies = pd.DataFrame(movies_data)
    
    # Generate poisoned data
    simulator = PoisonSimulator()
    clean_df, poisoned_df = simulator.generate_poisoned_dataset(movies, poison_ratio=0.2)
    
    # Extract characteristics
    extractor = CharacteristicVectorExtractor()
    
    # Combine clean and poisoned for testing
    test_df = pd.concat([clean_df, poisoned_df[poisoned_df['is_poisoned'] == True]])
    descriptions = test_df['poisoned_description'].tolist()
    genres = test_df['genres'].tolist()
    true_labels = test_df['is_poisoned'].tolist()
    
    characteristics = extractor.batch_extract(descriptions)
    
    # Train detector on clean data only
    clean_chars = [c for c, label in zip(characteristics, true_labels) if not label]
    clean_genres = [g for g, label in zip(genres, true_labels) if not label]
    
    detector = SimplePoisonDetector(threshold=0.3)
    detector.compute_centroids(clean_chars, clean_genres)
    
    # Test detection
    predictions, distances = detector.detect_poison(characteristics, genres)
    
    # Evaluate
    accuracy = np.mean([p == tl for p, tl in zip(predictions, true_labels)])
    print(f"Detection Accuracy: {accuracy:.3f}")
    
    # Print some examples
    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        status = "POISONED" if predictions[i] else "CLEAN"
        actual = "POISONED" if true_labels[i] else "CLEAN"
        print(f"Movie: {descriptions[i][:30]}... | Prediction: {status} | Actual: {actual} | Distance: {distances[i]:.3f}")


class SecureRAGSystem:
    """
    Integrated System: RAG + Poison Detection
    """
    
    def __init__(self):
        self.rag = RAGRecommender()
        self.detector = None
        self.poison_simulator = PoisonSimulator()
        self.feature_extractor = CharacteristicVectorExtractor()
        self.poisoned_items = set()
        
    def initialize_system(self, movies_df, build_detector=True):
        """
        Initialize the complete system
        """
        print("Initializing Secure RAG System...")
        
        # Step 1: Build RAG knowledge base
        self.rag.build_knowledge_base(movies_df)
        
        if build_detector:
            # Step 2: Train poison detector on clean data
            self._train_detector(movies_df)
        
        print("Secure RAG System initialized successfully!")
    
    def _train_detector(self, clean_movies_df):
        """
        Train poison detector using existing components
        """
        print("Training poison detector...")
        
        # Use existing feature extractor
        descriptions = clean_movies_df['title'] + " " + clean_movies_df['genres'].astype(str)
        genres = clean_movies_df['genres'].tolist()
        
        # Extract characteristics
        characteristics = self.feature_extractor.batch_extract(descriptions.tolist())
        
        # Initialize and train detector
        self.detector = SimplePoisonDetector(threshold=0.25)
        self.detector.compute_centroids(characteristics, genres)
        
        print("Poison detector trained!")
    
    def secure_recommend(self, query, top_k=10, filter_poisons=True):
        """
        Get secure recommendations with poison filtering
        """
        if not self.rag.is_built:
            raise ValueError("RAG system not initialized. Call initialize_system() first.")
        
        # Step 1: Get initial recommendations
        candidates = self.rag.retrieve(query, top_k=top_k * 3)  # Get more for filtering
        
        if filter_poisons and self.detector:
            # Step 2: Filter out detected poisons
            candidates = self._filter_poisons(candidates)
        
        # Step 3: Rerank and return top_k
        final_recommendations = self.rag._simulate_llm_reranking(candidates, query, top_k)
        
        # Add security info
        final_recommendations['is_secure'] = True
        
        return final_recommendations.head(top_k)
    
    def _filter_poisons(self, candidates):
        """
        Filter out poisoned items from candidates
        """
        if self.detector is None:
            return candidates
        
        # Extract characteristics for candidate items
        candidate_descriptions = candidates['description'].tolist()
        candidate_genres = candidates['genres'].tolist()
        
        characteristics = self.feature_extractor.batch_extract(candidate_descriptions)
        
        # Detect poisons
        predictions, distances = self.detector.detect_poison(characteristics, candidate_genres)
        
        # Filter out poisoned items - FIXED VERSION
        clean_indices = []
        poisoned_indices = []
        
        for idx, pred in enumerate(predictions):
            if not pred:  # If not poisoned
                clean_indices.append(idx)
            else:  # If poisoned
                poisoned_indices.append(candidates.index[idx])
        
        # Get clean candidates using indices
        clean_candidates = candidates.iloc[clean_indices].copy()
        
        # Store poisoned items for analysis
        self.poisoned_items.update(poisoned_indices)
        
        print(f"Filtered out {len(poisoned_indices)} potentially poisoned items")
        
        return clean_candidates
    
    def simulate_attack_and_defense(self, movies_df, poison_ratio=0.1, test_queries=None):
        """
        Complete simulation: Attack -> Defense -> Evaluation
        """
        if test_queries is None:
            test_queries = [
                "action comedy movies",
                "emotional drama", 
                "scary horror films",
                "romantic stories"
            ]
        
        print("Starting Attack-Defense Simulation...")
        
        # Step 1: Generate poisoned dataset
        print("Simulating poisoning attacks...")
        clean_df, poisoned_df = self.poison_simulator.generate_poisoned_dataset(
            movies_df, poison_ratio=poison_ratio
        )
        
        # Combine datasets (simulating real-world scenario)
        test_df = pd.concat([clean_df, poisoned_df[poisoned_df['is_poisoned'] == True]])
        
        # Step 2: Rebuild RAG system with poisoned data
        self.rag.build_knowledge_base(test_df)
        
        # Step 3: Test secure recommendations
        results = []
        for query in test_queries:
            secure_recs = self.secure_recommend(query, top_k=5, filter_poisons=True)
            insecure_recs = self.rag.recommend(query, top_k=5)  # Without filtering
            
            results.append({
                'query': query,
                'secure_recommendations': len(secure_recs),
                'insecure_recommendations': len(insecure_recs),
                'poisoned_blocked': len(insecure_recs) - len(secure_recs)
            })
        
        return pd.DataFrame(results)

# Demo and testing
if __name__ == "__main__":
    # Load data
    movies = pd.read_csv('data/ml-32m/movies.csv')
    
    # Initialize secure system
    secure_system = SecureRAGSystem()
    secure_system.initialize_system(movies)
    
    # Test secure recommendations
    print("\n" + "="*50)
    print("SECURE RECOMMENDATIONS DEMO")
    print("="*50)
    
    test_query = "action movies with comedy elements"
    recommendations = secure_system.secure_recommend(test_query, top_k=5)
    
    print(f"Query: '{test_query}'")
    print("\nRecommended Movies:")
    for idx, movie in recommendations.iterrows():
        print(f"{movie['title']} | {movie['genres']} | Score: {movie['similarity_score']:.3f}")
    
    # Run attack-defense simulation
    print("\n" + "="*50)
    print("ATTACK-DEFENSE SIMULATION")
    print("="*50)
    
    simulation_results = secure_system.simulate_attack_and_defense(
        movies, poison_ratio=0.1, test_queries=["action movies", "comedy films"]
    )
    
    print("\nSimulation Results:")
    print(simulation_results)