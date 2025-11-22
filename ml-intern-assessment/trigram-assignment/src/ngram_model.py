import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import random
import math


class TrigramModel:
    """
    A Trigram Language Model that learns probabilities of word sequences
    and can generate text or calculate perplexity.
    
    Design Choices:
    1. Uses defaultdict for efficient storage and automatic initialization
    2. Implements Laplace (add-one) smoothing to handle unseen trigrams
    3. Supports text preprocessing with configurable options
    4. Provides both greedy and probabilistic text generation
    """
    
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the Trigram Model.
        
        Args:
            smoothing: Laplace smoothing parameter (default: 1.0)
        """
        self.smoothing = smoothing
        self.trigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.vocabulary = set()
        self.vocab_size = 0
        
    def preprocess_text(self, text: str, lowercase: bool = True) -> List[str]:
        """
        Preprocess text by tokenizing and optionally lowercasing.
        
        Args:
            text: Input text string
            lowercase: Whether to convert to lowercase
            
        Returns:
            List of tokens
        """
        if lowercase:
            text = text.lower()
        
        # Split on whitespace and punctuation, keep punctuation as separate tokens
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        # Add start and end tokens
        tokens = ['<START>', '<START>'] + tokens + ['<END>']
        
        return tokens
    
    def train(self, texts: List[str], lowercase: bool = True) -> None:
        """
        Train the trigram model on a list of texts.
        
        Args:
            texts: List of text strings to train on
            lowercase: Whether to preprocess with lowercasing
        """
        # Reset counts
        self.trigram_counts.clear()
        self.bigram_counts.clear()
        self.unigram_counts.clear()
        self.vocabulary.clear()
        
        # Process all texts
        for text in texts:
            tokens = self.preprocess_text(text, lowercase)
            
            # Update vocabulary
            self.vocabulary.update(tokens)
            
            # Count unigrams
            for token in tokens:
                self.unigram_counts[token] += 1
            
            # Count bigrams and trigrams
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
                
                bigram = (w1, w2)
                trigram = (w1, w2, w3)
                
                self.bigram_counts[bigram] += 1
                self.trigram_counts[trigram] += 1
        
        self.vocab_size = len(self.vocabulary)
    
    def get_probability(self, trigram: Tuple[str, str, str]) -> float:
        """
        Calculate probability of a trigram using Laplace smoothing.
        P(w3|w1,w2) = (count(w1,w2,w3) + α) / (count(w1,w2) + α * V)
        
        Args:
            trigram: Tuple of three words (w1, w2, w3)
            
        Returns:
            Probability of the trigram
        """
        w1, w2, w3 = trigram
        bigram = (w1, w2)
        
        numerator = self.trigram_counts[trigram] + self.smoothing
        denominator = self.bigram_counts[bigram] + (self.smoothing * self.vocab_size)
        
        if denominator == 0:
            return 1.0 / self.vocab_size
        
        return numerator / denominator
    
    def generate_next_word(self, context: Tuple[str, str], 
                          method: str = 'greedy') -> str:
        """
        Generate the next word given a two-word context.
        
        Args:
            context: Tuple of two preceding words
            method: 'greedy' for most probable word, 'sample' for sampling
            
        Returns:
            Next word
        """
        candidates = {}
        
        # Find all possible next words
        for trigram in self.trigram_counts:
            if trigram[0] == context[0] and trigram[1] == context[1]:
                prob = self.get_probability(trigram)
                candidates[trigram[2]] = prob
        
        # If no candidates found in training data, sample from vocabulary
        if not candidates:
            # Exclude special tokens from random generation
            valid_words = [w for w in self.vocabulary 
                          if w not in ['<START>', '<END>']]
            return random.choice(valid_words) if valid_words else '<END>'
        
        if method == 'greedy':
            return max(candidates.items(), key=lambda x: x[1])[0]
        elif method == 'sample':
            words = list(candidates.keys())
            probs = list(candidates.values())
            # Normalize probabilities
            total = sum(probs)
            probs = [p / total for p in probs]
            return random.choices(words, weights=probs)[0]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_text(self, start_words: Optional[Tuple[str, str]] = None,
                     max_length: int = 50, method: str = 'greedy') -> str:
        """
        Generate text using the trained model.
        
        Args:
            start_words: Optional starting bigram, defaults to ('<START>', '<START>')
            max_length: Maximum number of words to generate
            method: 'greedy' or 'sample' for word selection
            
        Returns:
            Generated text string
        """
        if start_words is None:
            context = ('<START>', '<START>')
        else:
            context = start_words
        
        words = []
        
        for _ in range(max_length):
            next_word = self.generate_next_word(context, method=method)
            
            if next_word == '<END>':
                break
            
            # Skip START tokens in output
            if next_word != '<START>':
                words.append(next_word)
            
            # Update context
            context = (context[1], next_word)
        
        return ' '.join(words)
    
    def calculate_perplexity(self, text: str, lowercase: bool = True) -> float:
        """
        Calculate perplexity of the model on given text.
        Perplexity = exp(-1/N * sum(log P(wi|wi-2,wi-1)))
        
        Args:
            text: Text to evaluate
            lowercase: Whether to preprocess with lowercasing
            
        Returns:
            Perplexity value (lower is better)
        """
        tokens = self.preprocess_text(text, lowercase)
        
        if len(tokens) < 3:
            return float('inf')
        
        log_prob_sum = 0.0
        count = 0
        
        for i in range(2, len(tokens)):
            trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
            prob = self.get_probability(trigram)
            
            # Avoid log(0)
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                log_prob_sum += math.log(1e-10)  # Small value for unseen
            
            count += 1
        
        if count == 0:
            return float('inf')
        
        # Calculate perplexity
        avg_log_prob = log_prob_sum / count
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def get_most_common_trigrams(self, n: int = 10) -> List[Tuple[Tuple[str, str, str], int]]:
        """
        Get the n most common trigrams.
        
        Args:
            n: Number of trigrams to return
            
        Returns:
            List of (trigram, count) tuples
        """
        return Counter(self.trigram_counts).most_common(n)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file (simple pickle implementation).
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        model_data = {
            'smoothing': self.smoothing,
            'trigram_counts': dict(self.trigram_counts),
            'bigram_counts': dict(self.bigram_counts),
            'unigram_counts': dict(self.unigram_counts),
            'vocabulary': self.vocabulary,
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.smoothing = model_data['smoothing']
        self.trigram_counts = defaultdict(int, model_data['trigram_counts'])
        self.bigram_counts = defaultdict(int, model_data['bigram_counts'])
        self.unigram_counts = defaultdict(int, model_data['unigram_counts'])
        self.vocabulary = model_data['vocabulary']
        self.vocab_size = model_data['vocab_size']


# Example usage
if __name__ == "__main__":
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog was sleeping under the tree.",
        "A quick brown dog runs in the park.",
        "The fox and the dog are friends.",
    ]
    
    # Initialize and train model
    model = TrigramModel(smoothing=1.0)
    model.train(training_texts)
    
    # Generate text
    print("Generated text (greedy):")
    print(model.generate_text(max_length=20, method='greedy'))
    print()
    
    print("Generated text (sampling):")
    print(model.generate_text(max_length=20, method='sample'))
    print()
    
    # Calculate perplexity
    test_text = "The quick dog jumps over the tree."
    perplexity = model.calculate_perplexity(test_text)
    print(f"Perplexity on test text: {perplexity:.2f}")
    print()
    
    # Show most common trigrams
    print("Most common trigrams:")
    for trigram, count in model.get_most_common_trigrams(5):
        print(f"  {trigram}: {count}")