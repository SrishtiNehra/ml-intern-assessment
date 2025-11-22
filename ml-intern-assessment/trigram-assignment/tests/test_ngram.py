import pytest
import math
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ngram_model import TrigramModel


class TestTrigramModel:
    """Test suite for TrigramModel class."""
    
    @pytest.fixture
    def sample_texts(self):
        """Provide sample training texts."""
        return [
            "The cat sat on the mat.",
            "The dog sat on the log.",
            "The cat and the dog are friends.",
            "The mat is on the floor."
        ]
    
    @pytest.fixture
    def trained_model(self, sample_texts):
        """Provide a trained model."""
        model = TrigramModel(smoothing=1.0)
        model.train(sample_texts)
        return model
    
    def test_initialization(self):
        """Test model initialization."""
        model = TrigramModel(smoothing=0.5)
        assert model.smoothing == 0.5
        assert len(model.trigram_counts) == 0
        assert len(model.bigram_counts) == 0
        assert len(model.vocabulary) == 0
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        model = TrigramModel()
        tokens = model.preprocess_text("Hello World!")
        
        assert tokens[0] == '<START>'
        assert tokens[1] == '<START>'
        assert tokens[-1] == '<END>'
        assert 'hello' in tokens
        assert 'world' in tokens
        assert '!' in tokens
    
    def test_preprocess_no_lowercase(self):
        """Test preprocessing without lowercasing."""
        model = TrigramModel()
        tokens = model.preprocess_text("Hello World!", lowercase=False)
        
        assert 'Hello' in tokens
        assert 'World' in tokens
        assert 'hello' not in tokens
    
    def test_train(self, sample_texts):
        """Test model training."""
        model = TrigramModel()
        model.train(sample_texts)
        
        # Check vocabulary is populated
        assert len(model.vocabulary) > 0
        assert 'cat' in model.vocabulary
        assert 'dog' in model.vocabulary
        
        # Check counts are populated
        assert len(model.trigram_counts) > 0
        assert len(model.bigram_counts) > 0
        assert len(model.unigram_counts) > 0
    
    def test_get_probability(self, trained_model):
        """Test probability calculation."""
        # Test a trigram that should exist
        prob = trained_model.get_probability(('the', 'cat', 'sat'))
        assert 0 <= prob <= 1
        assert prob > 0  # With smoothing, probability should always be > 0
        
        # Test an unseen trigram (should still have non-zero prob due to smoothing)
        prob_unseen = trained_model.get_probability(('xyz', 'abc', 'def'))
        assert prob_unseen > 0
    
    def test_smoothing_effect(self, sample_texts):
        """Test that smoothing affects probabilities."""
        model1 = TrigramModel(smoothing=0.1)
        model2 = TrigramModel(smoothing=10.0)
        
        model1.train(sample_texts)
        model2.train(sample_texts)
        
        # Test with a seen trigram - smoothing affects all probabilities
        seen = ('the', 'cat', 'sat')
        prob1 = model1.get_probability(seen)
        prob2 = model2.get_probability(seen)
        
        # Different smoothing should give different probabilities
        assert prob1 != prob2
    
    def test_generate_next_word_greedy(self, trained_model):
        """Test greedy next word generation."""
        context = ('the', 'cat')
        next_word = trained_model.generate_next_word(context, method='greedy')
        
        assert isinstance(next_word, str)
        assert len(next_word) > 0
    
    def test_generate_next_word_sample(self, trained_model):
        """Test sampling next word generation."""
        context = ('the', 'dog')
        next_word = trained_model.generate_next_word(context, method='sample')
        
        assert isinstance(next_word, str)
        assert len(next_word) > 0
    
    def test_generate_text_default(self, trained_model):
        """Test text generation with default parameters."""
        text = trained_model.generate_text(max_length=20)
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Should not contain special tokens
        assert '<START>' not in text
        assert '<END>' not in text
    
    def test_generate_text_with_start(self, trained_model):
        """Test text generation with custom start words."""
        start = ('the', 'cat')
        text = trained_model.generate_text(start_words=start, max_length=10)
        
        assert isinstance(text, str)
    
    def test_generate_text_max_length(self, trained_model):
        """Test that generated text respects max length."""
        text = trained_model.generate_text(max_length=5)
        words = text.split()
        
        assert len(words) <= 5
    
    def test_calculate_perplexity(self, trained_model):
        """Test perplexity calculation."""
        test_text = "The cat sat on the mat."
        perplexity = trained_model.calculate_perplexity(test_text)
        
        assert perplexity > 0
        assert perplexity != float('inf')
        assert isinstance(perplexity, float)
    
    def test_perplexity_short_text(self, trained_model):
        """Test perplexity on very short text."""
        short_text = "Hi"
        perplexity = trained_model.calculate_perplexity(short_text)
        
        # Should return a valid perplexity value (not infinity)
        assert perplexity > 0
        assert isinstance(perplexity, float)
    
    def test_perplexity_comparison(self, sample_texts):
        """Test that perplexity is lower on training data."""
        model = TrigramModel()
        model.train(sample_texts)
        
        # Perplexity on training data
        train_perp = model.calculate_perplexity(sample_texts[0])
        
        # Perplexity on completely different text
        different_text = "Quantum physics explains molecular interactions."
        test_perp = model.calculate_perplexity(different_text)
        
        # Training data should have lower perplexity
        assert train_perp < test_perp
    
    def test_get_most_common_trigrams(self, trained_model):
        """Test getting most common trigrams."""
        common = trained_model.get_most_common_trigrams(n=5)
        
        assert len(common) <= 5
        assert all(isinstance(item, tuple) for item in common)
        assert all(len(item) == 2 for item in common)  # (trigram, count)
        
        # Check descending order
        counts = [count for _, count in common]
        assert counts == sorted(counts, reverse=True)
    
    def test_vocabulary_size(self, trained_model):
        """Test vocabulary size calculation."""
        assert trained_model.vocab_size > 0
        assert trained_model.vocab_size == len(trained_model.vocabulary)
    
    def test_empty_training(self):
        """Test behavior with empty training data."""
        model = TrigramModel()
        model.train([])
        
        assert len(model.vocabulary) == 0
        assert model.vocab_size == 0
    
    def test_single_sentence_training(self):
        """Test training on single sentence."""
        model = TrigramModel()
        model.train(["Hello world."])
        
        assert len(model.vocabulary) > 0
        text = model.generate_text(max_length=5)
        assert isinstance(text, str)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        model = TrigramModel()
        texts = ["Hello! How are you?", "I'm fine, thanks!"]
        model.train(texts)
        
        # Should handle punctuation
        assert '!' in model.vocabulary or '?' in model.vocabulary
    
    def test_save_and_load_model(self, trained_model, tmp_path):
        """Test model persistence."""
        # Save model
        filepath = tmp_path / "test_model.pkl"
        trained_model.save_model(str(filepath))
        
        # Load model
        new_model = TrigramModel()
        new_model.load_model(str(filepath))
        
        # Compare key attributes
        assert new_model.vocab_size == trained_model.vocab_size
        assert new_model.smoothing == trained_model.smoothing
        assert len(new_model.trigram_counts) == len(trained_model.trigram_counts)
    
    def test_model_consistency(self, sample_texts):
        """Test that model produces consistent results."""
        model1 = TrigramModel(smoothing=1.0)
        model2 = TrigramModel(smoothing=1.0)
        
        model1.train(sample_texts)
        model2.train(sample_texts)
        
        # Same training should produce same probabilities
        test_trigram = ('the', 'cat', 'sat')
        prob1 = model1.get_probability(test_trigram)
        prob2 = model2.get_probability(test_trigram)
        
        assert prob1 == prob2
    
    def test_invalid_generation_method(self, trained_model):
        """Test error handling for invalid generation method."""
        with pytest.raises(ValueError):
            trained_model.generate_next_word(('the', 'cat'), method='invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])