"""
Demonstration script for TrigramModel
Shows various use cases and capabilities
"""

from src.ngram_model import TrigramModel


def demo_basic_usage():
    """Demonstrate basic model usage."""
    print("=" * 60)
    print("DEMO 1: Basic Training and Generation")
    print("=" * 60)
    
    # Sample training data
    training_texts = [
        "The cat sat on the mat.",
        "The dog sat on the log.",
        "The cat and the dog are friends.",
        "The mat is on the floor.",
        "The log is in the forest.",
        "The cat likes to sit on the mat.",
        "The dog likes to sit on the log."
    ]
    
    # Initialize and train
    print("\n1. Training model on sample texts...")
    model = TrigramModel(smoothing=1.0)
    model.train(training_texts)
    
    print(f"   - Vocabulary size: {model.vocab_size}")
    print(f"   - Total trigrams: {len(model.trigram_counts)}")
    print(f"   - Total bigrams: {len(model.bigram_counts)}")
    
    # Generate text
    print("\n2. Generating text (greedy method):")
    for i in range(3):
        text = model.generate_text(max_length=15, method='greedy')
        print(f"   {i+1}. {text}")
    
    print("\n3. Generating text (sampling method):")
    for i in range(3):
        text = model.generate_text(max_length=15, method='sample')
        print(f"   {i+1}. {text}")
    
    # Show most common trigrams
    print("\n4. Most common trigrams:")
    for trigram, count in model.get_most_common_trigrams(5):
        print(f"   {trigram}: {count} occurrences")
    
    print()


def demo_perplexity():
    """Demonstrate perplexity calculation."""
    print("=" * 60)
    print("DEMO 2: Perplexity Evaluation")
    print("=" * 60)
    
    # Train on some data
    training_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "Neural networks are used in deep learning.",
        "Artificial intelligence systems can learn from data."
    ]
    
    model = TrigramModel(smoothing=1.0)
    model.train(training_texts)
    
    print("\n1. Training completed.")
    
    # Test on similar text (should have low perplexity)
    test_text_similar = "Machine learning systems can learn from data."
    perp_similar = model.calculate_perplexity(test_text_similar)
    print(f"\n2. Perplexity on similar text: {perp_similar:.2f}")
    print(f"   Text: '{test_text_similar}'")
    
    # Test on different text (should have higher perplexity)
    test_text_different = "The quick brown fox jumps over the lazy dog."
    perp_different = model.calculate_perplexity(test_text_different)
    print(f"\n3. Perplexity on different text: {perp_different:.2f}")
    print(f"   Text: '{test_text_different}'")
    
    print(f"\n4. Analysis: Similar text has {'lower' if perp_similar < perp_different else 'higher'} perplexity")
    print(f"   (Lower perplexity = better fit to training data)")
    
    print()


def demo_custom_context():
    """Demonstrate generation with custom starting context."""
    print("=" * 60)
    print("DEMO 3: Custom Starting Context")
    print("=" * 60)
    
    # Train on story-like data
    training_texts = [
        "Once upon a time, there was a brave knight.",
        "The brave knight fought a dragon.",
        "The dragon lived in a dark cave.",
        "In the dark cave, there was treasure.",
        "The knight found the treasure and returned home.",
        "Once upon a time, in a faraway land.",
        "The brave knight saved the kingdom."
    ]
    
    model = TrigramModel(smoothing=1.0)
    model.train(training_texts)
    
    print("\n1. Model trained on short stories.")
    
    # Generate with different starting contexts
    contexts = [
        ('once', 'upon'),
        ('the', 'brave'),
        ('the', 'dragon'),
        ('the', 'knight')
    ]
    
    print("\n2. Generating stories with different starting contexts:\n")
    for context in contexts:
        text = model.generate_text(start_words=context, max_length=12, method='sample')
        print(f"   Starting with '{context[0]} {context[1]}':")
        print(f"   → {context[0]} {context[1]} {text}\n")
    
    print()


def demo_smoothing_effect():
    """Demonstrate effect of smoothing parameter."""
    print("=" * 60)
    print("DEMO 4: Smoothing Parameter Effects")
    print("=" * 60)
    
    training_texts = [
        "The sun rises in the east.",
        "The sun sets in the west.",
        "The moon shines at night."
    ]
    
    smoothing_values = [0.1, 1.0, 5.0]
    test_text = "The sun rises at night."
    
    print("\n1. Testing different smoothing values:\n")
    
    for smooth in smoothing_values:
        model = TrigramModel(smoothing=smooth)
        model.train(training_texts)
        perplexity = model.calculate_perplexity(test_text)
        
        print(f"   Smoothing = {smooth}:")
        print(f"   - Perplexity: {perplexity:.2f}")
        
        # Show probability of an unseen trigram
        unseen_trigram = ('xyz', 'abc', 'def')
        prob = model.get_probability(unseen_trigram)
        print(f"   - P(unseen trigram): {prob:.6f}")
        print()
    
    print("   Analysis: Higher smoothing gives more uniform probabilities")
    print("   for unseen events, but may over-smooth the model.")
    print()


def demo_model_persistence():
    """Demonstrate saving and loading models."""
    print("=" * 60)
    print("DEMO 5: Model Persistence")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Train a model
    training_texts = [
        "Python is a programming language.",
        "Machine learning uses Python.",
        "Data science requires programming skills."
    ]
    
    print("\n1. Training initial model...")
    model1 = TrigramModel(smoothing=1.0)
    model1.train(training_texts)
    
    text1 = model1.generate_text(max_length=10, method='greedy')
    print(f"   Generated text: {text1}")
    
    # Save model
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, "demo_model.pkl")
    
    print(f"\n2. Saving model to: {filepath}")
    model1.save_model(filepath)
    print("   Model saved successfully!")
    
    # Load model
    print("\n3. Loading model from file...")
    model2 = TrigramModel()
    model2.load_model(filepath)
    
    text2 = model2.generate_text(max_length=10, method='greedy')
    print(f"   Generated text: {text2}")
    
    # Verify consistency
    print("\n4. Verification:")
    print(f"   Original vocab size: {model1.vocab_size}")
    print(f"   Loaded vocab size: {model2.vocab_size}")
    print(f"   Match: {model1.vocab_size == model2.vocab_size} ✓")
    
    # Clean up
    os.remove(filepath)
    print(f"\n5. Cleaned up temporary file.")
    print()


def demo_real_world_example():
    """Demonstrate a more realistic use case."""
    print("=" * 60)
    print("DEMO 6: Real-World Example - News Headlines")
    print("=" * 60)
    
    # Sample news-like headlines
    training_texts = [
        "Breaking news: Scientists discover new species in Amazon rainforest.",
        "Technology companies announce major breakthrough in artificial intelligence.",
        "Global markets react to economic policy changes.",
        "Scientists publish research on climate change effects.",
        "New technology could revolutionize healthcare industry.",
        "Economic growth continues despite global challenges.",
        "Researchers discover potential breakthrough in cancer treatment.",
        "Artificial intelligence transforms modern healthcare practices."
    ]
    
    print("\n1. Training on news headline corpus...")
    model = TrigramModel(smoothing=0.5)  # Lower smoothing for more specific generation
    model.train(training_texts)
    
    print(f"   - Headlines trained: {len(training_texts)}")
    print(f"   - Vocabulary size: {model.vocab_size}")
    
    print("\n2. Generating synthetic headlines:\n")
    for i in range(5):
        headline = model.generate_text(max_length=12, method='sample')
        # Capitalize first letter
        if headline:
            headline = headline[0].upper() + headline[1:]
        print(f"   {i+1}. {headline}")
    
    # Evaluate on held-out headline
    test_headline = "Technology breakthrough announced by researchers."
    perplexity = model.calculate_perplexity(test_headline)
    print(f"\n3. Model perplexity on test headline: {perplexity:.2f}")
    
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "    TRIGRAM LANGUAGE MODEL - COMPREHENSIVE DEMO    ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    demos = [
        demo_basic_usage,
        demo_perplexity,
        demo_custom_context,
        demo_smoothing_effect,
        demo_model_persistence,
        demo_real_world_example
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            demo()
            if i < len(demos):
                input("Press Enter to continue to next demo...")
                print("\n")
        except Exception as e:
            print(f"\nError in demo: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - README.md: Usage instructions")
    print("  - evaluation.md: Design choices and testing guide")
    print("  - tests/test_ngram.py: Comprehensive test suite")
    print()


if __name__ == "__main__":
    main()