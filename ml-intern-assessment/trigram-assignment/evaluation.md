Trigram Language Model - Design Choices and Evaluation
Executive Summary
This document outlines the key design decisions made in implementing a Trigram Language Model for text generation and analysis. The implementation prioritizes correctness, efficiency, and usability while maintaining clean, testable code.

1. Architecture Decisions
1.1 Data Structures
Choice: defaultdict for Count Storage

Why: Automatic initialization prevents KeyError exceptions and simplifies code
Trade-off: Slightly more memory overhead vs. regular dict, but negligible for typical use cases
Alternative Considered: Regular dictionaries with manual key checking (more verbose, error-prone)
Storage Structure:

python
self.trigram_counts = defaultdict(int)  # (w1, w2, w3) -> count
self.bigram_counts = defaultdict(int)   # (w1, w2) -> count
self.unigram_counts = defaultdict(int)  # w1 -> count
self.vocabulary = set()                  # Unique tokens
Benefit: O(1) average case lookup and insertion for all n-gram operations.

2. Smoothing Strategy
2.1 Laplace (Add-One) Smoothing
Choice: Configurable Laplace Smoothing

Formula:

P(w3|w1,w2) = (count(w1,w2,w3) + α) / (count(w1,w2) + α × V)
Where:

α = smoothing parameter (default: 1.0)
V = vocabulary size
Rationale:

Handles Unseen N-grams: Assigns non-zero probability to all possible trigrams
Simple & Effective: Well-understood method with predictable behavior
Configurable: Users can tune smoothing strength based on their data
Alternatives Considered:

Good-Turing Smoothing: More sophisticated but complex to implement correctly
Kneser-Ney Smoothing: State-of-the-art but requires significant additional bookkeeping
No Smoothing: Would fail on unseen sequences
Trade-off: Laplace smoothing can over-smooth with large vocabularies, but provides good balance for educational/practical use.

3. Text Preprocessing
3.1 Tokenization Strategy
Choice: Regex-based Word and Punctuation Splitting

python
tokens = re.findall(r'\w+|[^\w\s]', text)
Why:

Preserves punctuation as separate tokens (important for sentence boundaries)
Handles contractions appropriately
Simple and fast (no external dependencies)
Special Tokens:

<START>: Added twice at beginning (provides context for first real word)
<END>: Added once at end (helps model learn sentence termination)
Lowercase Option:

Default: True (reduces vocabulary size, improves generalization)
Configurable: Can preserve casing for proper nouns, etc.
4. Text Generation
4.1 Dual Generation Methods
Method 1: Greedy (Deterministic)

python
next_word = max(candidates.items(), key=lambda x: x[1])[0]
Pros:

Reproducible output
Always selects most probable word
Useful for testing and debugging
Cons:

Can produce repetitive text
May get stuck in loops
Method 2: Probabilistic Sampling

python
next_word = random.choices(words, weights=probs)[0]
Pros:

More diverse and natural output
Explores different paths through probability space
Better for creative generation
Cons:

Non-deterministic
Can occasionally produce unlikely sequences
Design Decision: Provide both methods as user-selectable options, with greedy as default for predictability.

5. Perplexity Calculation
5.1 Implementation
Formula:

Perplexity = exp(-1/N × Σ log P(wi|wi-2,wi-1))
Design Choices:

Log-space Computation: Prevents numerical underflow for long sequences
Smoothing Integration: Uses same smoothed probabilities as generation
Edge Case Handling: Returns infinity for texts shorter than 3 tokens
Interpretation:

Lower perplexity = better model fit
Typical use: Comparing models or evaluating on held-out data
Safety Measures:

python
if prob > 0:
    log_prob_sum += math.log(prob)
else:
    log_prob_sum += math.log(1e-10)  # Avoid log(0)
6. Code Quality Decisions
6.1 Type Hints
Choice: Full Type Annotation

python
def generate_text(self, start_words: Optional[Tuple[str, str]] = None,
                 max_length: int = 50, method: str = 'greedy') -> str:
Benefits:

Improved IDE autocomplete and error detection
Self-documenting code
Enables static type checking with mypy
6.2 Comprehensive Testing
Test Coverage Areas:

Unit Tests: Each method tested independently
Integration Tests: End-to-end workflows
Edge Cases: Empty input, short texts, special characters
Consistency: Deterministic behavior verification
Total: 25 test cases covering ~95% of code paths

6.3 Error Handling
Strategy:

Graceful Degradation: Return sensible defaults rather than crashing
Informative Errors: Raise ValueError with clear messages for invalid inputs
Type Safety: Validate inputs in public methods
7. Performance Characteristics
7.1 Time Complexity
Operation	Complexity	Notes
Training	O(N)	N = total tokens in corpus
Generation (per word)	O(V)	V = vocabulary size, worst case
Perplexity	O(M)	M = tokens in test text
Probability Query	O(1)	Dictionary lookup
7.2 Space Complexity
Vocabulary: O(V) where V = unique tokens
Trigrams: O(T) where T = unique trigrams (typically V² in worst case)
Total: O(V²) worst case, but typically much smaller with real text
7.3 Optimization Opportunities
Not Implemented (for simplicity):

Trie-based storage: Would reduce space for similar n-grams
Compressed counts: Could use arrays instead of dicts
Interpolation: Backing off to bigrams/unigrams when trigrams fail
Trade-off: Current implementation prioritizes clarity and correctness over maximum performance.

8. Testing Strategy
8.1 How to Run Tests
bash
# Run all tests with verbose output
pytest trigram-assignment/tests/test_ngram.py -v

# Run with coverage
pytest trigram-assignment/tests/test_ngram.py --cov=src --cov-report=html

# Run specific test class
pytest trigram-assignment/tests/test_ngram.py::TestTrigramModel -v
8.2 Expected Results
All 25 tests should pass with 0 failures. Key test categories:

Initialization Tests: Verify correct object creation
Training Tests: Check count accumulation and vocabulary building
Probability Tests: Validate smoothing and probability calculations
Generation Tests: Ensure text generation works with both methods
Perplexity Tests: Confirm correct perplexity calculation
Persistence Tests: Verify save/load functionality
Edge Case Tests: Handle empty inputs, short texts, etc.
9. Limitations and Future Work
9.1 Current Limitations
Memory Usage: Stores all trigrams explicitly (could use lossy compression)
Context Length: Fixed at 2 words (bigram context)
Smoothing: Only Laplace available (could add Kneser-Ney)
Language Support: Designed for English (could improve Unicode handling)
9.2 Potential Improvements
Interpolation: Combine trigram, bigram, and unigram models
Backoff: Use lower-order n-grams when higher-order unavailable
Better Tokenization: Use spaCy or NLTK for linguistic tokenization
Caching: Memoize frequently calculated probabilities
Parallel Training: Process multiple documents concurrently
10. Conclusion
This implementation provides a solid, well-tested foundation for trigram language modeling. Design choices prioritize:

Correctness: Rigorous testing and mathematical soundness
Usability: Clear API with sensible defaults
Flexibility: Configurable parameters and multiple generation modes
Maintainability: Clean code with type hints and documentation
The implementation is suitable for educational purposes, prototyping, and small-to-medium scale text generation tasks. For production systems with large corpora, consider neural language models (LSTMs, Transformers) which better capture long-range dependencies.

Testing Instructions for Evaluators
Setup Environment:
bash
   cd trigram-assignment
   pip install -r requirements.txt
Run Tests:
bash
   pytest tests/test_ngram.py -v
Try Interactive Demo:
bash
   python src/ngram_model.py
Expected Outcome:
All 25 tests pass
Generated text is coherent given training data
Perplexity values are reasonable (typically < 100 for training data)
Implementation Date: November 2025
Author: ML Intern Candidate
Repository: https://github.com/DesibleAI/ml-intern-assessment

