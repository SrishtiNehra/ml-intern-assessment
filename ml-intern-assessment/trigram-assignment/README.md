Trigram Language Model - Implementation Guide
Overview
This implementation provides a complete Trigram Language Model with support for:

Training on text corpora
Text generation (greedy and probabilistic sampling)
Perplexity calculation
Laplace (add-one) smoothing
Model persistence (save/load)
Installation
Prerequisites
Python 3.7 or higher
pip package manager
Setup
Clone the repository:
bash
git clone https://github.com/DesibleAI/ml-intern-assessment.git
cd ml-intern-assessment/trigram-assignment
Install dependencies:
bash
pip install -r requirements.txt
The requirements.txt should contain:

pytest>=7.0.0
pytest-cov>=4.0.0
Project Structure
trigram-assignment/
├── src/
│   └── ngram_model.py          # Main TrigramModel implementation
├── tests/
│   └── test_ngram.py           # Comprehensive test suite
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── evaluation.md               # Design choices documentation
Usage
Basic Usage
python
from src.ngram_model import TrigramModel

# Initialize model
model = TrigramModel(smoothing=1.0)

# Train on text data
training_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog was sleeping under the tree.",
    "A quick brown dog runs in the park."
]
model.train(training_texts)

# Generate text
generated = model.generate_text(max_length=20, method='greedy')
print(generated)

# Calculate perplexity
test_text = "The quick dog jumps."
perplexity = model.calculate_perplexity(test_text)
print(f"Perplexity: {perplexity:.2f}")
Text Generation Methods
Greedy generation (deterministic):

python
text = model.generate_text(max_length=50, method='greedy')
Probabilistic sampling (stochastic):

python
text = model.generate_text(max_length=50, method='sample')
Custom Starting Context
python
start_context = ('the', 'quick')
text = model.generate_text(start_words=start_context, max_length=30)
Model Persistence
Save model:

python
model.save_model('my_model.pkl')
Load model:

python
model = TrigramModel()
model.load_model('my_model.pkl')
Running Tests
Run the complete test suite:

bash
pytest trigram-assignment/tests/test_ngram.py -v
Run with coverage report:

bash
pytest trigram-assignment/tests/test_ngram.py --cov=src --cov-report=html
Run specific test:

bash
pytest trigram-assignment/tests/test_ngram.py::TestTrigramModel::test_train -v
Expected Test Output
test_ngram.py::TestTrigramModel::test_initialization PASSED
test_ngram.py::TestTrigramModel::test_preprocess_text PASSED
test_ngram.py::TestTrigramModel::test_train PASSED
test_ngram.py::TestTrigramModel::test_get_probability PASSED
test_ngram.py::TestTrigramModel::test_generate_text_default PASSED
test_ngram.py::TestTrigramModel::test_calculate_perplexity PASSED
...
======================== 25 passed in 0.45s ========================
API Reference
TrigramModel Class
__init__(smoothing: float = 1.0)
Initialize the trigram model with optional smoothing parameter.

train(texts: List[str], lowercase: bool = True) -> None
Train the model on a list of text strings.

Parameters:

texts: List of strings to train on
lowercase: Whether to convert text to lowercase (default: True)
generate_text(start_words: Optional[Tuple[str, str]] = None, max_length: int = 50, method: str = 'greedy') -> str
Generate text using the trained model.

Parameters:

start_words: Optional starting bigram
max_length: Maximum words to generate
method: 'greedy' or 'sample'
Returns: Generated text string

calculate_perplexity(text: str, lowercase: bool = True) -> float
Calculate perplexity of the model on given text (lower is better).

Parameters:

text: Text to evaluate
lowercase: Whether to preprocess with lowercasing
Returns: Perplexity value

get_probability(trigram: Tuple[str, str, str]) -> float
Calculate probability of a trigram with smoothing.

Returns: Probability value between 0 and 1

save_model(filepath: str) -> None
Save model to file.

load_model(filepath: str) -> None
Load model from file.

Performance Considerations
Memory: Model stores all n-gram counts, which can be large for big corpora
Training Time: O(N) where N is total number of tokens
Generation Time: O(V) per word where V is vocabulary size
Smoothing: Higher smoothing values make probability distribution more uniform
Troubleshooting
Common Issues
Issue: ModuleNotFoundError: No module named 'ngram_model' Solution: Ensure you're running from the correct directory and have the right PYTHONPATH

Issue: Tests fail with import errors Solution: Install the package in development mode:

bash
pip install -e .
Issue: Low quality text generation Solution:

Increase training data size
Adjust smoothing parameter (try values between 0.1 and 2.0)
Use probabilistic sampling instead of greedy
Contributing
When contributing, please:

Add tests for new features
Ensure all tests pass
Update documentation
Follow PEP 8 style guidelines
License
This project is part of the DesibleAI ML Intern Assessment.

Contact
For questions or issues, please open an issue on the GitHub repository.

