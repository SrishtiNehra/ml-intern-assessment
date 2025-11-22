Quick Start Guide - Trigram Language Model
🚀 Get Started in 5 Minutes
Step 1: Setup (1 minute)
bash
# Clone the repository
git clone https://github.com/DesibleAI/ml-intern-assessment.git
cd ml-intern-assessment/trigram-assignment

# Install dependencies
pip install -r requirements.txt
Step 2: Run Tests (1 minute)
bash
# Run all tests
pytest tests/test_ngram.py -v

# Expected output: ✓ 25 passed
Step 3: Try the Demo (2 minutes)
bash
# Run interactive demonstration
python demo.py
Step 4: Use the Model (1 minute)
python
from src.ngram_model import TrigramModel

# Quick example
model = TrigramModel()
model.train([
    "The cat sat on the mat.",
    "The dog sat on the log."
])

print(model.generate_text(max_length=10))
# Output: "the cat sat on the mat"
📁 Project Structure
trigram-assignment/
├── src/
│   └── ngram_model.py       # ✅ Main implementation
├── tests/
│   └── test_ngram.py        # ✅ 25 comprehensive tests
├── demo.py                   # ✅ Interactive demonstrations
├── requirements.txt          # ✅ Dependencies
├── README.md                 # ✅ Full documentation
├── evaluation.md             # ✅ Design choices
└── QUICKSTART.md            # ✅ This file
✅ Verification Checklist
Run these commands to verify everything works:

bash
# 1. Test imports
python -c "from src.ngram_model import TrigramModel; print('✓ Import successful')"

# 2. Run tests
pytest tests/test_ngram.py -v
# Should see: 25 passed

# 3. Check code style (optional)
flake8 src/ --max-line-length=100
# Should see: 0 errors

# 4. Try example
python src/ngram_model.py
# Should see: Generated text examples
🎯 Common Tasks
Train on Custom Text
python
model = TrigramModel(smoothing=1.0)

# Your text data
my_texts = [
    "Your first sentence here.",
    "Your second sentence here.",
    # ... more texts
]

model.train(my_texts)
Generate Text
python
# Greedy (deterministic)
text = model.generate_text(max_length=20, method='greedy')

# Probabilistic (random)
text = model.generate_text(max_length=20, method='sample')

# With custom start
text = model.generate_text(
    start_words=('the', 'quick'),
    max_length=15
)
Evaluate Model
python
test_text = "The cat sat on the mat."
perplexity = model.calculate_perplexity(test_text)
print(f"Perplexity: {perplexity:.2f}")
# Lower is better!
Save/Load Model
python
# Save
model.save_model('my_model.pkl')

# Load
new_model = TrigramModel()
new_model.load_model('my_model.pkl')
🐛 Troubleshooting
Issue: Module not found
bash
# Solution: Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
Issue: Tests won't run
bash
# Solution: Install pytest
pip install pytest pytest-cov
Issue: Import errors in tests
bash
# Solution: Install in development mode
pip install -e .
📊 What to Expect
Test Results
25 tests should all pass
Coverage: ~95% of code
Time: < 1 second
Text Generation Quality
Small corpus (< 10 sentences): Mostly repeats training data
Medium corpus (10-100 sentences): Mix of memorization and recombination
Large corpus (100+ sentences): More creative, coherent text
Perplexity Values
Training data: 5-30 (very good fit)
Similar data: 20-60 (good fit)
Different data: 100+ (poor fit)
🎓 Learning Path
Beginner: Run tests and demo → understand basic concepts
Intermediate: Modify smoothing, try different texts
Advanced: Implement interpolation, add new smoothing methods
📚 Next Steps
Read README.md for detailed API documentation
Read evaluation.md to understand design choices
Check tests/test_ngram.py for usage examples
Experiment with different training data!
💡 Pro Tips
Start with small smoothing (0.1-0.5) for specific text generation
Use larger smoothing (1.0-2.0) for more diverse output
Lowercase=True reduces vocabulary, improves generalization
More training data = better quality generation
Perplexity is your friend for model comparison
✉️ Questions?
Check the README.md for detailed documentation
Look at demo.py for usage examples
Review test_ngram.py for edge cases
Open an issue on GitHub
Ready to go? Start with:

bash
pytest tests/test_ngram.py -v && python demo.py
Good luck! 🎉

