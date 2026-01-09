# Makefile for clustering pipeline
# Provides convenient shortcuts for common tasks

.PHONY: help install test quick-test clean run-flyhash run-all setup lint format

# Default target
help:
	@echo "Clustering Pipeline - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install all dependencies"
	@echo "  make setup         - Run full setup (install + download data)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make quick-test    - Run quick tests only"
	@echo ""
	@echo "Running Baselines:"
	@echo "  make run-flyhash       - Run FlyHash baseline on MNIST"
	@echo "  make run-diehl-cook    - Run Diehl & Cook baseline on MNIST"
	@echo "  make run-all           - Run all available baselines"
	@echo ""
	@echo "Testing Baselines:"
	@echo "  make test-baseline BASELINE=name  - Test specific baseline"
	@echo "  make test-baselines-quick         - Quick test all baselines"
	@echo "  make test-baselines-full          - Full test all baselines"
	@echo "  make batch-test-quick             - Batch test (quick mode)"
	@echo "  make batch-test-full              - Batch test (full mode)"
	@echo ""
	@echo "Development:"
	@echo "  make lint          - Check code style"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean outputs and cache"
	@echo ""
	@echo "Utilities:"
	@echo "  make status        - Show project status"
	@echo "  make check-env     - Check environment"
	@echo "  make docs          - Show documentation index"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

setup: install
	@echo "Running setup script..."
	bash scripts/setup.sh

# Testing
test:
	@echo "Running full test suite..."
	bash scripts/run_tests.sh

quick-test:
	@echo "Running quick tests..."
	python scripts/quick_test.py

# Running baselines (using main entry point)
run-flyhash:
	@echo "Running FlyHash baseline on MNIST..."
	python run.py --baseline flyhash

run-diehl-cook:
	@echo "Running Diehl & Cook baseline on MNIST..."
	python run.py --baseline diehl_cook

run-all:
	@echo "Running all baselines..."
	bash scripts/run_all_baselines.sh

# Main entry point commands
run:
	@echo "Usage: make run BASELINE=name [DATASET=name] [SEED=n]"
	@echo "Example: make run BASELINE=flyhash DATASET=mnist SEED=0"

list-baselines:
	@python run.py --list

# Testing specific baselines
test-baseline:
	@echo "Test single baseline (usage: make test-baseline BASELINE=flyhash)"
	python scripts/test_baseline.py $(BASELINE)

test-baselines-quick:
	@echo "Quick test of all baselines (1 seed)..."
	python scripts/test_baseline.py --all --seeds 0

test-baselines-full:
	@echo "Full test of all baselines (3 seeds)..."
	python scripts/test_baseline.py --all --seeds 0 1 2

batch-test-quick:
	@echo "Batch test (quick mode)..."
	bash scripts/batch_test.sh --quick

batch-test-full:
	@echo "Batch test (full mode)..."
	bash scripts/batch_test.sh --full

# Development
lint:
	@echo "Checking code style..."
	flake8 pipeline/ baselines/ scripts/ --max-line-length=100 || true
	@echo "Note: Install flake8 with: pip install flake8"

format:
	@echo "Formatting code with black..."
	black pipeline/ baselines/ scripts/ tests/ || true
	@echo "Note: Install black with: pip install black"

# Cleaning
clean:
	@echo "Cleaning outputs and cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "✓ Cleaned"

# Documentation
docs:
	@echo "Documentation Index"
	@echo "==================="
	@echo ""
	@echo "All documentation is in docs/ directory"
	@echo ""
	@echo "Main Documentation:"
	@echo "  README.md                               - Project overview"
	@echo "  docs/README.md                          - Documentation index"
	@echo "  docs/INSTALL.md                         - Installation guide"
	@echo "  docs/TROUBLESHOOTING.md                 - Common issues"
	@echo ""
	@echo "Implementation Guides:"
	@echo "  docs/clustering_hashing_baseline_guide.md  - Complete guide"
	@echo "  docs/baseline_code_availability_report.md  - Code survey"
	@echo "  docs/BINDSNET_INTEGRATION.md               - BindsNET details"
	@echo ""
	@echo "Testing:"
	@echo "  docs/TESTING_SUMMARY.md                 - Quick reference"
	@echo "  docs/BASELINE_TESTING.md                - Complete guide"
	@echo "  docs/TEST_GUIDE.md                      - General testing"
	@echo ""
	@echo "View full index: cat docs/README.md"
	@echo ""

# Pytest (if installed)
pytest:
	@echo "Running pytest..."
	pytest tests/ -v || echo "Install pytest with: pip install pytest"

# Download datasets
download-data:
	@echo "Downloading datasets..."
	python -c "from torchvision import datasets; datasets.MNIST('./data', download=True); datasets.FashionMNIST('./data', download=True)"
	@echo "✓ MNIST and Fashion-MNIST downloaded"
	@echo ""
	@echo "For SIFT1M, run: bash scripts/download_sift1m.sh"

# Check environment
check-env:
	@echo "Checking environment..."
	@python --version
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"
	@python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: Not installed"
	@python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "Scikit-learn: Not installed"
	@echo ""

# Show status
status:
	@echo "Project Status"
	@echo "=============="
	@echo ""
	@echo "Baselines:"
	@test -f baselines/flyhash/encoder.py && echo "  ✓ FlyHash" || echo "  ✗ FlyHash"
	@test -f baselines/diehl_cook/encoder.py && echo "  ✓ Diehl & Cook" || echo "  ✗ Diehl & Cook"
	@echo ""
	@echo "Data:"
	@test -d data/mnist && echo "  ✓ MNIST" || echo "  ✗ MNIST (run: make download-data)"
	@test -d data/sift1m && echo "  ✓ SIFT1M" || echo "  ✗ SIFT1M (run: bash scripts/download_sift1m.sh)"
	@echo ""
	@echo "Outputs:"
	@test -d outputs && echo "  ✓ outputs/ exists" || echo "  ✗ outputs/ missing"
	@find outputs -name "*.npy" 2>/dev/null | wc -l | xargs -I {} echo "  {} cached feature files"
	@echo ""
