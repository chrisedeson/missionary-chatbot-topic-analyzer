# Backend Makefile for BYU Pathway Topic Analyzer

.PHONY: help install clean dev setup check-env create-env migrate generate test lint format

# Variables
PYTHON := python3
PIP := pip
VENV_PATH := .venv
VENV_PYTHON := $(VENV_PATH)/bin/python
VENV_PIP := $(VENV_PATH)/bin/pip

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Complete project setup (create venv, install dependencies, setup database)
	@echo "🚀 Setting up BYU Pathway Topic Analyzer Backend..."
	@$(MAKE) install
	@$(MAKE) check-env
	@$(MAKE) generate
	@echo "✅ Setup complete! Run 'make dev' to start the development server."

install: ## Create virtual environment and install dependencies
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_PATH)
	@echo "📦 Installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

generate: ## Generate Prisma client
	@echo "🔧 Generating Prisma client..."
	npx prisma generate
	@echo "✅ Prisma client generated!"

migrate: ## Run database migrations
	@echo "🔄 Running database migrations..."
	npx prisma db push
	@echo "✅ Database migrations completed!"

dev: ## Start development server
	@echo "🚀 Starting development server..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	@if [ ! -f ".env" ]; then echo "❌ .env file not found. Run 'make create-env' and configure it."; exit 1; fi
	$(VENV_PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

check-env: ## Check if environment variables are set
	@echo "🔍 Checking environment setup..."
	@if [ -f ".env" ]; then \
		echo "✅ .env file found"; \
		if grep -q "OPENAI_API_KEY" .env; then \
			echo "✅ OPENAI_API_KEY found in .env"; \
		else \
			echo "⚠️  OPENAI_API_KEY not found in .env file"; \
		fi; \
		if grep -q "DATABASE_URL" .env; then \
			echo "✅ DATABASE_URL found in .env"; \
		else \
			echo "⚠️  DATABASE_URL not found in .env file"; \
		fi \
	else \
		echo "⚠️  .env file not found. Create one with your configuration"; \
	fi

create-env: ## Create .env file from template
	@echo "📝 Creating .env file from template..."
	@if [ ! -f ".env" ]; then \
		cp .env.template .env; \
		echo "✅ .env file created from template. Please edit it with your actual values."; \
	else \
		echo "⚠️  .env file already exists. Not overwriting."; \
	fi

test: ## Run tests
	@echo "🧪 Running tests..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(VENV_PYTHON) -m pytest tests/ -v

lint: ## Run linting
	@echo "🔍 Running linting..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(VENV_PYTHON) -m flake8 app/
	$(VENV_PYTHON) -m isort --check-only app/

format: ## Format code
	@echo "🎨 Formatting code..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(VENV_PYTHON) -m black app/
	$(VENV_PYTHON) -m isort app/

clean: ## Remove virtual environment and cache files
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV_PATH)
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

info: ## Show project information
	@echo "📊 BYU Pathway Topic Analyzer Backend"
	@echo "===================================="
	@echo "Python Version: $$(python3 --version)"
	@echo "Virtual Environment: $(VENV_PATH)"
	@echo "Main App: app/main.py"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install        # Install dependencies"
	@echo "  2. make create-env      # Create .env from template"
	@echo "  3. Edit .env with your configuration"
	@echo "  4. make generate       # Generate Prisma client"
	@echo "  5. make migrate        # Setup database"
	@echo "  6. make dev            # Start development server"

docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t byu-pathway-backend .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 --env-file .env byu-pathway-backend