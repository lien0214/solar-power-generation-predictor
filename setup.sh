#!/bin/bash
# Quick setup script for the Solar Power Prediction API

echo "🔧 Setting up Solar Power Prediction API..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Run this from the repo/ directory."
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file (edit as needed)"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the server:"
echo "  python run.py"
echo ""
echo "Or with uvicorn:"
echo "  uvicorn app.main:app --reload"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
