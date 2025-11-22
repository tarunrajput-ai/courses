#!/bin/bash

# =============================================================================
# Transformer Demos Setup Script for MacBook
# Run: chmod +x setup-transformer-demos.sh && ./setup-transformer-demos.sh
# =============================================================================

set -e  # Exit on error

echo "🚀 Setting up Transformer Demos..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install node
fi

echo "✅ Node.js version: $(node --version)"

# Create project directory
PROJECT_NAME="transformer-demos"
echo "📁 Creating project: $PROJECT_NAME"

npm create vite@latest $PROJECT_NAME -- --template react
cd $PROJECT_NAME


