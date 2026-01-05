#!/bin/bash
# Setup script for pre-commit hooks

set -e

echo "🔧 Setting up pre-commit hooks..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."

    # Try to install with pip/pip3
    if command -v pip3 &> /dev/null; then
        pip3 install pre-commit
    elif command -v pip &> /dev/null; then
        pip install pre-commit
    elif command -v brew &> /dev/null; then
        echo "Installing via Homebrew..."
        brew install pre-commit
    else
        echo "❌ Error: Could not find pip, pip3, or brew to install pre-commit"
        echo "Please install pre-commit manually: https://pre-commit.com/#install"
        exit 1
    fi
fi

echo "✅ pre-commit is installed"

# Install the git hooks
echo "📌 Installing git hooks..."
pre-commit install

# Install commit-msg hook
pre-commit install --hook-type commit-msg

echo "🎉 Pre-commit hooks installed successfully!"
echo ""
echo "To run hooks manually on all files: pre-commit run --all-files"
echo "To update hooks to latest versions: pre-commit autoupdate"
echo ""
echo "Hooks will now run automatically on every commit."
