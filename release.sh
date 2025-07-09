#!/bin/bash

# SYNTHLA-EDU Release Script
# Usage: ./release.sh

echo "🚀 SYNTHLA-EDU v1.0 Release Script"
echo "====================================="

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: There are uncommitted changes. Please commit or stash them first."
    exit 1
fi

echo "✅ Git repository is clean"

# Create and push the tag
echo "📝 Creating git tag v1.0..."
git tag -a v1.0 -m "Release v1.0 - Initial release of SYNTHLA-EDU pipeline"

echo "📤 Pushing tag to remote..."
git push origin v1.0

echo "🎉 Release v1.0 has been tagged and pushed!"
echo ""
echo "Next steps:"
echo "1. Update the CI badge URL in README.md with your actual GitHub username"
echo "2. Update the citation information in README.md with your details"
echo "3. Create a GitHub release with the tag"
echo "4. Consider uploading to Zenodo for a DOI" 
