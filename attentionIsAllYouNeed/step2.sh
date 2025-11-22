#!/bin/bash

# =============================================================================
# Transformer Demos Setup Script for MacBook
# Run: chmod +x setup-transformer-demos.sh && ./setup-transformer-demos.sh
# =============================================================================

PROJECT_NAME="transformer-demos"
cd $PROJECT_NAME

# Install dependencies
echo "📦 Installing dependencies..."
npm install
npm install -D tailwindcss postcss autoprefixer lucide-react

# Initialize Tailwind
npx tailwindcss init -p

# Configure Tailwind
cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOF

# Setup CSS
cat > src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

