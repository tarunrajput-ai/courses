#!/bin/bash

# ============================================
# STEP 1: Run this in Terminal
# ============================================

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    # Download from https://nodejs.org or:
    brew install node
fi

# Create project
npm create vite@latest transformer-demo -- --template react
cd transformer-demo
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# ============================================
# STEP 2: Replace these files manually
# ============================================

# File: tailwind.config.js
# Copy this content:
: '
/** @type {import("tailwindcss").Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
'

# File: src/index.css
# Replace with:
: '
@tailwind base;
@tailwind components;
@tailwind utilities;
'

# File: src/App.jsx
# Replace with the React code from the next artifact

# ============================================
# STEP 3: Run the app
# ============================================
npm run dev

# Open http://localhost:5173 in browser
