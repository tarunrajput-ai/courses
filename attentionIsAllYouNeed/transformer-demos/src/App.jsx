import React, { useState } from 'react';
import './index.css';
// import './App.css'; // Keep or remove this based on your project setup

// 1. Import all components (as you already have)
import AttentionExplainer from './components/AttentionExplainer.jsx';
import EmbeddingExplainer from './components/EmbeddingExplainer.jsx';
import PositionEncodingExplainer from './components/PositionEncodingExplainer.jsx';
import PEVisualization from './components/PEVisualization.jsx';
import QKVProjection from './components/QKVProjection.jsx';
import ScaledAttention from './components/ScaledAttention.jsx';
import MultiHeadAttention from './components/MultiHeadAttention.jsx';

// ============================================
// MAIN APP
// ============================================

// 2. Define a map of component names to the component itself
const componentMap = {
  AttentionExplainer: { component: AttentionExplainer, name: 'Attention Explainer' },
  EmbeddingExplainer: { component: EmbeddingExplainer, name: 'Embedding Explainer' },
  PositionEncodingExplainer: { component: PositionEncodingExplainer, name: 'Position Encoding' },
  PEVisualization: { component: PEVisualization, name: 'PE Visualization' },
  QKVProjection: { component: QKVProjection, name: 'QKV Projection' },
  ScaledAttention: { component: ScaledAttention, name: 'Scaled Attention' },
  MultiHeadAttention: { component: MultiHeadAttention, name: 'Multi-Head Attention' },
};

export default function App() {
  // 3. Use state to manage which component is currently selected
  const [currentView, setCurrentView] = useState('MultiHeadAttention');
  
  // 4. Determine which component to render
  const CurrentComponent = componentMap[currentView] ? componentMap[currentView].component : MultiHeadAttention;
  
  return (
    <div className="p-8">
      
      {/* 5. Create the Navigation/Selection Menu (User Input) */}
      <nav className="mb-8 p-4 bg-gray-100 rounded-lg shadow-md">
        <h1 className="text-xl font-bold mb-3">Transformer Visualizations</h1>
        <div className="flex flex-wrap gap-2">
          {Object.entries(componentMap).map(([key, value]) => (
            <button
              key={key}
              onClick={() => setCurrentView(key)}
              className={`px-4 py-2 text-sm rounded-full transition-colors 
                ${currentView === key 
                  ? 'bg-blue-600 text-white font-semibold' 
                  : 'bg-white text-gray-700 hover:bg-blue-100 border'
                }`
              }
            >
              {value.name}
            </button>
          ))}
        </div>
      </nav>

      {/* 6. Render the selected component */}
      <div className="component-container">
        <CurrentComponent />
      </div>
    </div>
  );
}
