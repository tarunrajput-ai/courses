#!/bin/bash

# =============================================================================
# Transformer Demos Setup Script for MacBook
# Run: chmod +x setup-transformer-demos.sh && ./setup-transformer-demos.sh
# =============================================================================

PROJECT_NAME="transformer-demos"
cd $PROJECT_NAME

# Create components directory
mkdir -p src/components

echo "📝 Creating demo components..."

# =============================================================================
# Component 1: Embedding Demo
# =============================================================================
cat > src/components/EmbeddingDemo.jsx << 'COMPONENT'
import React, { useState } from 'react';

export default function EmbeddingDemo() {
  const [selectedToken, setSelectedToken] = useState(0);
  
  const vocabulary = {
    "<PAD>": 0, "<UNK>": 1, "I": 2, "love": 3, "AI": 4, 
    "you": 5, "hate": 6, "machine": 7, "learning": 8, "deep": 9
  };

  const embeddingMatrix = [
    [0.00, 0.00, 0.00, 0.00],
    [0.12, 0.34, 0.56, 0.78],
    [0.10, 0.20, 0.30, 0.40],
    [0.50, 0.10, 0.80, 0.20],
    [0.90, 0.70, 0.20, 0.10],
    [0.15, 0.25, 0.35, 0.45],
    [0.55, 0.15, 0.75, 0.25],
    [0.85, 0.65, 0.25, 0.15],
    [0.80, 0.60, 0.30, 0.20],
    [0.75, 0.55, 0.35, 0.25],
  ];

  const inputTokens = ["I", "love", "AI"];
  const tokenIds = [2, 3, 4];

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-purple-700">1. Token Embedding</h2>
      <p className="text-gray-600 mb-4">Convert tokens to dense vectors via lookup table</p>
      
      <div className="mb-6">
        <h3 className="font-semibold mb-2">Input: "{inputTokens.join(' ')}"</h3>
        <div className="flex gap-2 mb-4">
          {inputTokens.map((token, i) => (
            <button
              key={i}
              onClick={() => setSelectedToken(i)}
              className={`px-4 py-2 rounded ${selectedToken === i ? 'bg-purple-500 text-white' : 'bg-gray-200'}`}
            >
              {token} (ID: {tokenIds[i]})
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="font-semibold mb-2">Embedding Matrix (10 × 4)</h3>
          <div className="overflow-x-auto">
            <table className="text-xs font-mono">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-1">ID</th>
                  <th className="p-1">Token</th>
                  <th className="p-1" colSpan={4}>Vector</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(vocabulary).map(([token, id]) => (
                  <tr key={id} className={tokenIds[selectedToken] === id ? 'bg-yellow-200' : ''}>
                    <td className="p-1">{id}</td>
                    <td className="p-1">{token}</td>
                    {embeddingMatrix[id].map((v, i) => (
                      <td key={i} className="p-1">{v.toFixed(2)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h3 className="font-semibold mb-2">Lookup Result</h3>
          <div className="bg-green-100 p-4 rounded font-mono">
            <p>Token: "{inputTokens[selectedToken]}"</p>
            <p>ID: {tokenIds[selectedToken]}</p>
            <p>Vector: [{embeddingMatrix[tokenIds[selectedToken]].map(v => v.toFixed(2)).join(', ')}]</p>
          </div>
          <div className="mt-4 p-3 bg-blue-50 rounded text-sm">
            <strong>Key Insight:</strong> Embedding is just a table lookup using token ID as the row index!
          </div>
        </div>
      </div>
    </div>
  );
}
COMPONENT

# =============================================================================
# Component 2: Positional Encoding Demo
# =============================================================================
cat > src/components/PositionalEncodingDemo.jsx << 'COMPONENT'
import React, { useState, useMemo } from 'react';

export default function PositionalEncodingDemo() {
  const [maxLen, setMaxLen] = useState(20);
  const [dModel, setDModel] = useState(32);

  const calcPE = (pos, i, d) => {
    const divisor = Math.pow(10000, (2 * Math.floor(i/2)) / d);
    const angle = pos / divisor;
    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  };

  const peMatrix = useMemo(() => {
    const matrix = [];
    for (let pos = 0; pos < maxLen; pos++) {
      const row = [];
      for (let i = 0; i < dModel; i++) {
        row.push(calcPE(pos, i, dModel));
      }
      matrix.push(row);
    }
    return matrix;
  }, [maxLen, dModel]);

  const getColor = (val) => {
    if (val >= 0) {
      return `rgba(59, 130, 246, ${Math.abs(val)})`;
    } else {
      return `rgba(239, 68, 68, ${Math.abs(val)})`;
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-orange-700">2. Positional Encoding</h2>
      <p className="text-gray-600 mb-4">Add position information using sin/cos waves</p>

      <div className="flex gap-6 mb-4">
        <div>
          <label className="block text-sm font-semibold mb-1">Sequence Length: {maxLen}</label>
          <input 
            type="range" min="5" max="50" value={maxLen} 
            onChange={(e) => setMaxLen(parseInt(e.target.value))}
            className="w-40"
          />
        </div>
        <div>
          <label className="block text-sm font-semibold mb-1">d_model: {dModel}</label>
          <input 
            type="range" min="8" max="64" step="8" value={dModel} 
            onChange={(e) => setDModel(parseInt(e.target.value))}
            className="w-40"
          />
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-100 rounded font-mono text-sm">
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))<br/>
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
      </div>

      <div className="overflow-hidden rounded border">
        <div className="text-xs text-center bg-gray-200 p-1">
          PE Matrix ({maxLen} × {dModel}) — <span className="text-blue-600">Blue: positive</span>, <span className="text-red-600">Red: negative</span>
        </div>
        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
          {peMatrix.map((row, pos) => (
            <div key={pos} className="flex" style={{ height: '12px' }}>
              {row.map((val, dim) => (
                <div
                  key={dim}
                  style={{ flex: 1, backgroundColor: getColor(val), minWidth: '2px' }}
                  title={`pos=${pos}, dim=${dim}, val=${val.toFixed(3)}`}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-4 p-3 bg-yellow-50 rounded text-sm">
        <strong>Notice:</strong> Left columns oscillate fast (low dimensions), right columns oscillate slowly (high dimensions) — like a binary clock!
      </div>
    </div>
  );
}
COMPONENT

# =============================================================================
# Component 3: QKV Projection Demo
# =============================================================================
cat > src/components/QKVDemo.jsx << 'COMPONENT'
import React, { useState } from 'react';

export default function QKVDemo() {
  const X = [
    [1.34, 0.64, 0.81, 1.20],
    [1.81, 0.28, 0.22, 1.10],
    [0.70, 1.70, 0.70, 1.87],
  ];

  const Wq = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.3], [0.2, 0.3, 0.1, 0.5], [0.4, 0.2, 0.5, 0.1]];
  const Wk = [[0.3, 0.1, 0.4, 0.2], [0.2, 0.4, 0.1, 0.3], [0.1, 0.5, 0.3, 0.2], [0.5, 0.2, 0.2, 0.4]];
  const Wv = [[0.2, 0.3, 0.1, 0.5], [0.4, 0.1, 0.5, 0.2], [0.3, 0.2, 0.4, 0.1], [0.1, 0.5, 0.2, 0.3]];

  const matMul = (A, B) => {
    return A.map(row => 
      B[0].map((_, j) => row.reduce((sum, val, k) => sum + val * B[k][j], 0))
    );
  };

  const Q = matMul(X, Wq);
  const K = matMul(X, Wk);
  const V = matMul(X, Wv);

  const tokens = ["I", "love", "AI"];
  const fmt = (n) => n.toFixed(2);
  const fmtRow = (row) => `[${row.map(fmt).join(', ')}]`;

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-blue-700">3. Q, K, V Projections</h2>
      <p className="text-gray-600 mb-4">Linear transformations: Q = X·Wq, K = X·Wk, V = X·Wv</p>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-100 p-3 rounded">
          <h3 className="font-semibold mb-2">Input X (3×4)</h3>
          <div className="font-mono text-xs">
            {X.map((row, i) => <div key={i}>{tokens[i]}: {fmtRow(row)}</div>)}
          </div>
        </div>
        <div className="bg-blue-50 p-3 rounded">
          <h3 className="font-semibold mb-2">Weight Matrices (4×4 each)</h3>
          <div className="text-sm">
            <p>Wq, Wk, Wv are <strong>learned</strong> during training</p>
            <p className="text-gray-600">Shape: (d_model × d_model)</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="bg-blue-100 p-3 rounded">
          <h3 className="font-bold text-blue-700 mb-2">Q = X·Wq</h3>
          <div className="font-mono text-xs">
            {Q.map((row, i) => <div key={i}>{tokens[i]}: {fmtRow(row)}</div>)}
          </div>
          <p className="text-xs mt-2 text-gray-600">"What am I looking for?"</p>
        </div>
        <div className="bg-green-100 p-3 rounded">
          <h3 className="font-bold text-green-700 mb-2">K = X·Wk</h3>
          <div className="font-mono text-xs">
            {K.map((row, i) => <div key={i}>{tokens[i]}: {fmtRow(row)}</div>)}
          </div>
          <p className="text-xs mt-2 text-gray-600">"What do I contain?"</p>
        </div>
        <div className="bg-purple-100 p-3 rounded">
          <h3 className="font-bold text-purple-700 mb-2">V = X·Wv</h3>
          <div className="font-mono text-xs">
            {V.map((row, i) => <div key={i}>{tokens[i]}: {fmtRow(row)}</div>)}
          </div>
          <p className="text-xs mt-2 text-gray-600">"What info do I give?"</p>
        </div>
      </div>
    </div>
  );
}
COMPONENT

# =============================================================================
# Component 4: Scaled Dot-Product Attention Demo
# =============================================================================
cat > src/components/AttentionDemo.jsx << 'COMPONENT'
import React, { useState } from 'react';

export default function AttentionDemo() {
  const [step, setStep] = useState(0);

  const Q = [[1.10, 0.82, 1.10, 0.89], [1.18, 0.73, 0.98, 0.85], [1.42, 1.12, 1.23, 1.35]];
  const K = [[1.05, 0.95, 0.78, 1.12], [1.02, 0.68, 0.72, 0.92], [1.38, 1.28, 1.05, 1.42]];
  const V = [[0.98, 0.88, 1.02, 0.75], [0.85, 0.72, 0.95, 0.68], [1.25, 1.15, 1.32, 1.08]];

  const d_k = 4;
  const sqrt_dk = Math.sqrt(d_k);
  const tokens = ["I", "love", "AI"];

  const dotProduct = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);
  const QKt = Q.map(qRow => K.map(kRow => dotProduct(qRow, kRow)));
  const scaled = QKt.map(row => row.map(val => val / sqrt_dk));
  
  const softmax = (arr) => {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  };
  
  const weights = scaled.map(row => softmax(row));
  const output = weights.map(wRow => V[0].map((_, d) => wRow.reduce((sum, w, i) => sum + w * V[i][d], 0)));

  const fmt = (n) => n.toFixed(2);

  const steps = [
    { title: "QK^T", data: QKt, desc: "Dot product similarity scores" },
    { title: "Scaled (÷√d_k)", data: scaled, desc: `Divide by √${d_k} = ${fmt(sqrt_dk)}` },
    { title: "Softmax", data: weights, desc: "Normalize each row to sum=1" },
  ];

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-pink-700">4. Scaled Dot-Product Attention</h2>
      <p className="text-gray-600 mb-4">Attention(Q,K,V) = softmax(QK^T/√d_k)·V</p>

      <div className="flex gap-2 mb-4">
        {steps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`px-3 py-1 rounded ${step === i ? 'bg-pink-500 text-white' : 'bg-gray-200'}`}
          >
            {s.title}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="font-semibold mb-2">{steps[step].title}</h3>
          <p className="text-sm text-gray-600 mb-2">{steps[step].desc}</p>
          <table className="text-sm font-mono w-full">
            <thead>
              <tr className="bg-gray-200">
                <th className="p-2"></th>
                {tokens.map(t => <th key={t} className="p-2">K:{t}</th>)}
              </tr>
            </thead>
            <tbody>
              {steps[step].data.map((row, i) => (
                <tr key={i} className="border-b">
                  <td className="p-2 font-bold">Q:{tokens[i]}</td>
                  {row.map((val, j) => (
                    <td key={j} className={`p-2 text-center ${step === 2 && val > 0.35 ? 'bg-yellow-200 font-bold' : ''}`}>
                      {fmt(val)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3 className="font-semibold mb-2">Final Output (Weights × V)</h3>
          <div className="bg-purple-100 p-3 rounded font-mono text-sm">
            {output.map((row, i) => (
              <div key={i}>{tokens[i]}: [{row.map(fmt).join(', ')}]</div>
            ))}
          </div>
          <p className="text-sm mt-3 text-gray-600">
            Each token is now a weighted combination of all Value vectors!
          </p>
        </div>
      </div>
    </div>
  );
}
COMPONENT

# =============================================================================
# Component 5: Multi-Head Attention Demo
# =============================================================================
cat > src/components/MultiHeadDemo.jsx << 'COMPONENT'
import React, { useState } from 'react';

export default function MultiHeadDemo() {
  const [activeHead, setActiveHead] = useState(0);

  const tokens = ["I", "love", "AI"];
  
  const head1_attn = [[0.45, 0.30, 0.25], [0.35, 0.40, 0.25], [0.30, 0.25, 0.45]];
  const head2_attn = [[0.20, 0.50, 0.30], [0.25, 0.25, 0.50], [0.40, 0.35, 0.25]];
  
  const head1_out = [[0.45, 0.52, 0.38, 0.61], [0.48, 0.55, 0.41, 0.64], [0.51, 0.58, 0.44, 0.67]];
  const head2_out = [[0.72, 0.35, 0.68, 0.42], [0.75, 0.38, 0.71, 0.45], [0.78, 0.41, 0.74, 0.48]];

  const concat = head1_out.map((row, i) => [...row, ...head2_out[i]]);
  const fmt = (n) => n.toFixed(2);

  const heads = [
    { name: "Head 1", attn: head1_attn, out: head1_out, color: "green", desc: "Syntactic patterns" },
    { name: "Head 2", attn: head2_attn, out: head2_out, color: "yellow", desc: "Semantic patterns" },
  ];

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-indigo-700">5. Multi-Head Attention</h2>
      <p className="text-gray-600 mb-4">Multiple parallel attention heads learn different patterns</p>

      <div className="flex gap-2 mb-4">
        {heads.map((h, i) => (
          <button
            key={i}
            onClick={() => setActiveHead(i)}
            className={`px-4 py-2 rounded ${activeHead === i ? 'bg-indigo-500 text-white' : 'bg-gray-200'}`}
          >
            {h.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className={`bg-${heads[activeHead].color}-100 p-4 rounded`}>
          <h3 className="font-bold mb-2">{heads[activeHead].name} Attention Weights</h3>
          <p className="text-sm text-gray-600 mb-2">{heads[activeHead].desc}</p>
          <table className="text-sm font-mono w-full">
            <thead>
              <tr className="bg-gray-200">
                <th></th>
                {tokens.map(t => <th key={t} className="p-1">{t}</th>)}
              </tr>
            </thead>
            <tbody>
              {heads[activeHead].attn.map((row, i) => (
                <tr key={i}>
                  <td className="p-1 font-bold">{tokens[i]}</td>
                  {row.map((v, j) => (
                    <td key={j} className={`p-1 text-center ${v >= 0.4 ? 'bg-yellow-300 font-bold' : ''}`}>
                      {fmt(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-gray-100 p-4 rounded">
          <h3 className="font-bold mb-2">{heads[activeHead].name} Output (3×4)</h3>
          <div className="font-mono text-sm">
            {heads[activeHead].out.map((row, i) => (
              <div key={i}>{tokens[i]}: [{row.map(fmt).join(', ')}]</div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-purple-100 p-4 rounded">
        <h3 className="font-bold mb-2">Concatenated Output (3×8)</h3>
        <p className="text-sm text-gray-600 mb-2">Concat(head1, head2) → then multiply by W^O</p>
        <div className="font-mono text-xs">
          {concat.map((row, i) => (
            <div key={i}>{tokens[i]}: [{row.map(fmt).join(', ')}]</div>
          ))}
        </div>
      </div>
    </div>
  );
}
COMPONENT

# =============================================================================
# Component 6: Complete Transformer Flow
# =============================================================================
cat > src/components/TransformerFlow.jsx << 'COMPONENT'
import React, { useState } from 'react';

export default function TransformerFlow() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    { name: "Input", desc: "Tokenize text", color: "gray" },
    { name: "Embedding", desc: "Lookup vectors", color: "purple" },
    { name: "Positional Enc", desc: "Add positions", color: "orange" },
    { name: "Multi-Head Attn", desc: "Self-attention", color: "blue" },
    { name: "Add & Norm", desc: "Residual + LayerNorm", color: "green" },
    { name: "FFN", desc: "Feed-forward network", color: "pink" },
    { name: "Add & Norm", desc: "Residual + LayerNorm", color: "green" },
    { name: "Output", desc: "Linear + Softmax", color: "red" },
  ];

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-gray-700">6. Complete Transformer Flow</h2>
      <p className="text-gray-600 mb-4">Click each step to see details</p>

      <div className="flex flex-col items-center gap-2">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-4 w-full max-w-md">
            <button
              onClick={() => setActiveStep(i)}
              className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                activeStep === i 
                  ? `bg-${step.color}-100 border-${step.color}-500` 
                  : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
              }`}
            >
              <div className="font-bold">{step.name}</div>
              <div className="text-sm text-gray-600">{step.desc}</div>
            </button>
            {i < steps.length - 1 && (
              <div className="text-2xl text-gray-400">↓</div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded">
        <h3 className="font-bold mb-2">Step {activeStep + 1}: {steps[activeStep].name}</h3>
        <div className="text-sm">
          {activeStep === 0 && <p>"I love AI" → ["I", "love", "AI"] → [2, 3, 4]</p>}
          {activeStep === 1 && <p>Token IDs → Dense vectors via embedding matrix lookup</p>}
          {activeStep === 2 && <p>Add sin/cos positional encoding to preserve word order</p>}
          {activeStep === 3 && <p>8 parallel attention heads compute Q, K, V and attend</p>}
          {activeStep === 4 && <p>Add input (residual) and normalize: LayerNorm(x + Attention(x))</p>}
          {activeStep === 5 && <p>Two linear layers with ReLU: FFN(x) = ReLU(xW₁)W₂</p>}
          {activeStep === 6 && <p>Another residual connection and layer normalization</p>}
          {activeStep === 7 && <p>Project to vocab size and apply softmax for next token probabilities</p>}
        </div>
      </div>

      <div className="mt-4 p-3 bg-yellow-50 rounded text-sm">
        <strong>Note:</strong> The encoder has 6 identical layers (steps 3-6 repeated). The decoder is similar but adds masked attention and cross-attention.
      </div>
    </div>
  );
}
COMPONENT

# =============================================================================
# Main App.jsx
# =============================================================================
cat > src/App.jsx << 'APPCODE'
import React, { useState } from 'react';
import EmbeddingDemo from './components/EmbeddingDemo';
import PositionalEncodingDemo from './components/PositionalEncodingDemo';
import QKVDemo from './components/QKVDemo';
import AttentionDemo from './components/AttentionDemo';
import MultiHeadDemo from './components/MultiHeadDemo';
import TransformerFlow from './components/TransformerFlow';

export default function App() {
