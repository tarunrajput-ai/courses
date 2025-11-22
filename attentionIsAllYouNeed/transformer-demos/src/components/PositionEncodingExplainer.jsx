import React, { useState, useMemo } from 'react';

export default function PositionalEncodingExplainer() {
  const [step, setStep] = useState(0);
  const [selectedPos, setSelectedPos] = useState(0);
  const [dModel, setDModel] = useState(8);

  // Calculate PE for a position and dimension
  const calcPE = (pos, i, d) => {
    const angle = pos / Math.pow(10000, (2 * Math.floor(i/2)) / d);
    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  };

  // Generate PE matrix
  const peMatrix = useMemo(() => {
    const matrix = [];
    for (let pos = 0; pos < 6; pos++) {
      const row = [];
      for (let i = 0; i < dModel; i++) {
        row.push(calcPE(pos, i, dModel));
      }
      matrix.push(row);
    }
    return matrix;
  }, [dModel]);

  const tokens = ["I", "love", "AI", "very", "much", "!"];
  
  // Sample embeddings (simplified)
  const embeddings = [
    [0.10, 0.20, 0.30, 0.40, 0.15, 0.25, 0.35, 0.45],
    [0.50, 0.10, 0.80, 0.20, 0.55, 0.15, 0.85, 0.25],
    [0.90, 0.70, 0.20, 0.10, 0.95, 0.75, 0.25, 0.15],
    [0.30, 0.40, 0.50, 0.60, 0.35, 0.45, 0.55, 0.65],
    [0.60, 0.30, 0.70, 0.40, 0.65, 0.35, 0.75, 0.45],
    [0.20, 0.50, 0.40, 0.70, 0.25, 0.55, 0.45, 0.75],
  ];

  const steps = [
    {
      title: "Why Do We Need Positional Encoding?",
      content: (
        <div className="space-y-3">
          <div className="bg-red-50 border-l-4 border-red-400 p-3">
            <p className="font-semibold text-red-700">The Problem:</p>
            <p className="text-sm">Transformers process all tokens in parallel—they have no sense of order!</p>
          </div>
          <div className="bg-gray-100 p-3 rounded text-sm font-mono">
            <p>"I love AI" and "AI love I"</p>
            <p className="text-gray-500">↓ Without position info ↓</p>
            <p>Same attention scores! 😱</p>
          </div>
          <div className="bg-green-50 border-l-4 border-green-400 p-3">
            <p className="font-semibold text-green-700">The Solution:</p>
            <p className="text-sm">Add unique position information to each token's embedding.</p>
          </div>
        </div>
      )
    },
    {
      title: "The Formula",
      content: (
        <div className="space-y-4">
          <div className="bg-blue-50 p-4 rounded-lg text-center">
            <p className="font-mono text-sm mb-2">For even dimensions (2i):</p>
            <p className="font-mono font-bold">PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg text-center">
            <p className="font-mono text-sm mb-2">For odd dimensions (2i+1):</p>
            <p className="font-mono font-bold">PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</p>
          </div>
          <div className="text-sm space-y-1">
            <p><strong>pos</strong> = position in sequence (0, 1, 2, ...)</p>
            <p><strong>i</strong> = dimension index (0, 1, 2, ... d_model-1)</p>
            <p><strong>d_model</strong> = embedding dimension (512 in paper)</p>
          </div>
        </div>
      )
    },
    {
      title: "Calculate Step by Step",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Let's calculate PE for position 1, dimension 0 (d_model=8):</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs space-y-2">
            <p>pos = 1, i = 0 (even), d_model = 8</p>
            <p className="border-t pt-2">Step 1: Calculate the divisor</p>
            <p className="ml-4">10000^(2×0/8) = 10000^0 = 1</p>
            <p className="border-t pt-2">Step 2: Calculate the angle</p>
            <p className="ml-4">angle = 1 / 1 = 1</p>
            <p className="border-t pt-2">Step 3: Apply sin (even dimension)</p>
            <p className="ml-4">PE(1,0) = sin(1) = <strong>0.841</strong></p>
          </div>
          <div className="bg-yellow-50 p-3 rounded text-sm">
            <p><strong>For dimension 1 (odd):</strong></p>
            <p className="font-mono">PE(1,1) = cos(1) = <strong>0.540</strong></p>
          </div>
        </div>
      )
    },
    {
      title: "Why Sin/Cos? The Wavelength Intuition",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Each dimension has a different wavelength (frequency):</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs w-12">dim 0-1:</span>
              <div className="flex-1 h-4 bg-gradient-to-r from-blue-500 via-white to-blue-500 rounded" style={{backgroundSize: '20%'}}></div>
              <span className="text-xs">Fast (λ=2π)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs w-12">dim 2-3:</span>
              <div className="flex-1 h-4 bg-gradient-to-r from-green-500 via-white to-green-500 rounded" style={{backgroundSize: '40%'}}></div>
              <span className="text-xs">Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs w-12">dim 6-7:</span>
              <div className="flex-1 h-4 bg-gradient-to-r from-purple-500 via-white to-purple-500 rounded" style={{backgroundSize: '80%'}}></div>
              <span className="text-xs">Slow (λ=20000π)</span>
            </div>
          </div>
          <div className="bg-blue-50 p-3 rounded text-sm">
            <p><strong>Like a binary clock!</strong></p>
            <p>Lower bits change fast, higher bits change slow. This lets the model learn both local and global position patterns.</p>
          </div>
        </div>
      )
    },
    {
      title: "Interactive PE Matrix",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Click a position to see its PE values (d_model={dModel}):</p>
          <div className="flex gap-2 mb-2">
            {tokens.slice(0, 6).map((t, i) => (
              <button
                key={i}
                onClick={() => setSelectedPos(i)}
                className={`px-2 py-1 rounded text-xs font-mono ${selectedPos === i ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
              >
                {i}: {t}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="text-xs font-mono w-full">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-1">dim</th>
                  {Array(dModel).fill(0).map((_, i) => (
                    <th key={i} className="p-1">{i}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="bg-blue-100">
                  <td className="p-1 font-bold">PE[{selectedPos}]</td>
                  {peMatrix[selectedPos].map((v, i) => (
                    <td key={i} className={`p-1 text-center ${i % 2 === 0 ? 'bg-blue-50' : 'bg-purple-50'}`}>
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500">
            <span className="bg-blue-50 px-1">Blue</span> = sin (even), 
            <span className="bg-purple-50 px-1 ml-1">Purple</span> = cos (odd)
          </p>
        </div>
      )
    },
    {
      title: "Full PE Matrix Visualization",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Complete PE matrix for 6 positions × {dModel} dimensions:</p>
          <div className="overflow-x-auto">
            <table className="text-xs font-mono">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-1">pos</th>
                  {Array(dModel).fill(0).map((_, i) => (
                    <th key={i} className="p-1">{i}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {peMatrix.map((row, pos) => (
                  <tr key={pos} className={pos % 2 === 0 ? 'bg-gray-50' : ''}>
                    <td className="p-1 font-bold">{pos}</td>
                    {row.map((v, i) => {
                      const intensity = Math.abs(v);
                      const color = v >= 0 
                        ? `rgba(59, 130, 246, ${intensity})` 
                        : `rgba(239, 68, 68, ${intensity})`;
                      return (
                        <td key={i} className="p-1 text-center" style={{backgroundColor: color}}>
                          {v.toFixed(1)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500">
            <span className="text-blue-500">■ Blue</span> = positive, 
            <span className="text-red-500 ml-2">■ Red</span> = negative
          </p>
        </div>
      )
    },
    {
      title: "Add PE to Embeddings",
      content: (
        <div className="space-y-3">
          <p className="text-sm font-semibold">Final step: Element-wise addition</p>
          <div className="text-xs font-mono space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-20">Embedding:</span>
              <span className="bg-green-100 px-2 py-1 rounded">[{embeddings[1].slice(0,4).map(v=>v.toFixed(2)).join(', ')}...]</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-20">+ PE[1]:</span>
              <span className="bg-blue-100 px-2 py-1 rounded">[{peMatrix[1].slice(0,4).map(v=>v.toFixed(2)).join(', ')}...]</span>
            </div>
            <div className="flex items-center gap-2 border-t pt-2">
              <span className="w-20">= Result:</span>
              <span className="bg-yellow-100 px-2 py-1 rounded font-bold">
                [{embeddings[1].slice(0,4).map((v,i) => (v + peMatrix[1][i]).toFixed(2)).join(', ')}...]
              </span>
            </div>
          </div>
          <div className="bg-green-50 border-l-4 border-green-400 p-3 text-sm">
            <p><strong>Result:</strong> Each token now has unique position info baked into its vector!</p>
          </div>
        </div>
      )
    },
    {
      title: "Key Properties",
      content: (
        <div className="space-y-2">
          <div className="bg-blue-50 p-3 rounded">
            <p className="font-semibold text-blue-700">1. Unique per position</p>
            <p className="text-sm">Each position gets a distinct encoding.</p>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <p className="font-semibold text-green-700">2. Bounded values</p>
            <p className="text-sm">Always between -1 and 1 (sin/cos range).</p>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <p className="font-semibold text-purple-700">3. Relative positions learnable</p>
            <p className="text-sm">PE(pos+k) can be represented as a linear function of PE(pos).</p>
          </div>
          <div className="bg-yellow-50 p-3 rounded">
            <p className="font-semibold text-yellow-700">4. Generalizes to longer sequences</p>
            <p className="text-sm">Works for sequences longer than those seen during training.</p>
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-2">📍 Positional Encoding</h1>
      <p className="text-center text-sm text-gray-600 mb-4">How Transformers Know Word Order</p>
      
      <div className="mb-4 flex items-center justify-between">
        <button 
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300 text-sm"
        >
          ← Prev
        </button>
        <span className="text-sm text-gray-600">{step + 1} / {steps.length}</span>
        <button 
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          disabled={step === steps.length - 1}
          className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300 text-sm"
        >
          Next →
        </button>
      </div>

      <div className="bg-white border rounded-lg shadow-sm">
        <div className="bg-gradient-to-r from-orange-500 to-red-500 text-white p-3 rounded-t-lg">
          <h2 className="font-bold">{steps[step].title}</h2>
        </div>
        <div className="p-4">
          {steps[step].content}
        </div>
      </div>

      <div className="mt-4 flex justify-center gap-1">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`w-3 h-3 rounded-full ${i === step ? 'bg-orange-500' : 'bg-gray-300'}`}
          />
        ))}
      </div>
    </div>
  );
}
