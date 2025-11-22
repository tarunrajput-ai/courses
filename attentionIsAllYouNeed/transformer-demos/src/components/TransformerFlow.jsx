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
