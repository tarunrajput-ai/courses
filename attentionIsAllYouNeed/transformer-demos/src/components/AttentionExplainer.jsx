import React, { useState } from 'react';

export default function AttentionExplainer() {

  const [step, setStep] = useState(0);
  
  const steps = [
    {
      title: "1. Input Embedding",
      description: "Convert input tokens to vectors",
      data: {
        input: ["I", "love", "AI"],
        output: [
          [0.1, 0.2, 0.3, 0.4],
          [0.5, 0.1, 0.8, 0.2],
          [0.9, 0.7, 0.2, 0.1]
        ]
      },
      explanation: "Each word is mapped to a dense vector (d_model=4 for simplicity). In the real paper, d_model=512."
    },
    {
      title: "2. Positional Encoding",
      description: "Add position information using sine/cosine",
      data: {
        positions: [0, 1, 2],
        pe: [
          [0.00, 1.00, 0.00, 1.00],
          [0.84, 0.54, 0.01, 1.00],
          [0.91, -0.42, 0.02, 1.00]
        ],
        combined: [
          [0.10, 1.20, 0.30, 1.40],
          [1.34, 0.64, 0.81, 1.20],
          [1.81, 0.28, 0.22, 1.10]
        ]
      },
      explanation: "PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)). Added to embeddings."
    },
    {
      title: "3. Create Q, K, V Matrices",
      description: "Linear projections for Query, Key, Value",
      data: {
        Wq: "4×4 weight matrix",
        Wk: "4×4 weight matrix", 
        Wv: "4×4 weight matrix",
        Q: [[0.5, 0.3, 0.8, 0.2], [0.9, 0.4, 0.1, 0.7], [0.2, 0.8, 0.5, 0.3]],
        K: [[0.3, 0.7, 0.2, 0.9], [0.6, 0.1, 0.8, 0.4], [0.4, 0.5, 0.3, 0.7]],
        V: [[0.8, 0.2, 0.6, 0.1], [0.3, 0.9, 0.4, 0.5], [0.7, 0.4, 0.2, 0.8]]
      },
      explanation: "Q = X·Wq, K = X·Wk, V = X·Wv. Each token gets its own query, key, and value vectors."
    },
    {
      title: "4. Scaled Dot-Product Attention",
      description: "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
      data: {
        QKt: [
          [0.71, 0.62, 0.58],
          [0.89, 0.73, 0.81],
          [0.54, 0.47, 0.52]
        ],
        scaled: [
          [0.36, 0.31, 0.29],
          [0.45, 0.37, 0.41],
          [0.27, 0.24, 0.26]
        ],
        softmax: [
          [0.35, 0.33, 0.32],
          [0.36, 0.32, 0.32],
          [0.34, 0.33, 0.33]
        ]
      },
      explanation: "QK^T computes similarity scores. Divide by √d_k (=2) for stability. Softmax normalizes to probabilities."
    },
    {
      title: "5. Attention Output",
      description: "Multiply softmax weights by Values",
      data: {
        weights: "softmax output (3×3)",
        V: "Value matrix (3×4)",
        output: [
          [0.60, 0.50, 0.40, 0.47],
          [0.61, 0.51, 0.41, 0.46],
          [0.60, 0.50, 0.40, 0.47]
        ]
      },
      explanation: "Output = softmax(QK^T/√d_k) · V. Each position is now a weighted sum of all values."
    },
    {
      title: "6. Multi-Head Attention",
      description: "Run h=8 parallel attention heads",
      data: {
        heads: 8,
        d_k: "512/8 = 64 per head",
        process: "Concat(head_1, ..., head_8)·W_O",
        output_dim: "Back to d_model=512"
      },
      explanation: "Multiple heads let the model attend to different representation subspaces at different positions."
    },
    {
      title: "7. Add & Norm (Residual)",
      description: "LayerNorm(x + Sublayer(x))",
      data: {
        original_x: [0.10, 1.20, 0.30, 1.40],
        attention_out: [0.60, 0.50, 0.40, 0.47],
        sum: [0.70, 1.70, 0.70, 1.87],
        normalized: [-0.67, 0.87, -0.67, 1.13]
      },
      explanation: "Residual connection preserves original input. Layer normalization stabilizes training."
    },
    {
      title: "8. Feed-Forward Network",
      description: "FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2",
      data: {
        d_model: 512,
        d_ff: 2048,
        activation: "ReLU",
        structure: "512 → 2048 → 512"
      },
      explanation: "Two linear transformations with ReLU. Applied identically to each position. Another Add & Norm follows."
    },
    {
      title: "9. Encoder Stack",
      description: "N=6 identical encoder layers",
      data: {
        layers: 6,
        each_layer: ["Multi-Head Attention", "Add & Norm", "Feed-Forward", "Add & Norm"],
        output: "Encoded representations for all positions"
      },
      explanation: "Stack 6 encoder layers. Output of one layer becomes input to the next."
    },
    {
      title: "10. Decoder (with Masking)",
      description: "Similar structure + encoder-decoder attention",
      data: {
        components: [
          "Masked Multi-Head Self-Attention",
          "Multi-Head Cross-Attention (Q from decoder, K,V from encoder)",
          "Feed-Forward Network"
        ],
        masking: "Prevents attending to future tokens"
      },
      explanation: "Decoder attends to encoder output. Masking ensures autoregressive property during training."
    },
    {
      title: "11. Final Output",
      description: "Linear + Softmax for next token prediction",
      data: {
        linear: "Project to vocabulary size",
        vocab_size: 37000,
        softmax: "Probability distribution over all tokens",
        prediction: "argmax → next token"
      },
      explanation: "Final linear layer projects to vocab size. Softmax gives probability of each possible next token."
    }
  ];

  const currentStep = steps[step];

  const renderData = (data) => {
    return Object.entries(data).map(([key, value]) => (
      <div key={key} className="mb-2">
        <span className="font-semibold text-blue-600">{key}: </span>
        {Array.isArray(value) ? (
          <div className="ml-4 mt-1 font-mono text-xs bg-gray-100 p-2 rounded overflow-x-auto">
            {value.map((row, i) => (
              <div key={i}>
                [{Array.isArray(row) ? row.map(n => typeof n === 'number' ? n.toFixed(2) : n).join(', ') : `"${row}"`}]
              </div>
            ))}
          </div>
        ) : (
          <span className="font-mono text-sm">{typeof value === 'number' ? value : `${value}`}</span>
        )}
      </div>
    ));
  };

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-4">🔍 Attention Is All You Need</h1>
      
      <div className="mb-4 flex items-center justify-between">
        <button 
          onClick={() => setStep(Math.max(0, step - 1))}
          disabled={step === 0}
          className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
        >
          ← Previous
        </button>
        <span className="text-sm text-gray-600">Step {step + 1} of {steps.length}</span>
        <button 
          onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
          disabled={step === steps.length - 1}
          className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
        >
          Next →
        </button>
      </div>

      <div className="bg-white border rounded-lg shadow-sm">
        <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white p-3 rounded-t-lg">
          <h2 className="font-bold">{currentStep.title}</h2>
          <p className="text-sm opacity-90">{currentStep.description}</p>
        </div>
        
        <div className="p-4">
          <div className="mb-4">
            <h3 className="font-semibold mb-2 text-gray-700">📊 Example Data:</h3>
            <div className="text-sm">{renderData(currentStep.data)}</div>
          </div>
          
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3">
            <h3 className="font-semibold text-yellow-800 mb-1">💡 Explanation:</h3>
            <p className="text-sm text-yellow-900">{currentStep.explanation}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 flex justify-center gap-1">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`w-3 h-3 rounded-full ${i === step ? 'bg-blue-500' : 'bg-gray-300'}`}
          />
        ))}
      </div>

      <div className="mt-4 p-3 bg-gray-100 rounded text-xs">
        <strong>Test Input:</strong> "I love AI" → <strong>Target Output:</strong> "J'aime l'IA" (translation task)
      </div>
    </div>
  );
}
