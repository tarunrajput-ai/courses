import React, { useState } from 'react';

export default function EmbeddingExplainer() {
  const [step, setStep] = useState(0);
  const [highlightRow, setHighlightRow] = useState(null);

  // Simulated vocabulary (in reality, 30k-50k tokens)
  const vocabulary = {
    "<PAD>": 0, "<UNK>": 1, "I": 2, "love": 3, "AI": 4, 
    "you": 5, "hate": 6, "machine": 7, "learning": 8, "deep": 9
  };

  // Embedding matrix (vocab_size=10 x d_model=4)
  const embeddingMatrix = [
    [0.00, 0.00, 0.00, 0.00],  // 0: <PAD>
    [0.12, 0.34, 0.56, 0.78],  // 1: <UNK>
    [0.10, 0.20, 0.30, 0.40],  // 2: I
    [0.50, 0.10, 0.80, 0.20],  // 3: love
    [0.90, 0.70, 0.20, 0.10],  // 4: AI
    [0.15, 0.25, 0.35, 0.45],  // 5: you
    [0.55, 0.15, 0.75, 0.25],  // 6: hate
    [0.85, 0.65, 0.25, 0.15],  // 7: machine
    [0.80, 0.60, 0.30, 0.20],  // 8: learning
    [0.75, 0.55, 0.35, 0.25],  // 9: deep
  ];

  const inputTokens = ["I", "love", "AI"];
  const tokenIds = [2, 3, 4];

  const steps = [
    {
      title: "Step 1: Build Vocabulary",
      content: (
        <div>
          <p className="mb-3 text-sm">Before training, create a mapping from every unique token to an integer ID:</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs">
            {Object.entries(vocabulary).map(([token, id]) => (
              <div key={token} className="flex justify-between border-b border-gray-200 py-1">
                <span>"{token}"</span>
                <span>→ {id}</span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-sm text-gray-600">
            Real models use 30,000-50,000+ tokens (words, subwords, characters).
          </p>
        </div>
      )
    },
    {
      title: "Step 2: Tokenize Input",
      content: (
        <div>
          <p className="mb-3 text-sm">Convert the input sentence to token IDs:</p>
          <div className="bg-blue-50 p-4 rounded mb-4">
            <div className="text-center font-bold mb-2">"I love AI"</div>
            <div className="flex justify-center items-center gap-2">
              <span className="text-2xl">↓</span>
            </div>
            <div className="flex justify-center gap-4 mt-2">
              {inputTokens.map((token, i) => (
                <div key={i} className="text-center">
                  <div className="bg-white border-2 border-blue-400 rounded px-3 py-2 font-mono">
                    "{token}"
                  </div>
                  <div className="text-2xl">↓</div>
                  <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center mx-auto font-bold">
                    {tokenIds[i]}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <p className="text-sm text-gray-600">
            Token IDs: <code className="bg-gray-100 px-2 py-1 rounded">[2, 3, 4]</code>
          </p>
        </div>
      )
    },
    {
      title: "Step 3: Initialize Embedding Matrix",
      content: (
        <div>
          <p className="mb-3 text-sm">Create a learnable matrix of shape <code>(vocab_size × d_model)</code>:</p>
          <div className="overflow-x-auto">
            <table className="text-xs font-mono w-full">
              <thead>
                <tr className="bg-gray-200">
                  <th className="p-2 text-left">ID</th>
                  <th className="p-2 text-left">Token</th>
                  <th className="p-2" colSpan={4}>Embedding Vector (d=4)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(vocabulary).map(([token, id]) => (
                  <tr 
                    key={id} 
                    className={`border-b ${tokenIds.includes(id) ? 'bg-yellow-100' : ''}`}
                    onMouseEnter={() => setHighlightRow(id)}
                    onMouseLeave={() => setHighlightRow(null)}
                  >
                    <td className="p-2 font-bold">{id}</td>
                    <td className="p-2">{token}</td>
                    {embeddingMatrix[id].map((val, i) => (
                      <td key={i} className="p-2 text-center">{val.toFixed(2)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm text-gray-600">
            Initially random values. These are <strong>learned during training</strong> via backpropagation.
          </p>
        </div>
      )
    },
    {
      title: "Step 4: Lookup (The Key Operation!)",
      content: (
        <div>
          <p className="mb-3 text-sm">For each token ID, simply <strong>look up the corresponding row</strong> in the embedding matrix:</p>
          <div className="space-y-4">
            {inputTokens.map((token, i) => (
              <div key={i} className="flex items-center gap-3 bg-gradient-to-r from-blue-50 to-green-50 p-3 rounded">
                <div className="text-center">
                  <div className="text-xs text-gray-500">Token</div>
                  <div className="font-bold">"{token}"</div>
                </div>
                <div className="text-xl">→</div>
                <div className="text-center">
                  <div className="text-xs text-gray-500">ID</div>
                  <div className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">
                    {tokenIds[i]}
                  </div>
                </div>
                <div className="text-xl">→</div>
                <div className="text-center">
                  <div className="text-xs text-gray-500">Matrix Row {tokenIds[i]}</div>
                  <div className="bg-green-500 text-white rounded px-2 py-1 font-mono text-xs">
                    [{embeddingMatrix[tokenIds[i]].map(v => v.toFixed(2)).join(', ')}]
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-yellow-50 border-l-4 border-yellow-400 text-sm">
            <strong>This is just indexing!</strong> No matrix multiplication. Token ID = row index.
          </div>
        </div>
      )
    },
    {
      title: "Step 5: Output Embedding Matrix",
      content: (
        <div>
          <p className="mb-3 text-sm">The result is a matrix of shape <code>(sequence_length × d_model)</code>:</p>
          <div className="bg-green-50 p-4 rounded mb-4">
            <div className="font-mono text-sm">
              <div className="text-gray-500 mb-2">Input: "I love AI" → IDs: [2, 3, 4]</div>
              <div className="text-gray-500 mb-2">Output Shape: (3 × 4)</div>
              <div className="border-2 border-green-500 rounded p-3 bg-white">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-gray-400 w-16">I:</span>
                  <span>[0.10, 0.20, 0.30, 0.40]</span>
                </div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-gray-400 w-16">love:</span>
                  <span>[0.50, 0.10, 0.80, 0.20]</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-400 w-16">AI:</span>
                  <span>[0.90, 0.70, 0.20, 0.10]</span>
                </div>
              </div>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            This embedded matrix is then passed to the Transformer encoder (after adding positional encoding).
          </p>
        </div>
      )
    },
    {
      title: "Step 6: How Embeddings Learn",
      content: (
        <div>
          <p className="mb-3 text-sm">The embedding matrix is trained via backpropagation:</p>
          <div className="space-y-3">
            <div className="bg-purple-50 p-3 rounded">
              <div className="font-semibold text-purple-700">1. Forward Pass</div>
              <p className="text-sm">Token embeddings flow through the model to produce predictions.</p>
            </div>
            <div className="bg-purple-50 p-3 rounded">
              <div className="font-semibold text-purple-700">2. Compute Loss</div>
              <p className="text-sm">Compare predictions to actual targets (e.g., next word).</p>
            </div>
            <div className="bg-purple-50 p-3 rounded">
              <div className="font-semibold text-purple-700">3. Backward Pass</div>
              <p className="text-sm">Gradients flow back to update embedding vectors.</p>
            </div>
            <div className="bg-purple-50 p-3 rounded">
              <div className="font-semibold text-purple-700">4. Result</div>
              <p className="text-sm">Similar words get similar vectors: "love" ≈ "like", "AI" ≈ "ML"</p>
            </div>
          </div>
          <div className="mt-4 p-3 bg-blue-50 rounded text-sm">
            <strong>Famous example:</strong> king - man + woman ≈ queen
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-2">🔢 Token → Dense Vector</h1>
      <p className="text-center text-sm text-gray-600 mb-4">How Embedding Lookup Works</p>
      
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
        <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white p-3 rounded-t-lg">
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
            className={`w-3 h-3 rounded-full ${i === step ? 'bg-purple-500' : 'bg-gray-300'}`}
          />
        ))}
      </div>
    </div>
  );
}
