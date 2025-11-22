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
