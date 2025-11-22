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
