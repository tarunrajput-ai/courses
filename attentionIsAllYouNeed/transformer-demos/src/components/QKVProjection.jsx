import React, { useState } from 'react';

export default function QKVProjection() {
  const [step, setStep] = useState(0);
  const [showMultiHead, setShowMultiHead] = useState(false);

  // Example with d_model=4 for simplicity
  const dModel = 4;
  const seqLen = 3;
  
  // Input embeddings (after positional encoding)
  const X = [
    [1.34, 0.64, 0.81, 1.20],  // "I"
    [1.81, 0.28, 0.22, 1.10],  // "love"
    [0.70, 1.70, 0.70, 1.87],  // "AI"
  ];

  // Weight matrices (learned parameters)
  const Wq = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.1, 0.2, 0.3],
    [0.2, 0.3, 0.1, 0.5],
    [0.4, 0.2, 0.5, 0.1],
  ];

  const Wk = [
    [0.3, 0.1, 0.4, 0.2],
    [0.2, 0.4, 0.1, 0.3],
    [0.1, 0.5, 0.3, 0.2],
    [0.5, 0.2, 0.2, 0.4],
  ];

  const Wv = [
    [0.2, 0.3, 0.1, 0.5],
    [0.4, 0.1, 0.5, 0.2],
    [0.3, 0.2, 0.4, 0.1],
    [0.1, 0.5, 0.2, 0.3],
  ];

  // Matrix multiplication helper
  const matMul = (A, B) => {
    const result = [];
    for (let i = 0; i < A.length; i++) {
      result[i] = [];
      for (let j = 0; j < B[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < B.length; k++) {
          sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  };

  const Q = matMul(X, Wq);
  const K = matMul(X, Wk);
  const V = matMul(X, Wv);

  const formatNum = (n) => n.toFixed(2);
  const formatRow = (row) => `[${row.map(formatNum).join(', ')}]`;

  const steps = [
    {
      title: "Why Do We Need Q, K, V?",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Attention needs three different "views" of each token:</p>
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-blue-50 p-3 rounded text-center">
              <div className="text-2xl mb-1">🔍</div>
              <div className="font-bold text-blue-700">Query (Q)</div>
              <p className="text-xs mt-1">"What am I looking for?"</p>
            </div>
            <div className="bg-green-50 p-3 rounded text-center">
              <div className="text-2xl mb-1">🔑</div>
              <div className="font-bold text-green-700">Key (K)</div>
              <p className="text-xs mt-1">"What do I contain?"</p>
            </div>
            <div className="bg-purple-50 p-3 rounded text-center">
              <div className="text-2xl mb-1">📦</div>
              <div className="font-bold text-purple-700">Value (V)</div>
              <p className="text-xs mt-1">"What info do I give?"</p>
            </div>
          </div>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 text-sm">
            <strong>Analogy:</strong> Like a search engine — Query is your search, Keys are page titles, Values are page content.
          </div>
        </div>
      )
    },
    {
      title: "The Input: X (Embedded + Position)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Our input matrix X has shape <strong>(seq_len × d_model) = (3 × 4)</strong>:</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-sm">
            <div className="text-gray-500 mb-2">X = Embedding + Positional Encoding</div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="w-12 text-gray-500">"I":</span>
                <span>{formatRow(X[0])}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-12 text-gray-500">"love":</span>
                <span>{formatRow(X[1])}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-12 text-gray-500">"AI":</span>
                <span>{formatRow(X[2])}</span>
              </div>
            </div>
          </div>
          <div className="text-sm bg-blue-50 p-2 rounded">
            Shape: <strong>(3 tokens × 4 dimensions)</strong>
          </div>
        </div>
      )
    },
    {
      title: "Weight Matrices: Wq, Wk, Wv",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Three separate <strong>learnable</strong> weight matrices, each (d_model × d_model):</p>
          <div className="grid grid-cols-3 gap-2 text-xs font-mono">
            <div className="bg-blue-50 p-2 rounded">
              <div className="font-bold text-blue-700 mb-1 text-center">Wq (4×4)</div>
              {Wq.map((row, i) => (
                <div key={i}>{formatRow(row)}</div>
              ))}
            </div>
            <div className="bg-green-50 p-2 rounded">
              <div className="font-bold text-green-700 mb-1 text-center">Wk (4×4)</div>
              {Wk.map((row, i) => (
                <div key={i}>{formatRow(row)}</div>
              ))}
            </div>
            <div className="bg-purple-50 p-2 rounded">
              <div className="font-bold text-purple-700 mb-1 text-center">Wv (4×4)</div>
              {Wv.map((row, i) => (
                <div key={i}>{formatRow(row)}</div>
              ))}
            </div>
          </div>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 text-sm">
            <strong>Why 4×4?</strong> To keep output dimension = input dimension (d_model → d_model)
          </div>
        </div>
      )
    },
    {
      title: "The Linear Projection Formula",
      content: (
        <div className="space-y-4">
          <div className="bg-gray-100 p-4 rounded text-center font-mono">
            <div className="text-lg font-bold mb-2">Q = X · Wq</div>
            <div className="text-lg font-bold mb-2">K = X · Wk</div>
            <div className="text-lg font-bold">V = X · Wv</div>
          </div>
          <div className="text-sm space-y-2">
            <p><strong>Matrix multiplication:</strong></p>
            <div className="font-mono bg-white p-2 rounded border text-center">
              (3 × 4) · (4 × 4) = (3 × 4)
            </div>
            <div className="font-mono bg-white p-2 rounded border text-center text-gray-500">
              (seq_len × d_model) · (d_model × d_model) = (seq_len × d_model)
            </div>
          </div>
          <div className="bg-green-50 p-3 rounded text-sm">
            <strong>Result:</strong> Each token gets transformed into Q, K, V vectors of the <strong>same dimension</strong>!
          </div>
        </div>
      )
    },
    {
      title: "Step-by-Step: Computing Q[0] (for \"I\")",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Let's compute the Query vector for the first token "I":</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs overflow-x-auto">
            <div className="mb-2">X[0] = [{X[0].map(formatNum).join(', ')}]</div>
            <div className="mb-2">
              <div>Q[0] = X[0] · Wq</div>
              <div className="mt-2 ml-4">
                <div>Q[0][0] = {X[0].map((x, i) => `${formatNum(x)}×${formatNum(Wq[i][0])}`).join(' + ')}</div>
                <div className="ml-8">= {formatNum(X[0].reduce((sum, x, i) => sum + x * Wq[i][0], 0))}</div>
              </div>
              <div className="mt-1 ml-4">
                <div>Q[0][1] = {X[0].map((x, i) => `${formatNum(x)}×${formatNum(Wq[i][1])}`).join(' + ')}</div>
                <div className="ml-8">= {formatNum(X[0].reduce((sum, x, i) => sum + x * Wq[i][1], 0))}</div>
              </div>
              <div className="mt-1 ml-4 text-gray-500">... (same for dimensions 2, 3)</div>
            </div>
          </div>
          <div className="bg-blue-100 p-2 rounded font-mono text-sm">
            Q[0] = {formatRow(Q[0])}
          </div>
        </div>
      )
    },
    {
      title: "Complete Q, K, V Matrices",
      content: (
        <div className="space-y-3">
          <p className="text-sm">After projecting all tokens:</p>
          <div className="grid grid-cols-3 gap-2 text-xs font-mono">
            <div className="bg-blue-100 p-2 rounded">
              <div className="font-bold text-blue-700 mb-1 text-center">Q = X·Wq</div>
              {Q.map((row, i) => (
                <div key={i} className="flex items-center gap-1">
                  <span className="text-gray-500 w-8">{["I", "love", "AI"][i]}:</span>
                  <span>{formatRow(row)}</span>
                </div>
              ))}
            </div>
            <div className="bg-green-100 p-2 rounded">
              <div className="font-bold text-green-700 mb-1 text-center">K = X·Wk</div>
              {K.map((row, i) => (
                <div key={i} className="flex items-center gap-1">
                  <span className="text-gray-500 w-8">{["I", "love", "AI"][i]}:</span>
                  <span>{formatRow(row)}</span>
                </div>
              ))}
            </div>
            <div className="bg-purple-100 p-2 rounded">
              <div className="font-bold text-purple-700 mb-1 text-center">V = X·Wv</div>
              {V.map((row, i) => (
                <div key={i} className="flex items-center gap-1">
                  <span className="text-gray-500 w-8">{["I", "love", "AI"][i]}:</span>
                  <span>{formatRow(row)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="text-sm text-gray-600">
            All three matrices have shape <strong>(3 × 4)</strong> — same as input X
          </div>
        </div>
      )
    },
    {
      title: "Why Different W Matrices?",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Each weight matrix learns a <strong>different transformation</strong>:</p>
          <div className="space-y-2">
            <div className="bg-blue-50 p-3 rounded">
              <div className="font-semibold text-blue-700">Wq learns: "What to search for"</div>
              <p className="text-xs mt-1">Transforms token into a query that will match relevant keys</p>
            </div>
            <div className="bg-green-50 p-3 rounded">
              <div className="font-semibold text-green-700">Wk learns: "What I'm about"</div>
              <p className="text-xs mt-1">Transforms token into a key that can be matched by queries</p>
            </div>
            <div className="bg-purple-50 p-3 rounded">
              <div className="font-semibold text-purple-700">Wv learns: "What info to pass"</div>
              <p className="text-xs mt-1">Transforms token into the actual information to aggregate</p>
            </div>
          </div>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 text-sm">
            <strong>If we used the same W:</strong> Q, K, V would be identical — attention couldn't learn different relationships!
          </div>
        </div>
      )
    },
    {
      title: "Real Dimensions (Original Transformer)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">In the actual paper:</p>
          <div className="overflow-x-auto">
            <table className="text-sm w-full">
              <thead className="bg-gray-200">
                <tr>
                  <th className="p-2 text-left">Parameter</th>
                  <th className="p-2 text-left">Our Example</th>
                  <th className="p-2 text-left">Original Paper</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-2">d_model</td>
                  <td className="p-2 font-mono">4</td>
                  <td className="p-2 font-mono">512</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Wq, Wk, Wv shape</td>
                  <td className="p-2 font-mono">4 × 4</td>
                  <td className="p-2 font-mono">512 × 512</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Parameters per W</td>
                  <td className="p-2 font-mono">16</td>
                  <td className="p-2 font-mono">262,144</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Total Q,K,V params</td>
                  <td className="p-2 font-mono">48</td>
                  <td className="p-2 font-mono">786,432</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-blue-50 p-3 rounded text-sm">
            <strong>Note:</strong> With multi-head attention (h=8), each head uses d_k = 512/8 = 64 dimensions
          </div>
        </div>
      )
    },
    {
      title: "Summary: The Full Picture",
      content: (
        <div className="space-y-3">
          <div className="bg-gray-100 p-3 rounded font-mono text-xs text-center">
            <div className="mb-2">Input X: (seq_len × d_model)</div>
            <div className="text-2xl mb-2">↓ × Wq, × Wk, × Wv</div>
            <div className="flex justify-center gap-4">
              <div className="bg-blue-200 px-2 py-1 rounded">Q</div>
              <div className="bg-green-200 px-2 py-1 rounded">K</div>
              <div className="bg-purple-200 px-2 py-1 rounded">V</div>
            </div>
            <div className="mt-2 text-gray-500">All: (seq_len × d_model)</div>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-green-50 p-2 rounded">
              <strong>✓ Learnable:</strong> Wq, Wk, Wv are trained via backprop
            </div>
            <div className="bg-green-50 p-2 rounded">
              <strong>✓ Dimension preserved:</strong> d_model in = d_model out
            </div>
            <div className="bg-green-50 p-2 rounded">
              <strong>✓ Per-token:</strong> Each token gets its own Q, K, V
            </div>
            <div className="bg-green-50 p-2 rounded">
              <strong>✓ Different views:</strong> Q ≠ K ≠ V enables rich attention
            </div>
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-2">🔄 Q, K, V Linear Projections</h1>
      <p className="text-center text-sm text-gray-600 mb-4">How tokens become Query, Key, Value vectors</p>

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
        <div className="bg-gradient-to-r from-blue-500 via-green-500 to-purple-500 text-white p-3 rounded-t-lg">
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
            className={`w-3 h-3 rounded-full ${i === step ? 'bg-blue-500' : 'bg-gray-300'}`}
          />
        ))}
      </div>
    </div>
  );
}
