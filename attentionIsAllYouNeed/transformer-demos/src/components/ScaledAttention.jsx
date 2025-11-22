import React, { useState } from 'react';

export default function ScaledAttention() {
  const [step, setStep] = useState(0);

  // Our Q, K, V matrices (3 tokens × 4 dimensions)
  const Q = [
    [1.10, 0.82, 1.10, 0.89],  // "I"
    [1.18, 0.73, 0.98, 0.85],  // "love"
    [1.42, 1.12, 1.23, 1.35],  // "AI"
  ];

  const K = [
    [1.05, 0.95, 0.78, 1.12],  // "I"
    [1.02, 0.68, 0.72, 0.92],  // "love"
    [1.38, 1.28, 1.05, 1.42],  // "AI"
  ];

  const V = [
    [0.98, 0.88, 1.02, 0.75],  // "I"
    [0.85, 0.72, 0.95, 0.68],  // "love"
    [1.25, 1.15, 1.32, 1.08],  // "AI"
  ];

  const d_k = 4;
  const sqrt_dk = Math.sqrt(d_k);

  // Compute K transpose
  const K_T = K[0].map((_, colIndex) => K.map(row => row[colIndex]));

  // Compute QK^T
  const dotProduct = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);
  
  const QKt = Q.map(qRow => K.map(kRow => dotProduct(qRow, kRow)));

  // Scale by sqrt(d_k)
  const scaled = QKt.map(row => row.map(val => val / sqrt_dk));

  // Softmax function
  const softmax = (arr) => {
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  };

  const attention_weights = scaled.map(row => softmax(row));

  // Compute output: attention_weights × V
  const output = attention_weights.map(weightRow => 
    V[0].map((_, dim) => 
      weightRow.reduce((sum, w, i) => sum + w * V[i][dim], 0)
    )
  );

  const fmt = (n) => n.toFixed(2);
  const fmtRow = (row) => `[${row.map(fmt).join(', ')}]`;

  const tokens = ["I", "love", "AI"];

  const steps = [
    {
      title: "Overview: The Formula",
      content: (
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-blue-100 to-purple-100 p-4 rounded-lg text-center">
            <div className="font-mono text-lg font-bold">
              Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) · V
            </div>
          </div>
          <div className="grid grid-cols-4 gap-2 text-center text-sm">
            <div className="bg-blue-50 p-2 rounded">
              <div className="font-bold text-blue-700">Step 1</div>
              <div>QK<sup>T</sup></div>
              <div className="text-xs text-gray-500">Similarity scores</div>
            </div>
            <div className="bg-green-50 p-2 rounded">
              <div className="font-bold text-green-700">Step 2</div>
              <div>÷ √d<sub>k</sub></div>
              <div className="text-xs text-gray-500">Scale down</div>
            </div>
            <div className="bg-yellow-50 p-2 rounded">
              <div className="font-bold text-yellow-700">Step 3</div>
              <div>softmax</div>
              <div className="text-xs text-gray-500">Normalize</div>
            </div>
            <div className="bg-purple-50 p-2 rounded">
              <div className="font-bold text-purple-700">Step 4</div>
              <div>× V</div>
              <div className="text-xs text-gray-500">Weighted sum</div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Input: Q, K, V Matrices",
      content: (
        <div className="space-y-3">
          <p className="text-sm">From our previous projections (3 tokens × 4 dims):</p>
          <div className="grid grid-cols-3 gap-2 text-xs font-mono">
            <div className="bg-blue-50 p-2 rounded">
              <div className="font-bold text-blue-700 mb-1 text-center">Q (Query)</div>
              {Q.map((row, i) => (
                <div key={i} className="flex gap-1">
                  <span className="text-gray-400 w-10">{tokens[i]}:</span>
                  <span>{fmtRow(row)}</span>
                </div>
              ))}
            </div>
            <div className="bg-green-50 p-2 rounded">
              <div className="font-bold text-green-700 mb-1 text-center">K (Key)</div>
              {K.map((row, i) => (
                <div key={i} className="flex gap-1">
                  <span className="text-gray-400 w-10">{tokens[i]}:</span>
                  <span>{fmtRow(row)}</span>
                </div>
              ))}
            </div>
            <div className="bg-purple-50 p-2 rounded">
              <div className="font-bold text-purple-700 mb-1 text-center">V (Value)</div>
              {V.map((row, i) => (
                <div key={i} className="flex gap-1">
                  <span className="text-gray-400 w-10">{tokens[i]}:</span>
                  <span>{fmtRow(row)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="text-sm text-gray-600">
            d_k = {d_k} (dimension of keys), √d_k = {fmt(sqrt_dk)}
          </div>
        </div>
      )
    },
    {
      title: "Step 1a: Transpose K → K^T",
      content: (
        <div className="space-y-3">
          <p className="text-sm">First, transpose K to enable matrix multiplication:</p>
          <div className="flex items-center justify-center gap-4">
            <div className="bg-green-50 p-3 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">K (3×4)</div>
              {K.map((row, i) => <div key={i}>{fmtRow(row)}</div>)}
            </div>
            <div className="text-2xl">→</div>
            <div className="bg-green-100 p-3 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">K<sup>T</sup> (4×3)</div>
              {K_T.map((row, i) => <div key={i}>{fmtRow(row)}</div>)}
            </div>
          </div>
          <div className="bg-yellow-50 p-2 rounded text-sm text-center">
            Rows become columns, columns become rows
          </div>
        </div>
      )
    },
    {
      title: "Step 1b: Compute QK^T (Attention Scores)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Multiply Q × K<sup>T</sup> to get similarity scores:</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs text-center">
            Q(3×4) · K<sup>T</sup>(4×3) = QK<sup>T</sup>(3×3)
          </div>
          <div className="bg-blue-50 p-3 rounded">
            <div className="font-bold text-center mb-2">QK<sup>T</sup> = Attention Scores</div>
            <table className="w-full text-sm font-mono">
              <thead>
                <tr className="text-gray-500">
                  <th className="p-1"></th>
                  {tokens.map(t => <th key={t} className="p-1 text-green-700">K:{t}</th>)}
                </tr>
              </thead>
              <tbody>
                {QKt.map((row, i) => (
                  <tr key={i}>
                    <td className="p-1 text-blue-700 font-bold">Q:{tokens[i]}</td>
                    {row.map((val, j) => (
                      <td key={j} className={`p-1 text-center ${i === j ? 'bg-yellow-200' : ''}`}>
                        {fmt(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-xs text-gray-600">
            Each cell = how much Query[row] attends to Key[col]
          </div>
        </div>
      )
    },
    {
      title: "Step 1c: How QK^T[0][0] is Calculated",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Let's compute the attention score for "I" query → "I" key:</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs space-y-2">
            <div>Q["I"] = {fmtRow(Q[0])}</div>
            <div>K["I"] = {fmtRow(K[0])}</div>
            <div className="border-t pt-2 mt-2">
              <div className="font-bold">Dot Product:</div>
              <div className="ml-2">
                = ({fmt(Q[0][0])} × {fmt(K[0][0])}) + ({fmt(Q[0][1])} × {fmt(K[0][1])}) + ({fmt(Q[0][2])} × {fmt(K[0][2])}) + ({fmt(Q[0][3])} × {fmt(K[0][3])})
              </div>
              <div className="ml-2">
                = {fmt(Q[0][0] * K[0][0])} + {fmt(Q[0][1] * K[0][1])} + {fmt(Q[0][2] * K[0][2])} + {fmt(Q[0][3] * K[0][3])}
              </div>
              <div className="ml-2 font-bold text-blue-700">
                = {fmt(QKt[0][0])}
              </div>
            </div>
          </div>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-2 text-sm">
            <strong>Intuition:</strong> Higher dot product = vectors point in similar direction = more attention
          </div>
        </div>
      )
    },
    {
      title: "Step 2: Scale by √d_k",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Divide all scores by √d_k = √{d_k} = {fmt(sqrt_dk)}:</p>
          <div className="flex items-center justify-center gap-4">
            <div className="bg-blue-50 p-3 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">QK<sup>T</sup></div>
              {QKt.map((row, i) => <div key={i}>{fmtRow(row)}</div>)}
            </div>
            <div className="text-xl">÷ {fmt(sqrt_dk)} =</div>
            <div className="bg-green-100 p-3 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">Scaled</div>
              {scaled.map((row, i) => <div key={i}>{fmtRow(row)}</div>)}
            </div>
          </div>
          <div className="bg-red-50 border-l-4 border-red-400 p-3 text-sm">
            <strong>Why scale?</strong> Without scaling, large d_k causes dot products to grow large, pushing softmax into regions with tiny gradients (vanishing gradient problem).
          </div>
        </div>
      )
    },
    {
      title: "Why Scaling Matters (Visual)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Large values make softmax "peaky" (one-hot-like):</p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-red-50 p-3 rounded">
              <div className="font-bold text-red-700 mb-2 text-center">Without Scaling</div>
              <div className="text-xs font-mono mb-2">scores = [10, 11, 10.5]</div>
              <div className="flex gap-1 h-20 items-end">
                <div className="flex-1 bg-red-300 rounded-t" style={{height: '5%'}}></div>
                <div className="flex-1 bg-red-500 rounded-t" style={{height: '90%'}}></div>
                <div className="flex-1 bg-red-300 rounded-t" style={{height: '5%'}}></div>
              </div>
              <div className="text-xs text-center mt-1">[0.01, 0.98, 0.01]</div>
              <div className="text-xs text-red-600 text-center">Almost one-hot!</div>
            </div>
            <div className="bg-green-50 p-3 rounded">
              <div className="font-bold text-green-700 mb-2 text-center">With Scaling (÷√d_k)</div>
              <div className="text-xs font-mono mb-2">scores = [1.25, 1.38, 1.31]</div>
              <div className="flex gap-1 h-20 items-end">
                <div className="flex-1 bg-green-400 rounded-t" style={{height: '30%'}}></div>
                <div className="flex-1 bg-green-500 rounded-t" style={{height: '40%'}}></div>
                <div className="flex-1 bg-green-400 rounded-t" style={{height: '30%'}}></div>
              </div>
              <div className="text-xs text-center mt-1">[0.31, 0.38, 0.31]</div>
              <div className="text-xs text-green-600 text-center">Smoother distribution!</div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Step 3: Apply Softmax (Row-wise)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Apply softmax to each row (each query's attention distribution):</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs mb-3">
            softmax([a, b, c]) = [e<sup>a</sup>, e<sup>b</sup>, e<sup>c</sup>] / (e<sup>a</sup> + e<sup>b</sup> + e<sup>c</sup>)
          </div>
          <div className="bg-yellow-100 p-3 rounded">
            <div className="font-bold text-center mb-2">Attention Weights</div>
            <table className="w-full text-sm font-mono">
              <thead>
                <tr className="text-gray-500">
                  <th className="p-1"></th>
                  {tokens.map(t => <th key={t} className="p-1">→{t}</th>)}
                  <th className="p-1 text-gray-400">Sum</th>
                </tr>
              </thead>
              <tbody>
                {attention_weights.map((row, i) => (
                  <tr key={i}>
                    <td className="p-1 font-bold">{tokens[i]}</td>
                    {row.map((val, j) => (
                      <td key={j} className={`p-1 text-center ${val > 0.35 ? 'bg-yellow-300 font-bold' : ''}`}>
                        {fmt(val)}
                      </td>
                    ))}
                    <td className="p-1 text-center text-gray-400">{fmt(row.reduce((a,b) => a+b, 0))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-xs text-gray-600">
            Each row sums to 1.0 — it's a probability distribution!
          </div>
        </div>
      )
    },
    {
      title: "Step 3b: Softmax Calculation Example",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Let's calculate softmax for "I"'s attention (row 0):</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs space-y-2">
            <div>Scaled scores for "I": {fmtRow(scaled[0])}</div>
            <div className="border-t pt-2">
              <div>Step 1: Compute e^x for each:</div>
              <div className="ml-2">e^{fmt(scaled[0][0])} = {fmt(Math.exp(scaled[0][0]))}</div>
              <div className="ml-2">e^{fmt(scaled[0][1])} = {fmt(Math.exp(scaled[0][1]))}</div>
              <div className="ml-2">e^{fmt(scaled[0][2])} = {fmt(Math.exp(scaled[0][2]))}</div>
            </div>
            <div className="border-t pt-2">
              <div>Step 2: Sum = {fmt(Math.exp(scaled[0][0]) + Math.exp(scaled[0][1]) + Math.exp(scaled[0][2]))}</div>
            </div>
            <div className="border-t pt-2">
              <div>Step 3: Divide each by sum:</div>
              <div className="ml-2 font-bold text-blue-700">
                softmax = {fmtRow(attention_weights[0])}
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Step 4: Multiply by V (Weighted Sum)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Final step: Attention weights × V = Output</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs text-center mb-3">
            Weights(3×3) · V(3×4) = Output(3×4)
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="font-bold text-center mb-2">Output (Contextualized Embeddings)</div>
            <table className="w-full text-sm font-mono">
              <tbody>
                {output.map((row, i) => (
                  <tr key={i}>
                    <td className="p-1 font-bold text-purple-700">{tokens[i]}:</td>
                    <td className="p-1">{fmtRow(row)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-xs text-gray-600">
            Each output is a weighted combination of ALL Value vectors!
          </div>
        </div>
      )
    },
    {
      title: "Step 4b: Output Calculation Example",
      content: (
        <div className="space-y-3">
          <p className="text-sm">How Output["I"] is computed (weighted sum of all V's):</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-xs space-y-2">
            <div>Attention weights for "I": {fmtRow(attention_weights[0])}</div>
            <div className="border-t pt-2">
              <div className="font-bold">Output["I"] = w₀·V["I"] + w₁·V["love"] + w₂·V["AI"]</div>
              <div className="mt-2 ml-2">
                = {fmt(attention_weights[0][0])} × {fmtRow(V[0])}
              </div>
              <div className="ml-2">
                + {fmt(attention_weights[0][1])} × {fmtRow(V[1])}
              </div>
              <div className="ml-2">
                + {fmt(attention_weights[0][2])} × {fmtRow(V[2])}
              </div>
              <div className="mt-2 ml-2 font-bold text-purple-700">
                = {fmtRow(output[0])}
              </div>
            </div>
          </div>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-2 text-sm">
            <strong>Key insight:</strong> "I" now contains information from ALL tokens, weighted by relevance!
          </div>
        </div>
      )
    },
    {
      title: "Complete Picture",
      content: (
        <div className="space-y-3">
          <div className="bg-gradient-to-r from-blue-100 via-green-100 to-purple-100 p-3 rounded text-center font-mono text-sm">
            <div className="mb-2">Q(3×4) · K<sup>T</sup>(4×3) = (3×3)</div>
            <div className="mb-2">↓ ÷ √4 = 2</div>
            <div className="mb-2">↓ softmax (row-wise)</div>
            <div className="mb-2">↓ × V(3×4)</div>
            <div className="font-bold">Output(3×4)</div>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-blue-50 p-2 rounded">
              <strong>Input:</strong> Independent token embeddings
            </div>
            <div className="bg-purple-50 p-2 rounded">
              <strong>Output:</strong> Context-aware embeddings
            </div>
          </div>
          <div className="bg-green-50 border-l-4 border-green-400 p-3 text-sm">
            <strong>Magic:</strong> Each token now "sees" all other tokens, weighted by relevance. This is how Transformers capture long-range dependencies!
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-2">⚡ Scaled Dot-Product Attention</h1>
      <p className="text-center text-sm text-gray-600 mb-4">softmax(QK<sup>T</sup>/√d<sub>k</sub>)V — Step by Step</p>

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
        <div className="bg-gradient-to-r from-orange-500 to-pink-500 text-white p-3 rounded-t-lg">
          <h2 className="font-bold">{steps[step].title}</h2>
        </div>
        <div className="p-4">
          {steps[step].content}
        </div>
      </div>

      <div className="mt-4 flex justify-center gap-1 flex-wrap">
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
