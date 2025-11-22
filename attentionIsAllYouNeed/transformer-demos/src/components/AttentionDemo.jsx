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
