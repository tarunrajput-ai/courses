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
