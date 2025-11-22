import React, { useState } from 'react';

export default function MultiHeadAttention() {
  const [step, setStep] = useState(0);
  const [activeHead, setActiveHead] = useState(0);

  const d_model = 8;  // Simplified (paper uses 512)
  const num_heads = 2; // Simplified (paper uses 8)
  const d_k = d_model / num_heads; // 8/2 = 4 per head

  const tokens = ["I", "love", "AI"];
  const seq_len = 3;

  // Input X (3 tokens × 8 dims)
  const X = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  ];

  // Simulated head outputs (after each head's attention)
  const head1_output = [
    [0.45, 0.52, 0.38, 0.61],
    [0.48, 0.55, 0.41, 0.64],
    [0.51, 0.58, 0.44, 0.67],
  ];

  const head2_output = [
    [0.72, 0.35, 0.68, 0.42],
    [0.75, 0.38, 0.71, 0.45],
    [0.78, 0.41, 0.74, 0.48],
  ];

  // Concatenated (3 × 8)
  const concat_output = head1_output.map((row, i) => [...row, ...head2_output[i]]);

  // Final output after W_O projection
  const final_output = [
    [0.58, 0.62, 0.55, 0.68, 0.61, 0.54, 0.67, 0.59],
    [0.61, 0.65, 0.58, 0.71, 0.64, 0.57, 0.70, 0.62],
    [0.64, 0.68, 0.61, 0.74, 0.67, 0.60, 0.73, 0.65],
  ];

  const fmt = (n) => n.toFixed(2);
  const fmtRow = (row) => `[${row.map(fmt).join(', ')}]`;

  // Different attention patterns for each head
  const head1_attention = [
    [0.45, 0.30, 0.25],  // "I" attends to...
    [0.35, 0.40, 0.25],  // "love" attends to...
    [0.30, 0.25, 0.45],  // "AI" attends to...
  ];

  const head2_attention = [
    [0.20, 0.50, 0.30],  // Different pattern!
    [0.25, 0.25, 0.50],
    [0.40, 0.35, 0.25],
  ];

  const steps = [
    {
      title: "Why Multi-Head Attention?",
      content: (
        <div className="space-y-3">
          <div className="bg-red-50 border-l-4 border-red-400 p-3">
            <p className="font-semibold text-red-700">Problem with Single-Head:</p>
            <p className="text-sm">One attention head can only focus on ONE type of relationship at a time.</p>
          </div>
          <div className="bg-gray-100 p-3 rounded text-sm">
            <p className="font-mono">"The cat sat on the mat because it was tired"</p>
            <p className="mt-2">What does "it" refer to? We need to track:</p>
            <ul className="ml-4 mt-1 space-y-1">
              <li>• <strong>Syntactic:</strong> "it" → "cat" (subject reference)</li>
              <li>• <strong>Semantic:</strong> "tired" → "cat" (who is tired?)</li>
              <li>• <strong>Positional:</strong> nearby words</li>
            </ul>
          </div>
          <div className="bg-green-50 border-l-4 border-green-400 p-3">
            <p className="font-semibold text-green-700">Solution: Multiple Heads!</p>
            <p className="text-sm">Run several attention operations in parallel, each learning different patterns.</p>
          </div>
        </div>
      )
    },
    {
      title: "The Multi-Head Formula",
      content: (
        <div className="space-y-4">
          <div className="bg-blue-50 p-4 rounded-lg font-mono text-sm">
            <div className="font-bold mb-2">MultiHead(Q, K, V) = Concat(head₁, ..., head_h) · W<sup>O</sup></div>
            <div className="text-gray-600 mt-2">where head_i = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)</div>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-purple-50 p-3 rounded">
              <div className="font-bold text-purple-700">Original Paper:</div>
              <div className="font-mono text-xs mt-1">
                <div>d_model = 512</div>
                <div>h (heads) = 8</div>
                <div>d_k = d_v = 512/8 = 64</div>
              </div>
            </div>
            <div className="bg-orange-50 p-3 rounded">
              <div className="font-bold text-orange-700">Our Example:</div>
              <div className="font-mono text-xs mt-1">
                <div>d_model = 8</div>
                <div>h (heads) = 2</div>
                <div>d_k = d_v = 8/2 = 4</div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Step 1: Split into Heads",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Each head gets a <strong>smaller slice</strong> of dimensions:</p>
          <div className="bg-gray-100 p-3 rounded">
            <div className="font-mono text-xs mb-2">
              d_model = {d_model}, num_heads = {num_heads}, d_k = {d_model}/{num_heads} = {d_k}
            </div>
            <div className="flex items-center justify-center gap-2 my-3">
              <div className="bg-blue-200 px-3 py-6 rounded text-center">
                <div className="text-xs text-gray-600">Input</div>
                <div className="font-bold">{d_model} dims</div>
              </div>
              <div className="text-2xl">→</div>
              <div className="flex gap-2">
                <div className="bg-green-200 px-3 py-3 rounded text-center">
                  <div className="text-xs">Head 1</div>
                  <div className="font-bold">{d_k} dims</div>
                </div>
                <div className="bg-yellow-200 px-3 py-3 rounded text-center">
                  <div className="text-xs">Head 2</div>
                  <div className="font-bold">{d_k} dims</div>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-2 text-sm">
            <strong>Key insight:</strong> Total computation stays the same! We just split it into parallel smaller pieces.
          </div>
        </div>
      )
    },
    {
      title: "Step 2: Separate W Matrices per Head",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Each head has its <strong>own learned projections</strong>:</p>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead className="bg-gray-200">
                <tr>
                  <th className="p-2">Matrix</th>
                  <th className="p-2">Single-Head</th>
                  <th className="p-2">Multi-Head (per head)</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-2 font-mono">W<sup>Q</sup></td>
                  <td className="p-2 font-mono">(512 × 512)</td>
                  <td className="p-2 font-mono">(512 × 64) × 8 heads</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2 font-mono">W<sup>K</sup></td>
                  <td className="p-2 font-mono">(512 × 512)</td>
                  <td className="p-2 font-mono">(512 × 64) × 8 heads</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2 font-mono">W<sup>V</sup></td>
                  <td className="p-2 font-mono">(512 × 512)</td>
                  <td className="p-2 font-mono">(512 × 64) × 8 heads</td>
                </tr>
                <tr>
                  <td className="p-2 font-mono">W<sup>O</sup></td>
                  <td className="p-2 font-mono">N/A</td>
                  <td className="p-2 font-mono">(512 × 512)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-blue-50 p-2 rounded text-xs">
            <strong>Our example:</strong> W<sub>1</sub><sup>Q</sup>: (8×4), W<sub>2</sub><sup>Q</sup>: (8×4), etc.
          </div>
        </div>
      )
    },
    {
      title: "Step 3: Each Head Computes Attention",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Each head runs the full attention formula independently:</p>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-green-50 p-3 rounded">
              <div className="font-bold text-green-700 text-center mb-2">Head 1</div>
              <div className="font-mono text-xs space-y-1">
                <div>Q₁ = X · W₁<sup>Q</sup></div>
                <div>K₁ = X · W₁<sup>K</sup></div>
                <div>V₁ = X · W₁<sup>V</sup></div>
                <div className="border-t pt-1 mt-1">
                  head₁ = softmax(Q₁K₁<sup>T</sup>/√d_k)V₁
                </div>
                <div className="text-gray-500">Output: (3 × 4)</div>
              </div>
            </div>
            <div className="bg-yellow-50 p-3 rounded">
              <div className="font-bold text-yellow-700 text-center mb-2">Head 2</div>
              <div className="font-mono text-xs space-y-1">
                <div>Q₂ = X · W₂<sup>Q</sup></div>
                <div>K₂ = X · W₂<sup>K</sup></div>
                <div>V₂ = X · W₂<sup>V</sup></div>
                <div className="border-t pt-1 mt-1">
                  head₂ = softmax(Q₂K₂<sup>T</sup>/√d_k)V₂
                </div>
                <div className="text-gray-500">Output: (3 × 4)</div>
              </div>
            </div>
          </div>
          <div className="text-xs text-center text-gray-600">
            Both run in <strong>parallel</strong> on GPU!
          </div>
        </div>
      )
    },
    {
      title: "Each Head Learns Different Patterns!",
      content: (
        <div className="space-y-3">
          <p className="text-sm">This is the magic — each head attends to different things:</p>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-green-50 p-3 rounded">
              <div className="font-bold text-green-700 text-center mb-2">Head 1: Syntactic?</div>
              <table className="text-xs font-mono w-full">
                <thead>
                  <tr className="text-gray-500">
                    <th></th>
                    {tokens.map(t => <th key={t} className="p-1">{t}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {head1_attention.map((row, i) => (
                    <tr key={i}>
                      <td className="font-bold">{tokens[i]}</td>
                      {row.map((v, j) => (
                        <td key={j} className={`p-1 text-center ${v >= 0.4 ? 'bg-green-300 font-bold' : ''}`}>
                          {fmt(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-xs mt-2 text-gray-600">Focuses on self & adjacent</div>
            </div>
            <div className="bg-yellow-50 p-3 rounded">
              <div className="font-bold text-yellow-700 text-center mb-2">Head 2: Semantic?</div>
              <table className="text-xs font-mono w-full">
                <thead>
                  <tr className="text-gray-500">
                    <th></th>
                    {tokens.map(t => <th key={t} className="p-1">{t}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {head2_attention.map((row, i) => (
                    <tr key={i}>
                      <td className="font-bold">{tokens[i]}</td>
                      {row.map((v, j) => (
                        <td key={j} className={`p-1 text-center ${v >= 0.4 ? 'bg-yellow-300 font-bold' : ''}`}>
                          {fmt(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-xs mt-2 text-gray-600">Focuses on "love" and "AI"</div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Step 4: Concatenate All Heads",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Stack head outputs side by side:</p>
          <div className="flex items-center justify-center gap-2 flex-wrap">
            <div className="bg-green-100 p-2 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">Head 1 (3×4)</div>
              {head1_output.map((row, i) => (
                <div key={i}>{fmtRow(row)}</div>
              ))}
            </div>
            <div className="text-xl">+</div>
            <div className="bg-yellow-100 p-2 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">Head 2 (3×4)</div>
              {head2_output.map((row, i) => (
                <div key={i}>{fmtRow(row)}</div>
              ))}
            </div>
            <div className="text-xl">=</div>
            <div className="bg-purple-100 p-2 rounded font-mono text-xs">
              <div className="font-bold text-center mb-1">Concat (3×8)</div>
              {concat_output.map((row, i) => (
                <div key={i}>{fmtRow(row)}</div>
              ))}
            </div>
          </div>
          <div className="bg-blue-50 p-2 rounded text-sm text-center">
            Concat(head₁, head₂) → Back to d_model = 8 dimensions!
          </div>
        </div>
      )
    },
    {
      title: "Step 5: Project with W^O",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Final linear projection mixes information from all heads:</p>
          <div className="bg-gray-100 p-3 rounded font-mono text-sm text-center">
            Output = Concat(head₁, head₂) · W<sup>O</sup>
            <div className="text-xs text-gray-500 mt-1">
              (3 × 8) · (8 × 8) = (3 × 8)
            </div>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="font-bold text-center mb-2">Final Output (3 × 8)</div>
            <div className="font-mono text-xs">
              {final_output.map((row, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-gray-500 w-12">{tokens[i]}:</span>
                  <span>{fmtRow(row)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-green-50 border-l-4 border-green-400 p-2 text-sm">
            <strong>W<sup>O</sup> learns</strong> how to combine insights from different heads into a unified representation.
          </div>
        </div>
      )
    },
    {
      title: "Visual: Complete Multi-Head Flow",
      content: (
        <div className="space-y-2">
          <div className="bg-gray-100 p-3 rounded text-xs font-mono">
            <div className="text-center mb-2 font-bold">Input X: (seq_len × d_model) = (3 × 8)</div>
            <div className="flex justify-center">
              <div className="text-center">↓ Split into heads</div>
            </div>
            <div className="flex justify-center gap-4 my-2">
              <div className="bg-green-200 p-2 rounded text-center">
                <div>Head 1</div>
                <div className="text-gray-600">d_k=4</div>
                <div>↓</div>
                <div>Attention</div>
                <div>↓</div>
                <div>(3×4)</div>
              </div>
              <div className="bg-yellow-200 p-2 rounded text-center">
                <div>Head 2</div>
                <div className="text-gray-600">d_k=4</div>
                <div>↓</div>
                <div>Attention</div>
                <div>↓</div>
                <div>(3×4)</div>
              </div>
            </div>
            <div className="flex justify-center">
              <div className="text-center">↓ Concatenate</div>
            </div>
            <div className="flex justify-center my-2">
              <div className="bg-purple-200 p-2 rounded text-center">
                Concat: (3 × 8)
              </div>
            </div>
            <div className="flex justify-center">
              <div className="text-center">↓ × W<sup>O</sup> (8×8)</div>
            </div>
            <div className="flex justify-center my-2">
              <div className="bg-blue-200 p-2 rounded text-center font-bold">
                Output: (3 × 8)
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Real Paper Dimensions",
      content: (
        <div className="space-y-3">
          <p className="text-sm">In "Attention Is All You Need":</p>
          <div className="overflow-x-auto">
            <table className="text-sm w-full">
              <thead className="bg-gray-200">
                <tr>
                  <th className="p-2 text-left">Parameter</th>
                  <th className="p-2">Our Example</th>
                  <th className="p-2">Original Paper</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-2">d_model</td>
                  <td className="p-2 text-center font-mono">8</td>
                  <td className="p-2 text-center font-mono">512</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">num_heads (h)</td>
                  <td className="p-2 text-center font-mono">2</td>
                  <td className="p-2 text-center font-mono">8</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">d_k = d_v</td>
                  <td className="p-2 text-center font-mono">8/2 = 4</td>
                  <td className="p-2 text-center font-mono">512/8 = 64</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> (per head)</td>
                  <td className="p-2 text-center font-mono">(8 × 4)</td>
                  <td className="p-2 text-center font-mono">(512 × 64)</td>
                </tr>
                <tr>
                  <td className="p-2">W<sup>O</sup></td>
                  <td className="p-2 text-center font-mono">(8 × 8)</td>
                  <td className="p-2 text-center font-mono">(512 × 512)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )
    },
    {
      title: "What Each Head Learns (Research)",
      content: (
        <div className="space-y-3">
          <p className="text-sm">Studies show different heads specialize in different patterns:</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-blue-50 p-2 rounded">
              <div className="font-bold text-blue-700">Head A: Position</div>
              <p>Attends to previous/next token</p>
            </div>
            <div className="bg-green-50 p-2 rounded">
              <div className="font-bold text-green-700">Head B: Syntax</div>
              <p>Subject-verb agreement</p>
            </div>
            <div className="bg-yellow-50 p-2 rounded">
              <div className="font-bold text-yellow-700">Head C: Coreference</div>
              <p>"it" → "the cat"</p>
            </div>
            <div className="bg-purple-50 p-2 rounded">
              <div className="font-bold text-purple-700">Head D: Rare words</div>
              <p>Attends to unusual tokens</p>
            </div>
            <div className="bg-pink-50 p-2 rounded">
              <div className="font-bold text-pink-700">Head E: Punctuation</div>
              <p>Sentence boundaries</p>
            </div>
            <div className="bg-orange-50 p-2 rounded">
              <div className="font-bold text-orange-700">Head F: Long-range</div>
              <p>Distant dependencies</p>
            </div>
          </div>
          <div className="text-xs text-gray-600 text-center">
            This emerges automatically during training!
          </div>
        </div>
      )
    },
    {
      title: "Summary: Why Multi-Head Works",
      content: (
        <div className="space-y-3">
          <div className="grid grid-cols-1 gap-2">
            <div className="bg-green-50 p-3 rounded flex items-start gap-2">
              <span className="text-green-500 font-bold">✓</span>
              <div>
                <div className="font-semibold">Multiple representation subspaces</div>
                <div className="text-xs text-gray-600">Each head learns different features</div>
              </div>
            </div>
            <div className="bg-green-50 p-3 rounded flex items-start gap-2">
              <span className="text-green-500 font-bold">✓</span>
              <div>
                <div className="font-semibold">Same computational cost</div>
                <div className="text-xs text-gray-600">d_k × h = d_model (split, not duplicate)</div>
              </div>
            </div>
            <div className="bg-green-50 p-3 rounded flex items-start gap-2">
              <span className="text-green-500 font-bold">✓</span>
              <div>
                <div className="font-semibold">Parallel computation</div>
                <div className="text-xs text-gray-600">All heads run simultaneously on GPU</div>
              </div>
            </div>
            <div className="bg-green-50 p-3 rounded flex items-start gap-2">
              <span className="text-green-500 font-bold">✓</span>
              <div>
                <div className="font-semibold">Richer attention patterns</div>
                <div className="text-xs text-gray-600">Capture syntax, semantics, position, etc.</div>
              </div>
            </div>
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-2">🎯 Multi-Head Attention</h1>
      <p className="text-center text-sm text-gray-600 mb-4">Multiple parallel attention heads</p>

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
        <div className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white p-3 rounded-t-lg">
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
            className={`w-3 h-3 rounded-full ${i === step ? 'bg-indigo-500' : 'bg-gray-300'}`}
          />
        ))}
      </div>
    </div>
  );
}
