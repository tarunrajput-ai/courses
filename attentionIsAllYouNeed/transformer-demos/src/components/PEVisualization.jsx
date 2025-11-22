import React, { useState, useMemo } from 'react';

export default function PEVisualization() {
  const [maxLen, setMaxLen] = useState(50);
  const [dModel, setDModel] = useState(64);
  const [showFormula, setShowFormula] = useState(true);

  const calcPE = (pos, i, d) => {
    const divisor = Math.pow(10000, (2 * Math.floor(i/2)) / d);
    const angle = pos / divisor;
    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  };

  const peMatrix = useMemo(() => {
    const matrix = [];
    for (let pos = 0; pos < maxLen; pos++) {
      const row = [];
      for (let i = 0; i < dModel; i++) {
        row.push(calcPE(pos, i, dModel));
      }
      matrix.push(row);
    }
    return matrix;
  }, [maxLen, dModel]);

  const getColor = (val) => {
    if (val >= 0) {
      return `rgb(${Math.floor(255 - val * 200)}, ${Math.floor(255 - val * 100)}, 255)`;
    } else {
      return `rgb(255, ${Math.floor(255 + val * 100)}, ${Math.floor(255 + val * 200)})`;
    }
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-xl font-bold text-center mb-2">📊 Positional Encoding Matrix</h1>
      <p className="text-center text-sm text-gray-600 mb-4">Fixed & Deterministic for any d_model</p>

      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 mb-4 text-sm">
        <strong>Key Point:</strong> This matrix is computed ONCE and reused. Same inputs = same outputs, always!
      </div>

      <div className="flex flex-wrap gap-4 mb-4 justify-center">
        <div>
          <label className="text-sm font-semibold block mb-1">max_seq_len: {maxLen}</label>
          <input 
            type="range" 
            min="10" 
            max="100" 
            value={maxLen} 
            onChange={(e) => setMaxLen(parseInt(e.target.value))}
            className="w-32"
          />
        </div>
        <div>
          <label className="text-sm font-semibold block mb-1">d_model: {dModel}</label>
          <input 
            type="range" 
            min="16" 
            max="128" 
            step="16"
            value={dModel} 
            onChange={(e) => setDModel(parseInt(e.target.value))}
            className="w-32"
          />
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-100 rounded text-sm font-mono text-center">
        PE Matrix Shape: <strong>({maxLen} × {dModel})</strong>
        <span className="text-gray-500 ml-2">= {(maxLen * dModel).toLocaleString()} values</span>
      </div>

      <div className="overflow-hidden rounded-lg border shadow-sm mb-4">
        <div className="overflow-x-auto">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="flex text-xs text-gray-500 bg-gray-100 sticky top-0">
              <div className="w-10 p-1 text-center font-semibold border-r">pos</div>
              <div className="flex-1 p-1 text-center">← dimensions (0 to {dModel - 1}) →</div>
            </div>
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {peMatrix.map((row, pos) => (
                <div key={pos} className="flex" style={{ height: '6px' }}>
                  <div className="w-10 text-xs text-gray-400 flex items-center justify-center border-r bg-gray-50">
                    {pos % 10 === 0 ? pos : ''}
                  </div>
                  <div className="flex flex-1">
                    {row.map((val, dim) => (
                      <div
                        key={dim}
                        style={{
                          flex: 1,
                          backgroundColor: getColor(val),
                          minWidth: '1px'
                        }}
                        title={`pos=${pos}, dim=${dim}, val=${val.toFixed(3)}`}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-center gap-4 text-sm mb-4">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgb(55, 155, 255)'}}></div>
          <span>+1 (sin/cos max)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-white border"></div>
          <span>0</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded" style={{backgroundColor: 'rgb(255, 155, 55)'}}></div>
          <span>-1 (sin/cos min)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-blue-50 p-3 rounded">
          <h3 className="font-semibold text-blue-700 mb-2">🔍 What You See:</h3>
          <ul className="text-sm space-y-1">
            <li>• <strong>Left columns (low dim):</strong> Fast oscillation</li>
            <li>• <strong>Right columns (high dim):</strong> Slow oscillation</li>
            <li>• <strong>Each row:</strong> Unique position signature</li>
            <li>• <strong>Patterns:</strong> Sine waves at different frequencies</li>
          </ul>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <h3 className="font-semibold text-green-700 mb-2">✅ Why Fixed is Good:</h3>
          <ul className="text-sm space-y-1">
            <li>• No training required</li>
            <li>• Zero extra parameters</li>
            <li>• Generalizes to any sequence length</li>
            <li>• Consistent across runs</li>
          </ul>
        </div>
      </div>

      <div className="mt-4 p-3 bg-purple-50 rounded">
        <h3 className="font-semibold text-purple-700 mb-2">🧮 The Fixed Formula:</h3>
        <div className="font-mono text-sm bg-white p-2 rounded">
          <p>PE[pos, 2i] = sin(pos / 10000^(2i/{dModel}))</p>
          <p>PE[pos, 2i+1] = cos(pos / 10000^(2i/{dModel}))</p>
        </div>
        <p className="text-sm mt-2 text-gray-600">
          Given pos and dimension i, the output is <strong>always the same</strong>. 
          No randomness, no learning, completely deterministic.
        </p>
      </div>
    </div>
  );
}
