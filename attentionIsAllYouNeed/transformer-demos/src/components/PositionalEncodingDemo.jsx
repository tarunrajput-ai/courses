import React, { useState, useMemo } from 'react';

export default function PositionalEncodingDemo() {
  const [maxLen, setMaxLen] = useState(20);
  const [dModel, setDModel] = useState(32);

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
      return `rgba(59, 130, 246, ${Math.abs(val)})`;
    } else {
      return `rgba(239, 68, 68, ${Math.abs(val)})`;
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-orange-700">2. Positional Encoding</h2>
      <p className="text-gray-600 mb-4">Add position information using sin/cos waves</p>

      <div className="flex gap-6 mb-4">
        <div>
          <label className="block text-sm font-semibold mb-1">Sequence Length: {maxLen}</label>
          <input 
            type="range" min="5" max="50" value={maxLen} 
            onChange={(e) => setMaxLen(parseInt(e.target.value))}
            className="w-40"
          />
        </div>
        <div>
          <label className="block text-sm font-semibold mb-1">d_model: {dModel}</label>
          <input 
            type="range" min="8" max="64" step="8" value={dModel} 
            onChange={(e) => setDModel(parseInt(e.target.value))}
            className="w-40"
          />
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-100 rounded font-mono text-sm">
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))<br/>
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
      </div>

      <div className="overflow-hidden rounded border">
        <div className="text-xs text-center bg-gray-200 p-1">
          PE Matrix ({maxLen} × {dModel}) — <span className="text-blue-600">Blue: positive</span>, <span className="text-red-600">Red: negative</span>
        </div>
        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
          {peMatrix.map((row, pos) => (
            <div key={pos} className="flex" style={{ height: '12px' }}>
              {row.map((val, dim) => (
                <div
                  key={dim}
                  style={{ flex: 1, backgroundColor: getColor(val), minWidth: '2px' }}
                  title={`pos=${pos}, dim=${dim}, val=${val.toFixed(3)}`}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-4 p-3 bg-yellow-50 rounded text-sm">
        <strong>Notice:</strong> Left columns oscillate fast (low dimensions), right columns oscillate slowly (high dimensions) — like a binary clock!
      </div>
    </div>
  );
}
