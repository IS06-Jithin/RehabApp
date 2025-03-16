import React from 'react';

function App() {
  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Welcome to Rehab App</h1>
      <button onClick={() => alert('Starting exercise...')}>
        Start Exercise
      </button>
    </div>
  );
}

export default App;