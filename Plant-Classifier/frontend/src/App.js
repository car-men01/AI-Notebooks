import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import Results from './components/Results';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  const handleResults = (data, imageUrl) => {
    setResults(data);
    setSelectedImage(imageUrl);
    setError(null);
  };

  const handleError = (err) => {
    setError(err);
    setResults(null);
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
    setSelectedImage(null);
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🌿 Plant Care Classifier</h1>
          <p className="subtitle">AI-Powered Plant Identification & Care Guide</p>
        </div>
      </header>

      <main className="App-main">
        {!results && !loading && (
          <ImageUpload
            onResults={handleResults}
            onError={handleError}
            onLoadingChange={setLoading}
          />
        )}

        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Analyzing your plant... This may take a moment.</p>
          </div>
        )}

        {error && (
          <div className="error-container">
            <h3>❌ Error</h3>
            <p>{error}</p>
            <button onClick={handleReset} className="btn btn-primary">
              Try Again
            </button>
          </div>
        )}

        {results && !loading && (
          <Results
            results={results}
            imageUrl={selectedImage}
            onReset={handleReset}
          />
        )}
      </main>

      <footer className="App-footer">
        <p>Powered by EfficientNet & GPT-4o-mini | Built with React & FastAPI</p>
      </footer>
    </div>
  );
}

export default App;
