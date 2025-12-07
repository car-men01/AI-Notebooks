import React, { useState, useRef } from 'react';
import { plantApi } from '../services/api';
import './ImageUpload.css';

function ImageUpload({ onResults, onError, onLoadingChange }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState('rag'); // 'basic', 'full', 'rag'
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        onError('Please select an image file');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      onError('Please select an image first');
      return;
    }

    onLoadingChange(true);
    onError(null);

    try {
      let data;
      if (mode === 'basic') {
        data = await plantApi.classifyPlant(selectedFile);
      } else if (mode === 'full') {
        data = await plantApi.getPlantCare(selectedFile);
      } else if (mode === 'rag') {
        data = await plantApi.getPlantCareRAG(selectedFile);
      }
      
      onResults(data, previewUrl);
    } catch (err) {
      onError(err.response?.data?.detail || err.message || 'Failed to analyze image');
    } finally {
      onLoadingChange(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-card">
        <h2>Upload Plant Image</h2>
        
        <div
          className="drop-zone"
          onClick={() => fileInputRef.current.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="preview-image" />
          ) : (
            <div className="drop-zone-content">
              <span className="upload-icon">📷</span>
              <p>Click or drag image here</p>
              <span className="file-types">JPG, PNG, JPEG</span>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>

        {selectedFile && (
          <div className="selected-file">
            <span>✓ {selectedFile.name}</span>
            <button
              onClick={() => {
                setSelectedFile(null);
                setPreviewUrl(null);
              }}
              className="btn-text"
            >
              Remove
            </button>
          </div>
        )}

        <div className="mode-selector">
          <h3>Analysis Mode</h3>
          <div className="mode-options">
            <label className={`mode-option ${mode === 'basic' ? 'selected' : ''}`}>
              <input
                type="radio"
                value="basic"
                checked={mode === 'basic'}
                onChange={(e) => setMode(e.target.value)}
              />
              <div className="mode-content">
                <strong>Basic</strong>
                <span>Classification only</span>
              </div>
            </label>

            <label className={`mode-option ${mode === 'full' ? 'selected' : ''}`}>
              <input
                type="radio"
                value="full"
                checked={mode === 'full'}
                onChange={(e) => setMode(e.target.value)}
              />
              <div className="mode-content">
                <strong>Full Analysis</strong>
                <span>3 care card methods</span>
              </div>
            </label>

            <label className={`mode-option ${mode === 'rag' ? 'selected' : ''}`}>
              <input
                type="radio"
                value="rag"
                checked={mode === 'rag'}
                onChange={(e) => setMode(e.target.value)}
              />
              <div className="mode-content">
                <strong>RAG Enhanced</strong>
                <span>Best accuracy</span>
              </div>
            </label>
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={!selectedFile}
          className="btn btn-primary btn-large"
        >
          Analyze Plant
        </button>
      </div>
    </div>
  );
}

export default ImageUpload;
