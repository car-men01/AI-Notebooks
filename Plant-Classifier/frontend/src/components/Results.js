import React, { useState } from 'react';
import './Results.css';

function Results({ results, imageUrl, onReset }) {
  const [activeTab, setActiveTab] = useState('direct_llm');

  // Determine if this is basic mode (only classification) or full/rag mode
  const classification = results.classification || results;
  const hasCareCareCards = results.plant_care_cards || results.plant_care_card;

  const renderClassification = () => {
    if (!classification) return null;

    return (
      <div className="classification-section">
        <h2>🔍 Classification Results</h2>
        <div className="classification-card">
          {classification.top_3_predictions && classification.top_3_predictions.length > 0 ? (
            <div className="top-predictions">
              <h4>🏆 Top 3 Most Likely Plants (Descending by Confidence):</h4>
              <div className="predictions-list">
                {classification.top_3_predictions.map((pred, idx) => (
                  <div key={idx} className={`prediction-item ${idx === 0 ? 'primary' : ''}`}>
                    <div className="prediction-rank">#{idx + 1}</div>
                    <div className="prediction-details">
                      <div className="prediction-name">{pred.class}</div>
                      <div className="confidence-bar-container">
                        <div
                          className="confidence-bar"
                          style={{ width: `${(pred.confidence * 100).toFixed(1)}%` }}
                        ></div>
                        <span className="confidence-value">
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <>
              <div className="predicted-class">
                <span className="label">Identified Plant:</span>
                <span className="value">{classification.predicted_class}</span>
              </div>
              <div className="confidence">
                <span className="label">Confidence:</span>
                <div className="confidence-bar-container">
                  <div
                    className="confidence-bar"
                    style={{ width: `${(classification.confidence * 100).toFixed(1)}%` }}
                  ></div>
                  <span className="confidence-value">
                    {(classification.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  const renderCareCard = (card, title) => {
    if (!card) return null;

    return (
      <div className="care-card">
        <h3>{title || card.plant_name}</h3>
        
        <div className="care-section">
          <h4>📋 Basic Information</h4>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Common Name:</span>
              <span className="info-value">{card.plant_name}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Latin Name:</span>
              <span className="info-value italic">{card.latin_name}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Location:</span>
              <span className="info-value">
                {card.outdoors ? '🌳 Outdoors' : '🏠 Indoors'}
              </span>
            </div>
          </div>
        </div>

        <div className="care-section">
          <h4>☀️ Lighting</h4>
          <p><strong>Type:</strong> {card.lighting_conditions.type}</p>
          {card.lighting_conditions.duration && (
            <p><strong>Duration:</strong> {card.lighting_conditions.duration}</p>
          )}
          {card.lighting_conditions.notes && (
            <p className="notes">{card.lighting_conditions.notes}</p>
          )}
        </div>

        <div className="care-section">
          <h4>💧 Watering</h4>
          <p><strong>Frequency:</strong> {card.watering.frequency}</p>
          {card.watering.method && (
            <p><strong>Method:</strong> {card.watering.method}</p>
          )}
          {card.watering.notes && (
            <p className="notes">{card.watering.notes}</p>
          )}
        </div>

        <div className="care-section">
          <h4>🌡️ Temperature & Humidity</h4>
          <div className="info-grid">
            {card.temperature_range.min_celsius !== null && (
              <div className="info-item">
                <span className="info-label">Temperature Range:</span>
                <span className="info-value">
                  {card.temperature_range.min_celsius}°C - {card.temperature_range.max_celsius}°C
                </span>
              </div>
            )}
            <div className="info-item">
              <span className="info-label">Humidity Level:</span>
              <span className="info-value">{card.humidity.level}</span>
            </div>
          </div>
          {card.temperature_range.notes && (
            <p className="notes">{card.temperature_range.notes}</p>
          )}
          {card.humidity.notes && (
            <p className="notes">{card.humidity.notes}</p>
          )}
        </div>

        <div className="care-section">
          <h4>🌱 Soil & Propagation</h4>
          <p><strong>Soil Type:</strong> {card.soil_type}</p>
          <p><strong>Propagation:</strong> {card.propagation}</p>
        </div>

        {card.special_care && (
          <div className="care-section special-care">
            <h4>⚠️ Special Care Notes</h4>
            <p>{card.special_care}</p>
          </div>
        )}
      </div>
    );
  };

  const renderResults = () => {
    // RAG mode - single care card
    if (results.plant_care_card && results.method) {
      return (
        <div className="care-cards-container">
          {renderCareCard(results.plant_care_card, `${results.plant_care_card.plant_name} Care Guide`)}
        </div>
      );
    }

    // Full mode - 3 care cards with tabs
    if (results.plant_care_cards) {
      return (
        <div className="care-cards-container">
          <div className="tabs">
            <button
              className={`tab ${activeTab === 'direct_llm' ? 'active' : ''}`}
              onClick={() => setActiveTab('direct_llm')}
            >
              Direct LLM
            </button>
            <button
              className={`tab ${activeTab === 'web_search' ? 'active' : ''}`}
              onClick={() => setActiveTab('web_search')}
            >
              Web Search
            </button>
            <button
              className={`tab ${activeTab === 'combined' ? 'active' : ''}`}
              onClick={() => setActiveTab('combined')}
            >
              Combined
            </button>
          </div>

          {activeTab === 'direct_llm' && renderCareCard(results.plant_care_cards.direct_llm, 'Direct LLM Method')}
          {activeTab === 'web_search' && renderCareCard(results.plant_care_cards.web_search, 'Web Search Method')}
          {activeTab === 'combined' && renderCareCard(results.plant_care_cards.combined, 'Combined Method')}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>🌿 Analysis Complete</h2>
        <button onClick={onReset} className="btn btn-secondary">
          Analyze Another Plant
        </button>
      </div>

      {imageUrl && (
        <div className="image-display">
          <img src={imageUrl} alt="Analyzed plant" />
        </div>
      )}

      {renderClassification()}
      {renderResults()}
    </div>
  );
}

export default Results;
