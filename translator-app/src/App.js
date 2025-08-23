import React, { useState, useEffect } from 'react';
import './App.css';

// require('dotenv').config();

const API_BASE_URL = process.env.REACT_APP_API_URL 

function App() {
  const [activeTab] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [translation, setTranslation] = useState(null);
  const [loading, setLoading] = useState(false);          
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      console.log('Checking API health at:', `${API_BASE_URL}/health`);
      const response = await fetch(`${API_BASE_URL}/health`);
      console.log('Health response status:', response.status);
      const data = await response.json();
      console.log('Health data:', data);
      setHealth(data);
    } catch (err) {
      console.error('Health check failed:', err);
      setError('Cannot connect to translation service');
    }
  };

  const handleTextTranslation = async () => {
    if (!textInput.trim()) {
      setError('Please enter some text to translate');
      return;
    }

    setLoading(true);
    setError(null);
    setTranslation(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); 
  
    //  translation request start time
    const startTime = Date.now();
    console.log(`Translation request started at: ${new Date(startTime).toISOString()}`);
  
    try {
      const response = await fetch(`${API_BASE_URL}/translate/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textInput,
          beam_size: 5
        }),
        signal: controller.signal 
      });
  
      clearTimeout(timeoutId); 
      const data = await response.json();
      
      // logging translation request end time and duration
      const endTime = Date.now();
      const duration = endTime - startTime;
      console.log(` Translation response received at: ${new Date(endTime).toISOString()}`);
      console.log(`⏱Translation request duration: ${duration}ms (${(duration / 1000).toFixed(2)}s)`);
      
      if (data.success) {
        setTranslation(data);
      } else {
        setError(data.error || 'Translation failed');
      }
    } catch (err) {
      clearTimeout(timeoutId);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      console.log(`Translation request failed at: ${new Date(endTime).toISOString()}`);
      console.log(` Translation request duration (failed): ${duration}ms (${(duration / 1000).toFixed(2)}s)`);

    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Kannada to English Translator</h1>   
          
          {health && (
            <div className={`health-status ${health.model_loaded ? 'healthy' : 'unhealthy'}`}>
              <span className="status-indicator"></span>
              {health.model_loaded ? 'Translation model ready' : 'Model loading...'}
              <small> ({health.device})</small>
            </div>
          )}
        </header>

        <div className="content">
          {activeTab === 'text' && (
            <div className="text-translation">
              <div className="input-section">
                <label htmlFor="text-input">Enter Kannada text:</label>
                <textarea
                  id="text-input"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Type your Kannada text here..."
                  rows="4"
                />
                <button
                  onClick={handleTextTranslation}
                  disabled={loading || !textInput.trim()}
                  className="translate-btn"
                >
                  {loading ? 'Translating...' : 'Translate'}
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {translation && (
            <div className="translation-result">
              <h3>Translation Result</h3>
              
              {translation.original_text && (
                <div className="text-section">
                  <div className="text-header">
                    <span className="text-label">Original Text (Kannada):</span>
                  </div>
                  <div className="text-content original">
                    {translation.original_text}
                  </div>
                </div>
              )}

              <div className="text-section">
                <div className="text-header">
                  <span className="text-label">Translated Text (English):</span>
                </div>
                <div className="text-content translated">
                  {translation.translated_text}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;