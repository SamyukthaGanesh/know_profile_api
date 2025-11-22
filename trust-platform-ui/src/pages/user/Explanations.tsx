import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { api } from '../../services/api';
import { ExplanationResponse } from '../../types/api';
import './Explanations.css';

export const Explanations: React.FC = () => {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [literacyLevel, setLiteracyLevel] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadExplanation();
  }, [literacyLevel]);

  const loadExplanation = async () => {
    try {
      const data = await api.getExplanation('APP-2024-001', literacyLevel);
      setExplanation(data);
    } catch (error) {
      console.error('Failed to load explanation:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !explanation) {
    return <div className="loading">Loading explanation...</div>;
  }

  return (
    <div className="explanations-page">
      <Card>
        <CardHeader 
          title="🧠 Personal AI Literacy Assistant"
          action={
            <div className="literacy-selector">
              <button 
                className={`literacy-btn ${literacyLevel === 'beginner' ? 'active' : ''}`}
                onClick={() => setLiteracyLevel('beginner')}
              >
                👶 Beginner
              </button>
              <button 
                className={`literacy-btn ${literacyLevel === 'intermediate' ? 'active' : ''}`}
                onClick={() => setLiteracyLevel('intermediate')}
              >
                🎓 Intermediate
              </button>
              <button 
                className={`literacy-btn ${literacyLevel === 'advanced' ? 'active' : ''}`}
                onClick={() => setLiteracyLevel('advanced')}
              >
                🏆 Advanced
              </button>
            </div>
          }
        />

        <div className="explanation-summary">
          <h3>Your Loan Decision Explained Simply</h3>
          <p>{explanation.explanation.summary}</p>
        </div>

        <h3 className="section-title">Main Factors Affecting Your Decision:</h3>

        {explanation.explanation.factors.map((factor, index) => (
          <div 
            key={index} 
            className={`factor-item ${factor.impact}`}
          >
            <div className="factor-icon">
              {factor.impact === 'positive' ? '✓' : '✗'}
            </div>
            <div className="factor-content">
              <strong>{factor.humanReadableName}</strong>
              <p>{factor.description}</p>
            </div>
            <div className={`factor-impact ${factor.impact}`}>
              {factor.impactPercentage > 0 ? '+' : ''}{factor.impactPercentage}%
            </div>
          </div>
        ))}

        {explanation.educationalContent.length > 0 && (
          <div className="educational-content">
            <h3>📚 What Do These Terms Mean?</h3>
            {explanation.educationalContent.map((content, index) => (
              <div key={index} className="educational-item">
                <strong>{content.topic}:</strong> {content.content}
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHeader title="💡 Your Personalized Improvement Plan" />

        {explanation.improvementSuggestions.map((suggestion, index) => (
          <div key={index} className={`improvement-item priority-${suggestion.priority}`}>
            <div className="improvement-number">{suggestion.priority}</div>
            <div className="improvement-content">
              <strong>{suggestion.action}</strong>
              <p>{suggestion.description}</p>
              <div className="improvement-meta">
                <span className="timeline">⏰ Timeline: {suggestion.timelineMonths} months</span>
                <span className="difficulty">Difficulty: {suggestion.difficulty}</span>
                <span className="impact">{suggestion.expectedImpact}</span>
              </div>
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
};

