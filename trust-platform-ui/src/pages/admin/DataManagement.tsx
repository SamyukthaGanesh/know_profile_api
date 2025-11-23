import React, { useState } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import './DataManagement.css';

export const DataManagement: React.FC = () => {
  const [bootstrapping, setBootstrapping] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleBootstrap = async () => {
    if (!window.confirm('Generate synthetic data? This will create 200 user profiles.')) {
      return;
    }

    setBootstrapping(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/bootstrap?seed=42&n_users=200', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error('Bootstrap failed');
      }

      const data = await response.json();
      setResult(data);
      alert(`✅ Data generation complete!\n\n${data.users_created || 200} users created\n${data.transactions_created || 'Many'} transactions created`);
    } catch (error) {
      console.error('Bootstrap error:', error);
      alert('❌ Failed to generate data. Check TrustBank backend.');
    } finally {
      setBootstrapping(false);
    }
  };

  return (
    <div className="data-management">
      <div className="page-header">
        <div>
          <h1>🗄️ Data Management</h1>
          <p>Generate synthetic data and manage datasets</p>
        </div>
      </div>

      <div className="management-grid">
        <Card>
          <CardHeader title="📊 Bootstrap Data Generator" />
          <div className="card-content">
            <p>
              Generate synthetic user profiles and transaction data for testing and demonstrations.
            </p>
            <div className="data-stats">
              <div className="stat-item">
                <span className="stat-label">Users to Generate:</span>
                <span className="stat-value">200</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Seed:</span>
                <span className="stat-value">42</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Transactions per User:</span>
                <span className="stat-value">~50-200</span>
              </div>
            </div>
            <Button
              variant="primary"
              onClick={handleBootstrap}
              disabled={bootstrapping}
              style={{ width: '100%', marginTop: '16px' }}
            >
              {bootstrapping ? '⏳ Generating Data...' : '🚀 Generate Synthetic Data'}
            </Button>
          </div>
        </Card>

        {result && (
          <Card>
            <CardHeader title="✅ Generation Results" />
            <div className="card-content">
              <pre className="result-json">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </Card>
        )}

        <Card>
          <CardHeader title="📈 Dataset Information" />
          <div className="card-content">
            <div className="info-list">
              <div className="info-item">
                <strong>User Profiles:</strong> CSV-based storage
              </div>
              <div className="info-item">
                <strong>Transactions:</strong> Historical transaction data
              </div>
              <div className="info-item">
                <strong>Consent Records:</strong> GDPR-compliant consent tracking
              </div>
              <div className="info-item">
                <strong>ML Predictions:</strong> Real-time credit risk assessments
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

