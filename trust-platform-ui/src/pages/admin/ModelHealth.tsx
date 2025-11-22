import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { ProgressBar } from '../../components/shared/ProgressBar';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import './ModelHealth.css';

export const ModelHealth: React.FC = () => {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data for demo
    const mockModels = [
      {
        id: 'model-1',
        name: 'Credit Scoring Model v2.1',
        status: 'healthy',
        accuracy: 94.3,
        drift: 2.1,
        lastTraining: '2024-11-15',
        predictions: 45231,
        avgLatency: 45
      },
      {
        id: 'model-2',
        name: 'Loan Approval Model v3.0',
        status: 'warning',
        accuracy: 89.7,
        drift: 5.8,
        lastTraining: '2024-10-20',
        predictions: 32145,
        avgLatency: 67
      },
      {
        id: 'model-3',
        name: 'Fraud Detection Model v1.5',
        status: 'healthy',
        accuracy: 96.8,
        drift: 1.2,
        lastTraining: '2024-11-18',
        predictions: 78432,
        avgLatency: 32
      }
    ];

    setModels(mockModels);
    setLoading(false);
  }, []);

  const performanceData = [
    { date: 'Nov 15', accuracy: 93.5, precision: 92.1, recall: 94.2 },
    { date: 'Nov 16', accuracy: 93.8, precision: 92.4, recall: 94.5 },
    { date: 'Nov 17', accuracy: 94.0, precision: 92.8, recall: 94.7 },
    { date: 'Nov 18', accuracy: 94.1, precision: 93.0, recall: 94.8 },
    { date: 'Nov 19', accuracy: 94.3, precision: 93.2, recall: 95.0 },
    { date: 'Nov 20', accuracy: 94.2, precision: 93.1, recall: 94.9 },
    { date: 'Nov 21', accuracy: 94.3, precision: 93.3, recall: 95.1 }
  ];

  const driftData = [
    { feature: 'Income', drift: 1.8 },
    { feature: 'Credit Score', drift: 2.1 },
    { feature: 'DTI Ratio', drift: 3.2 },
    { feature: 'Payment History', drift: 1.5 },
    { feature: 'Credit Utilization', drift: 2.8 }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'critical': return 'danger';
      default: return 'info';
    }
  };

  if (loading) {
    return <div className="loading">Loading model health data...</div>;
  }

  return (
    <div className="model-health">
      <div className="page-header">
        <div>
          <h1 className="page-title">🧠 Model Health Monitor</h1>
          <p className="page-subtitle">Real-time model performance and drift detection</p>
        </div>
        <div className="header-actions">
          <Button variant="secondary">View Logs</Button>
          <Button variant="primary">Retrain Models</Button>
        </div>
      </div>

      {/* Model Cards */}
      <div className="model-cards-grid">
        {models.map((model) => (
          <Card key={model.id} className="model-card">
            <div className="model-card-header">
              <div>
                <h3 className="model-name">{model.name}</h3>
                <Badge variant={getStatusColor(model.status)}>
                  {model.status.toUpperCase()}
                </Badge>
              </div>
              <Button variant="secondary" size="small">Details</Button>
            </div>

            <div className="model-metrics">
              <div className="metric">
                <span className="metric-label">Accuracy</span>
                <span className="metric-value success">{model.accuracy}%</span>
              </div>
              <div className="metric">
                <span className="metric-label">Drift Score</span>
                <span className={`metric-value ${model.drift > 5 ? 'danger' : 'warning'}`}>
                  {model.drift}%
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Predictions</span>
                <span className="metric-value">{model.predictions.toLocaleString()}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Avg Latency</span>
                <span className="metric-value">{model.avgLatency}ms</span>
              </div>
            </div>

            <div className="model-footer">
              <span>Last Training: {new Date(model.lastTraining).toLocaleDateString()}</span>
              {model.drift > 5 && (
                <span className="warning-text">⚠️ Retraining recommended</span>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Performance Over Time */}
      <Card className="chart-card">
        <CardHeader title="📈 Model Performance Over Time" />
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis domain={[85, 100]} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="accuracy" stroke="#6366f1" strokeWidth={2} name="Accuracy" />
            <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2} name="Precision" />
            <Line type="monotone" dataKey="recall" stroke="#f59e0b" strokeWidth={2} name="Recall" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Feature Drift Analysis */}
      <Card className="chart-card">
        <CardHeader title="🔍 Feature Drift Detection" />
        <p className="drift-description">
          Feature drift indicates changes in input data distribution. Values above 5% require attention.
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={driftData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 10]} />
            <YAxis dataKey="feature" type="category" width={150} />
            <Tooltip formatter={(value: any) => `${value}%`} />
            <Bar dataKey="drift" fill="#6366f1">
              {driftData.map((entry, index) => (
                <Bar
                  key={`bar-${index}`}
                  dataKey="drift"
                  fill={entry.drift > 5 ? '#ef4444' : entry.drift > 3 ? '#f59e0b' : '#10b981'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Retraining Controls */}
      <Card className="retraining-card">
        <CardHeader title="🔄 Automated Retraining" />
        <div className="retraining-controls">
          <div className="control-group">
            <label>Drift Threshold</label>
            <input type="range" min="1" max="10" defaultValue="5" />
            <span>5%</span>
          </div>
          <div className="control-group">
            <label>Retraining Schedule</label>
            <select>
              <option>Weekly</option>
              <option>Bi-weekly</option>
              <option>Monthly</option>
              <option>On-demand</option>
            </select>
          </div>
          <div className="control-group">
            <label>Validation Split</label>
            <input type="range" min="10" max="30" defaultValue="20" />
            <span>20%</span>
          </div>
        </div>
        <div className="retraining-actions">
          <Button variant="secondary">Save Configuration</Button>
          <Button variant="primary">Trigger Retraining Now</Button>
        </div>
      </Card>
    </div>
  );
};

