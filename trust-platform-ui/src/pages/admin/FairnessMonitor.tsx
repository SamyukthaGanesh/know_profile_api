import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { ProgressBar } from '../../components/shared/ProgressBar';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import './FairnessMonitor.css';

export const FairnessMonitor: React.FC = () => {
  const [fairnessData, setFairnessData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data for demo
    const mockData = {
      overallBiasScore: 0.023,
      protectedGroups: [
        { group: 'Gender', biasScore: 0.012, approvalRate: 68.5, trend: 'improving' },
        { group: 'Age', biasScore: 0.031, approvalRate: 65.2, trend: 'stable' },
        { group: 'Income Level', biasScore: 0.045, approvalRate: 62.8, trend: 'worsening' },
        { group: 'Location', biasScore: 0.019, approvalRate: 67.3, trend: 'improving' }
      ],
      rlOptimizer: {
        episodes: 1247,
        lastAdjustment: {
          action: 'Reduced weight on age feature',
          biasReduction: 2.8,
          accuracyImpact: -0.3
        }
      }
    };

    setFairnessData(mockData);
    setLoading(false);
  }, []);

  const biasHistory = [
    { episode: 0, bias: 0.089, accuracy: 92.1 },
    { episode: 200, bias: 0.067, accuracy: 92.3 },
    { episode: 400, bias: 0.051, accuracy: 92.8 },
    { episode: 600, bias: 0.042, accuracy: 93.2 },
    { episode: 800, bias: 0.034, accuracy: 93.5 },
    { episode: 1000, bias: 0.028, accuracy: 93.9 },
    { episode: 1247, bias: 0.023, accuracy: 94.3 }
  ];

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return '📈';
      case 'worsening': return '📉';
      default: return '➡️';
    }
  };

  const getBiasLevel = (score: number): { label: string; color: 'success' | 'warning' | 'danger' } => {
    if (score < 0.03) return { label: 'Low', color: 'success' };
    if (score < 0.05) return { label: 'Moderate', color: 'warning' };
    return { label: 'High', color: 'danger' };
  };

  if (loading || !fairnessData) {
    return <div className="loading">Loading fairness data...</div>;
  }

  return (
    <div className="fairness-monitor">
      <div className="page-header">
        <div>
          <h1 className="page-title">⚖️ Fairness Monitor</h1>
          <p className="page-subtitle">AI-powered bias detection and mitigation with RL optimizer</p>
        </div>
        <div className="header-actions">
          <Button variant="secondary">Export Report</Button>
          <Button variant="primary">Run Fairness Test</Button>
        </div>
      </div>

      {/* Overall Bias Score */}
      <Card className="bias-score-card">
        <div className="bias-score-content">
          <div className="score-section">
            <h2>Overall Bias Score</h2>
            <div className="score-display">
              <span className="score-value">{fairnessData.overallBiasScore.toFixed(3)}</span>
              <Badge variant={getBiasLevel(fairnessData.overallBiasScore).color}>
                {getBiasLevel(fairnessData.overallBiasScore).label} BIAS
              </Badge>
            </div>
            <p className="score-description">
              Target: &lt; 0.030 (Industry Standard) | Your Score: {fairnessData.overallBiasScore < 0.03 ? '✅ Compliant' : '⚠️ Needs Attention'}
            </p>
          </div>

          <div className="optimizer-section">
            <h3>🤖 Dynamic Fairness Optimizer (RL)</h3>
            <div className="optimizer-stats">
              <div className="optimizer-stat">
                <span className="stat-label">Episodes Run</span>
                <span className="stat-value">{fairnessData.rlOptimizer.episodes.toLocaleString()}</span>
              </div>
              <div className="optimizer-stat">
                <span className="stat-label">Last Adjustment</span>
                <span className="stat-value">{fairnessData.rlOptimizer.lastAdjustment.action}</span>
              </div>
              <div className="optimizer-stat">
                <span className="stat-label">Bias Reduction</span>
                <span className="stat-value success">-{fairnessData.rlOptimizer.lastAdjustment.biasReduction}%</span>
              </div>
              <div className="optimizer-stat">
                <span className="stat-label">Accuracy Impact</span>
                <span className="stat-value warning">{fairnessData.rlOptimizer.lastAdjustment.accuracyImpact}%</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Protected Groups Analysis */}
      <Card className="protected-groups-card">
        <CardHeader title="👥 Protected Groups Analysis" />
        <div className="groups-grid">
          {fairnessData.protectedGroups.map((group: any, index: number) => (
            <div key={index} className="group-card">
              <div className="group-header">
                <h4>{group.group}</h4>
                <span className="trend-badge">{getTrendIcon(group.trend)} {group.trend}</span>
              </div>
              <div className="group-metrics">
                <div className="group-metric">
                  <span className="metric-label">Bias Score</span>
                  <span className={`metric-value ${getBiasLevel(group.biasScore).color}`}>
                    {group.biasScore.toFixed(3)}
                  </span>
                </div>
                <div className="group-metric">
                  <span className="metric-label">Approval Rate</span>
                  <span className="metric-value">{group.approvalRate}%</span>
                </div>
              </div>
              <ProgressBar
                value={((0.1 - group.biasScore) / 0.1) * 100}
                variant={getBiasLevel(group.biasScore).color}
              />
            </div>
          ))}
        </div>
      </Card>

      {/* RL Optimizer Performance */}
      <Card className="chart-card">
        <CardHeader title="🧠 RL Optimizer: Bias Reduction Over Time" />
        <p className="chart-description">
          Our Reinforcement Learning optimizer continuously adjusts model weights to minimize bias while maintaining accuracy.
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={biasHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="episode" label={{ value: 'RL Episodes', position: 'insideBottom', offset: -5 }} />
            <YAxis yAxisId="left" label={{ value: 'Bias Score', angle: -90, position: 'insideLeft' }} />
            <YAxis yAxisId="right" orientation="right" label={{ value: 'Accuracy (%)', angle: 90, position: 'insideRight' }} />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="bias" stroke="#ef4444" strokeWidth={3} name="Bias Score" />
            <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={3} name="Accuracy" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Bias Mitigation Actions */}
      <Card className="mitigation-card">
        <CardHeader title="🛠️ Bias Mitigation Actions" />
        <div className="actions-list">
          <div className="action-item">
            <span className="action-icon">✅</span>
            <div className="action-content">
              <strong>Pre-processing</strong>
              <p>Reweighting training data to balance protected groups</p>
            </div>
            <Badge variant="success">Active</Badge>
          </div>
          <div className="action-item">
            <span className="action-icon">✅</span>
            <div className="action-content">
              <strong>In-processing</strong>
              <p>Adversarial debiasing during model training</p>
            </div>
            <Badge variant="success">Active</Badge>
          </div>
          <div className="action-item">
            <span className="action-icon">⚙️</span>
            <div className="action-content">
              <strong>Post-processing</strong>
              <p>Threshold optimization for equalized odds</p>
            </div>
            <Badge variant="warning">Testing</Badge>
          </div>
          <div className="action-item">
            <span className="action-icon">🤖</span>
            <div className="action-content">
              <strong>RL Optimizer</strong>
              <p>Dynamic weight adjustment using reinforcement learning</p>
            </div>
            <Badge variant="success">Active</Badge>
          </div>
        </div>
      </Card>
    </div>
  );
};

