import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { api } from '../../services/api';
import { AdminOverviewResponse } from '../../types/api';
import './Overview.css';

export const AdminOverview: React.FC = () => {
  const [overview, setOverview] = useState<AdminOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadOverview();
  }, []);

  const loadOverview = async () => {
    try {
      const data = await api.getAdminOverview();
      setOverview(data);
    } catch (error) {
      console.error('Failed to load overview:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !overview) {
    return <div className="loading">Loading dashboard...</div>;
  }

  const { systemHealth, metrics, alerts, realtimeMetrics } = overview;

  const getAlertClass = (severity: string) => {
    if (severity === 'critical') return 'alert-critical';
    if (severity === 'warning') return 'alert-warning';
    return 'alert-info';
  };

  return (
    <div className="admin-overview">
      <div className="page-header">
        <div>
          <h1 className="page-title">AI Governance Overview</h1>
          <p className="page-subtitle">Last Update: Just now</p>
        </div>
        <div className="header-actions">
          {alerts.filter(a => a.severity === 'critical').length > 0 && (
            <Badge variant="danger" className="alert-badge">
              {alerts.filter(a => a.severity === 'critical').length} Critical Alerts
            </Badge>
          )}
          <Button variant="primary">Export Report</Button>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value positive">{systemHealth.overall}%</div>
          <div className="metric-label">System Health</div>
          <div className="metric-change positive">
            {systemHealth.trend.direction === 'up' ? '↑' : '↓'} {systemHealth.trend.change}% from last {systemHealth.trend.period}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{metrics.decisionsToday.toLocaleString()}</div>
          <div className="metric-label">Decisions Today</div>
          <div className="metric-change positive">
            ↑ {(metrics.decisionsToday - metrics.decisionsYesterday).toLocaleString()} from yesterday
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-value warning">{metrics.pendingApprovals}</div>
          <div className="metric-label">Pending Approvals</div>
          <div className="metric-change warning">Requires attention</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{metrics.biasScore.toFixed(3)}</div>
          <div className="metric-label">Bias Score</div>
          <div className="metric-change positive">
            ↓ Improvement
          </div>
        </div>
      </div>

      <Card className="dark-card">
        <CardHeader title="🔴 Critical Alerts" />

        {alerts.map((alert) => (
          <div key={alert.alertId} className={`alert-box ${getAlertClass(alert.severity)}`}>
            <span className="alert-icon">
              {alert.severity === 'critical' ? '⚠️' : 
               alert.severity === 'warning' ? '📊' : 'ℹ️'}
            </span>
            <div className="alert-content">
              <strong>{alert.title}</strong>
              <p>{alert.message}</p>
              <span className="alert-time">{new Date(alert.timestamp).toLocaleString()}</span>
            </div>
            <Button 
              variant={alert.severity === 'critical' ? 'danger' : 'warning'}
              size="small"
            >
              Handle
            </Button>
          </div>
        ))}
      </Card>

      <Card className="dark-card">
        <CardHeader title="📈 Real-Time Metrics" />

        <div className="realtime-grid">
          <div className="realtime-stat">
            <div className="stat-number success">{realtimeMetrics.approvalRate}%</div>
            <div className="stat-name">Approval Rate</div>
          </div>
          <div className="realtime-stat">
            <div className="stat-number danger">{realtimeMetrics.denialRate}%</div>
            <div className="stat-name">Denial Rate</div>
          </div>
          <div className="realtime-stat">
            <div className="stat-number warning">{realtimeMetrics.manualReviewRate}%</div>
            <div className="stat-name">Manual Review</div>
          </div>
          <div className="realtime-stat">
            <div className="stat-number">{realtimeMetrics.avgLatencyMs}ms</div>
            <div className="stat-name">Avg Latency</div>
          </div>
          <div className="realtime-stat">
            <div className="stat-number">{realtimeMetrics.throughputPerMin.toLocaleString()}</div>
            <div className="stat-name">Throughput/min</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

