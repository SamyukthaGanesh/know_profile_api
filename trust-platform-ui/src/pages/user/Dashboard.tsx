import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { api } from '../../services/api';
import { UserDashboardResponse } from '../../types/api';
import './Dashboard.css';

export const UserDashboard: React.FC = () => {
  const [dashboard, setDashboard] = useState<UserDashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    try {
      const data = await api.getUserDashboard();
      setDashboard(data);
    } catch (error) {
      console.error('Failed to load dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !dashboard) {
    return <div className="loading">Loading your dashboard...</div>;
  }

  const { user, trustScore, activeLoanApplication, quickStats } = dashboard;

  const getTrendIcon = (trend: string) => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  const getRiskColor = (risk: string) => {
    if (risk === 'low') return 'success';
    if (risk === 'medium') return 'warning';
    return 'danger';
  };

  return (
    <div className="user-dashboard">
      <div className="hero-section">
        <h1 className="hero-title">Welcome back, {user.name.split(' ')[0]}! 👋</h1>
        <p className="hero-subtitle">Your AI-powered banking with complete transparency</p>
        
        <div className="quick-stats-grid">
          <div className="stat-box">
            <div className="stat-value">{trustScore.overall}</div>
            <div className="stat-label">AI Trust Score</div>
            <div className={`stat-trend ${trustScore.trend}`}>
              {getTrendIcon(trustScore.trend)} {trustScore.trend}
            </div>
          </div>
          
          <div className="stat-box">
            <div className="stat-value">{quickStats.fairnessRating}%</div>
            <div className="stat-label">Fairness Rating</div>
          </div>
          
          <div className="stat-box">
            <div className="stat-value">{quickStats.activeConsents}</div>
            <div className="stat-label">Active Consents</div>
          </div>
          
          <div className="stat-box">
            <div className="stat-value">{quickStats.pendingActions}</div>
            <div className="stat-label">Pending Actions</div>
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        {activeLoanApplication && (
          <Card>
            <CardHeader 
              title="💳 Loan Application"
              badge={
                <Badge variant={
                  activeLoanApplication.status === 'approved' ? 'success' :
                  activeLoanApplication.status === 'denied' ? 'danger' :
                  'warning'
                }>
                  {activeLoanApplication.status.replace('_', ' ').toUpperCase()}
                </Badge>
              }
            />
            
            <div className="loan-amount">
              ₹{activeLoanApplication.amount.toLocaleString('en-IN')}
            </div>
            
            <div className="loan-details">
              <div className="loan-detail-item">
                <div className="detail-label">Confidence</div>
                <div className="detail-value">{Math.round(activeLoanApplication.confidence * 100)}%</div>
              </div>
              
              <div className="loan-detail-item">
                <div className="detail-label">Risk Level</div>
                <div className={`detail-value risk-${activeLoanApplication.riskLevel}`}>
                  {activeLoanApplication.riskLevel}
                </div>
              </div>
            </div>
            
            <div className="loan-actions">
              <Button variant="primary">🤔 Why This Decision?</Button>
              <Button variant="success">📈 How to Improve</Button>
            </div>
          </Card>
        )}

        <Card>
          <CardHeader title="🏆 Trust Score Breakdown" />
          
          <div className="trust-components">
            {Object.entries(trustScore.components).map(([key, value]) => (
              <div key={key} className="trust-component">
                <div className="component-header">
                  <span className="component-name">
                    {key.charAt(0).toUpperCase() + key.slice(1)}
                  </span>
                  <span className="component-value">{value}%</span>
                </div>
                <div className="component-bar">
                  <div 
                    className="component-fill" 
                    style={{ width: `${value}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardHeader title="🔐 Quick Actions" />
          
          <div className="quick-actions">
            <Button variant="primary" style={{ width: '100%' }}>
              View Consent Receipts
            </Button>
            <Button variant="primary" style={{ width: '100%' }}>
              Check AI Fairness
            </Button>
            <Button variant="primary" style={{ width: '100%' }}>
              View Decision Log
            </Button>
            <Button variant="primary" style={{ width: '100%' }}>
              Talk to AI Assistant
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};

