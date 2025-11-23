import React, { useState, useEffect } from 'react';
import { ghciApi } from '../../services/ghciApi';
import type { Policy, PolicyType, PolicyAction } from '../../types/ghci';
import { Card } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { ToggleSwitch } from '../../components/shared/ToggleSwitch';
import './PolicyManager.css';

export const PolicyManager: React.FC = () => {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'enabled' | 'disabled'>('all');
  const [selectedPolicy, setSelectedPolicy] = useState<Policy | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadPolicies();
  }, [filter]);

  const loadPolicies = async () => {
    try {
      setLoading(true);
      const enabledOnly = filter === 'enabled' ? true : filter === 'disabled' ? false : undefined;
      const response = await ghciApi.policy.listPolicies(enabledOnly);
      setPolicies(response.policies);
    } catch (error) {
      console.error('Failed to load policies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePolicy = async (policyId: string, currentStatus: boolean) => {
    try {
      await ghciApi.policy.updatePolicyStatus(policyId, !currentStatus);
      loadPolicies();
    } catch (error) {
      console.error('Failed to toggle policy:', error);
    }
  };

  const handleCreatePolicy = async () => {
    const name = prompt('Policy Name:');
    if (!name) return;
    const description = prompt('Policy Description:');
    if (!description) return;

    try {
      const response = await fetch('http://localhost:8001/compliance/policies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          description,
          policy_type: 'credit_risk',
          category: 'lending',
          severity: 'medium',
          conditions: [],
          action: 'flag_for_review',
          enabled: true
        })
      });

      if (response.ok) {
        alert('✅ Policy created successfully!');
        loadPolicies();
      } else {
        throw new Error('Failed to create policy');
      }
    } catch (error) {
      console.error('Error creating policy:', error);
      alert('❌ Failed to create policy');
    }
  };

  const handleCheckCompliance = async () => {
    try {
      const response = await fetch('http://localhost:8001/compliance/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: 'home_credit_default_predictor_v2',
          decision_id: 'test_decision_' + Date.now(),
          features: {
            credit_score: 700,
            dti_ratio: 0.35
          },
          prediction: 0.75,
          user_id: '1'
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`✅ Compliance Check Complete!\n\nPassed: ${result.passed}\nViolations: ${result.violations?.length || 0}`);
      } else {
        throw new Error('Compliance check failed');
      }
    } catch (error) {
      console.error('Error checking compliance:', error);
      alert('❌ Failed to check compliance');
    }
  };

  const getPolicyTypeColor = (type: PolicyType): 'success' | 'warning' | 'danger' | 'info' | 'default' => {
    switch (type) {
      case 'credit_risk':
        return 'danger';
      case 'fairness':
        return 'info';
      case 'data_protection':
        return 'warning';
      case 'anti_discrimination':
        return 'success';
      default:
        return 'default';
    }
  };

  const getActionColor = (action: PolicyAction): 'success' | 'warning' | 'danger' | 'info' | 'default' => {
    switch (action) {
      case 'deny':
      case 'block':
        return 'danger';
      case 'flag_for_review':
      case 'require_explanation':
        return 'warning';
      case 'log_warning':
        return 'info';
      case 'alert':
        return 'warning';
      default:
        return 'default';
    }
  };

  const filteredPolicies = policies.filter((policy) => {
    if (!searchTerm) return true;
    const search = searchTerm.toLowerCase();
    return (
      policy.name.toLowerCase().includes(search) ||
      policy.description.toLowerCase().includes(search) ||
      policy.regulation_source.toLowerCase().includes(search) ||
      policy.policy_id.toLowerCase().includes(search)
    );
  });

  if (loading) {
    return (
      <div className="policy-manager">
        <div className="loading">Loading policies...</div>
      </div>
    );
  }

  return (
    <div className="policy-manager">
      <div className="page-header">
        <div>
          <h1 className="page-title">Policy Manager</h1>
          <p className="page-subtitle">Manage regulatory policies and compliance rules</p>
        </div>
        <div className="header-actions">
          <Button variant="secondary" onClick={loadPolicies}>
            🔄 Refresh
          </Button>
        </div>
      </div>

      <div className="policy-stats-grid">
        <Card className="stat-card">
          <div className="stat-number">{policies.length}</div>
          <div className="stat-label">Total Policies</div>
        </Card>
        <Card className="stat-card">
          <div className="stat-number success">{policies.filter((p) => p.enabled).length}</div>
          <div className="stat-label">Enabled</div>
        </Card>
        <Card className="stat-card">
          <div className="stat-number warning">{policies.filter((p) => !p.enabled).length}</div>
          <div className="stat-label">Disabled</div>
        </Card>
      </div>

      <Card>
        <div className="policy-controls">
          <input
            type="text"
            placeholder="🔍 Search policies..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <div className="filter-buttons">
            <Button
              variant={filter === 'all' ? 'primary' : 'secondary'}
              onClick={() => setFilter('all')}
            >
              All
            </Button>
            <Button
              variant={filter === 'enabled' ? 'primary' : 'secondary'}
              onClick={() => setFilter('enabled')}
            >
              Enabled
            </Button>
            <Button
              variant={filter === 'disabled' ? 'primary' : 'secondary'}
              onClick={() => setFilter('disabled')}
            >
              Disabled
            </Button>
          </div>
        </div>

        <div className="policy-list">
          {filteredPolicies.length === 0 ? (
            <div className="no-policies">No policies found</div>
          ) : (
            filteredPolicies.map((policy) => (
              <div key={policy.policy_id} className="policy-item">
                <div className="policy-header">
                  <div className="policy-info">
                    <h3 className="policy-name">{policy.name}</h3>
                    <div className="policy-meta">
                      <span className="policy-id">{policy.policy_id}</span>
                      <Badge variant={getPolicyTypeColor(policy.policy_type)}>
                        {policy.policy_type?.replace(/_/g, ' ') || 'Unknown'}
                      </Badge>
                      {policy.action && (
                        <Badge variant={getActionColor(policy.action)}>
                          {policy.action?.replace(/_/g, ' ') || policy.action}
                        </Badge>
                      )}
                      <span className="regulation-source">📋 {policy.regulation_source}</span>
                    </div>
                  </div>
                  <div className="policy-actions">
                    <ToggleSwitch
                      checked={policy.enabled}
                      onChange={() => handleTogglePolicy(policy.policy_id, policy.enabled)}
                      label={policy.enabled ? 'Enabled' : 'Disabled'}
                    />
                    <Button
                      variant="secondary"
                      onClick={() => setSelectedPolicy(policy)}
                    >
                      View Details
                    </Button>
                  </div>
                </div>
                <p className="policy-description">{policy.description}</p>
                {policy.rationale && (
                  <div className="policy-rationale">
                    <strong>Rationale:</strong> {policy.rationale}
                  </div>
                )}
                <div className="policy-footer">
                  <span>Priority: {policy.priority}</span>
                  {policy.jurisdiction && <span>Jurisdiction: {policy.jurisdiction}</span>}
                  {policy.tags && policy.tags.length > 0 && (
                    <span className="tags">
                      {policy.tags.map((tag) => (
                        <Badge key={tag} variant="default">
                          {tag}
                        </Badge>
                      ))}
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </Card>

      {selectedPolicy && (
        <div className="modal-overlay" onClick={() => setSelectedPolicy(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedPolicy.name}</h2>
              <button className="close-btn" onClick={() => setSelectedPolicy(null)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              <div className="detail-section">
                <h3>Policy Details</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <label>Policy ID:</label>
                    <span>{selectedPolicy.policy_id}</span>
                  </div>
                  <div className="detail-item">
                    <label>Regulation Source:</label>
                    <span>{selectedPolicy.regulation_source}</span>
                  </div>
                  <div className="detail-item">
                    <label>Type:</label>
                    <Badge variant={getPolicyTypeColor(selectedPolicy.policy_type)}>
                      {selectedPolicy.policy_type?.replace(/_/g, ' ') || selectedPolicy.policy_type || 'Unknown'}
                    </Badge>
                  </div>
                  <div className="detail-item">
                    <label>Action:</label>
                    <Badge variant={getActionColor(selectedPolicy.action)}>
                      {selectedPolicy.action?.replace(/_/g, ' ') || selectedPolicy.action || 'N/A'}
                    </Badge>
                  </div>
                  <div className="detail-item">
                    <label>Version:</label>
                    <span>{selectedPolicy.version}</span>
                  </div>
                  <div className="detail-item">
                    <label>Priority:</label>
                    <span>{selectedPolicy.priority}</span>
                  </div>
                  {selectedPolicy.jurisdiction && (
                    <div className="detail-item">
                      <label>Jurisdiction:</label>
                      <span>{selectedPolicy.jurisdiction}</span>
                    </div>
                  )}
                  <div className="detail-item">
                    <label>Status:</label>
                    <Badge variant={selectedPolicy.enabled ? 'success' : 'warning'}>
                      {selectedPolicy.enabled ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3>Description</h3>
                <p>{selectedPolicy.description}</p>
              </div>

              {selectedPolicy.rationale && (
                <div className="detail-section">
                  <h3>Rationale</h3>
                  <p>{selectedPolicy.rationale}</p>
                </div>
              )}

              <div className="detail-section">
                <h3>Policy Condition</h3>
                <pre className="condition-display">
                  {JSON.stringify(selectedPolicy.condition, null, 2)}
                </pre>
              </div>

              {selectedPolicy.references && selectedPolicy.references.length > 0 && (
                <div className="detail-section">
                  <h3>References</h3>
                  <ul>
                    {selectedPolicy.references.map((ref, idx) => (
                      <li key={idx}>{ref}</li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedPolicy.tags && selectedPolicy.tags.length > 0 && (
                <div className="detail-section">
                  <h3>Tags</h3>
                  <div className="tag-list">
                    {selectedPolicy.tags.map((tag) => (
                      <Badge key={tag} variant="default">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <div className="detail-section">
                <h3>Audit Trail</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <label>Created:</label>
                    <span>{new Date(selectedPolicy.created_at).toLocaleString()}</span>
                  </div>
                  {selectedPolicy.created_by && (
                    <div className="detail-item">
                      <label>Created By:</label>
                      <span>{selectedPolicy.created_by}</span>
                    </div>
                  )}
                  {selectedPolicy.updated_at && (
                    <div className="detail-item">
                      <label>Last Updated:</label>
                      <span>{new Date(selectedPolicy.updated_at).toLocaleString()}</span>
                    </div>
                  )}
                  {selectedPolicy.updated_by && (
                    <div className="detail-item">
                      <label>Updated By:</label>
                      <span>{selectedPolicy.updated_by}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

