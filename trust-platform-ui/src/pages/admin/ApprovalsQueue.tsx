import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { Modal } from '../../components/shared/Modal';
import './ApprovalsQueue.css';

interface Approval {
  id: string;
  type: 'model_update' | 'policy_change' | 'threshold_adjustment' | 'feature_deployment';
  title: string;
  description: string;
  requestedBy: string;
  requestedAt: string;
  priority: 'high' | 'medium' | 'low';
  status: 'pending' | 'approved' | 'rejected';
  details: any;
}

export const ApprovalsQueue: React.FC = () => {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [selectedApproval, setSelectedApproval] = useState<Approval | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [filter, setFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('pending');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data for demo
    const mockApprovals: Approval[] = [
      {
        id: 'appr-001',
        type: 'model_update',
        title: 'Credit Scoring Model v2.2 Deployment',
        description: 'New model with improved accuracy (94.3% → 95.1%) and reduced bias',
        requestedBy: 'ML Team',
        requestedAt: '2024-11-21T10:30:00Z',
        priority: 'high',
        status: 'pending',
        details: {
          accuracyImprovement: 0.8,
          biasReduction: 1.2,
          backtestingPassed: true,
          fairnessPassed: true
        }
      },
      {
        id: 'appr-002',
        type: 'policy_change',
        title: 'Update Loan Approval Threshold',
        description: 'Adjust approval threshold from 0.65 to 0.60 for low-income applicants',
        requestedBy: 'Policy Team',
        requestedAt: '2024-11-21T09:15:00Z',
        priority: 'high',
        status: 'pending',
        details: {
          currentThreshold: 0.65,
          proposedThreshold: 0.60,
          impactedUsers: 3240,
          estimatedApprovalIncrease: 12.5
        }
      },
      {
        id: 'appr-003',
        type: 'threshold_adjustment',
        title: 'Fraud Detection Sensitivity Increase',
        description: 'Increase fraud detection sensitivity to reduce false negatives',
        requestedBy: 'Security Team',
        requestedAt: '2024-11-20T16:45:00Z',
        priority: 'medium',
        status: 'pending',
        details: {
          currentSensitivity: 0.85,
          proposedSensitivity: 0.90,
          falseNegativeReduction: 15,
          falsePositiveIncrease: 8
        }
      },
      {
        id: 'appr-004',
        type: 'feature_deployment',
        title: 'Add Social Media Score Feature',
        description: 'Integrate social media activity score into credit assessment',
        requestedBy: 'Product Team',
        requestedAt: '2024-11-20T14:20:00Z',
        priority: 'low',
        status: 'pending',
        details: {
          featureName: 'social_media_score',
          expectedImpact: 'Accuracy +2.1%',
          privacyConcerns: 'Medium',
          regulatoryApprovalRequired: true
        }
      }
    ];

    setApprovals(mockApprovals);
    setLoading(false);
  }, []);

  const handleApprove = (id: string) => {
    setApprovals(approvals.map(a => 
      a.id === id ? { ...a, status: 'approved' as const } : a
    ));
    setShowModal(false);
  };

  const handleReject = (id: string) => {
    setApprovals(approvals.map(a => 
      a.id === id ? { ...a, status: 'rejected' as const } : a
    ));
    setShowModal(false);
  };

  const openModal = (approval: Approval) => {
    setSelectedApproval(approval);
    setShowModal(true);
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'model_update': return '🤖';
      case 'policy_change': return '📋';
      case 'threshold_adjustment': return '⚖️';
      case 'feature_deployment': return '🚀';
      default: return '📄';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'danger';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const filteredApprovals = approvals.filter(a => 
    filter === 'all' || a.status === filter
  );

  if (loading) {
    return <div className="loading">Loading approvals queue...</div>;
  }

  return (
    <div className="approvals-queue">
      <div className="page-header">
        <div>
          <h1 className="page-title">✅ Human-in-the-Loop Approvals</h1>
          <p className="page-subtitle">Review and approve critical system changes</p>
        </div>
        <div className="header-stats">
          <div className="stat-badge pending">
            <span className="stat-number">{approvals.filter(a => a.status === 'pending').length}</span>
            <span className="stat-label">Pending</span>
          </div>
          <div className="stat-badge approved">
            <span className="stat-number">{approvals.filter(a => a.status === 'approved').length}</span>
            <span className="stat-label">Approved</span>
          </div>
          <div className="stat-badge rejected">
            <span className="stat-number">{approvals.filter(a => a.status === 'rejected').length}</span>
            <span className="stat-label">Rejected</span>
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="filter-tabs">
        <button
          className={`filter-tab ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          All ({approvals.length})
        </button>
        <button
          className={`filter-tab ${filter === 'pending' ? 'active' : ''}`}
          onClick={() => setFilter('pending')}
        >
          Pending ({approvals.filter(a => a.status === 'pending').length})
        </button>
        <button
          className={`filter-tab ${filter === 'approved' ? 'active' : ''}`}
          onClick={() => setFilter('approved')}
        >
          Approved ({approvals.filter(a => a.status === 'approved').length})
        </button>
        <button
          className={`filter-tab ${filter === 'rejected' ? 'active' : ''}`}
          onClick={() => setFilter('rejected')}
        >
          Rejected ({approvals.filter(a => a.status === 'rejected').length})
        </button>
      </div>

      {/* Approvals List */}
      <div className="approvals-list">
        {filteredApprovals.map((approval) => (
          <Card key={approval.id} className="approval-card">
            <div className="approval-header">
              <div className="approval-title-section">
                <span className="type-icon">{getTypeIcon(approval.type)}</span>
                <div>
                  <h3 className="approval-title">{approval.title}</h3>
                  <p className="approval-description">{approval.description}</p>
                </div>
              </div>
              <div className="approval-badges">
                <Badge variant={getPriorityColor(approval.priority) as any}>
                  {approval.priority.toUpperCase()}
                </Badge>
                <Badge variant={
                  approval.status === 'approved' ? 'success' :
                  approval.status === 'rejected' ? 'danger' : 'warning'
                }>
                  {approval.status.toUpperCase()}
                </Badge>
              </div>
            </div>

            <div className="approval-meta">
              <span>👤 Requested by: <strong>{approval.requestedBy}</strong></span>
              <span>🕒 {new Date(approval.requestedAt).toLocaleString()}</span>
            </div>

            <div className="approval-actions">
              <Button variant="secondary" size="small" onClick={() => openModal(approval)}>
                View Details
              </Button>
              {approval.status === 'pending' && (
                <>
                  <Button variant="success" size="small" onClick={() => handleApprove(approval.id)}>
                    ✓ Approve
                  </Button>
                  <Button variant="danger" size="small" onClick={() => handleReject(approval.id)}>
                    ✗ Reject
                  </Button>
                </>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Detail Modal */}
      {showModal && selectedApproval && (
        <Modal
          isOpen={showModal}
          onClose={() => setShowModal(false)}
          title={selectedApproval.title}
        >
          <div className="approval-detail">
            <div className="detail-section">
              <h4>Description</h4>
              <p>{selectedApproval.description}</p>
            </div>

            <div className="detail-section">
              <h4>Request Information</h4>
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Requested By:</span>
                  <span className="detail-value">{selectedApproval.requestedBy}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Requested At:</span>
                  <span className="detail-value">
                    {new Date(selectedApproval.requestedAt).toLocaleString()}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Priority:</span>
                  <Badge variant={getPriorityColor(selectedApproval.priority) as any}>
                    {selectedApproval.priority.toUpperCase()}
                  </Badge>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Status:</span>
                  <Badge variant={
                    selectedApproval.status === 'approved' ? 'success' :
                    selectedApproval.status === 'rejected' ? 'danger' : 'warning'
                  }>
                    {selectedApproval.status.toUpperCase()}
                  </Badge>
                </div>
              </div>
            </div>

            <div className="detail-section">
              <h4>Technical Details</h4>
              <pre className="detail-code">
                {JSON.stringify(selectedApproval.details, null, 2)}
              </pre>
            </div>

            <div className="modal-actions">
              {selectedApproval.status === 'pending' && (
                <>
                  <Button variant="success" onClick={() => handleApprove(selectedApproval.id)}>
                    ✓ Approve
                  </Button>
                  <Button variant="danger" onClick={() => handleReject(selectedApproval.id)}>
                    ✗ Reject
                  </Button>
                </>
              )}
              <Button variant="secondary" onClick={() => setShowModal(false)}>
                Close
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

