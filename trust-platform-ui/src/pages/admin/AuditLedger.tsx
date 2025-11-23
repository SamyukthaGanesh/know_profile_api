import React, { useState, useEffect } from 'react';
import { ghciApi } from '../../services/ghciApi';
import type { AuditReceipt, LedgerStats, ComplianceStatus } from '../../types/ghci';
import { Card } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import './AuditLedger.css';

export const AuditLedger: React.FC = () => {
  const [receipts, setReceipts] = useState<AuditReceipt[]>([]);
  const [stats, setStats] = useState<LedgerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedReceipt, setSelectedReceipt] = useState<AuditReceipt | null>(null);
  const [chainValid, setChainValid] = useState<boolean>(true);
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({
    start: '',
    end: '',
  });
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadLedger();
    verifyChain();
  }, []);

  const loadLedger = async () => {
    try {
      setLoading(true);
      const response = await ghciApi.ledger.queryLedger({
        start_date: dateRange.start || undefined,
        end_date: dateRange.end || undefined,
        limit: 100,
      });
      setReceipts(response.receipts);
      setStats(response.stats);
      setChainValid(response.chain_valid);
    } catch (error) {
      console.error('Failed to load ledger:', error);
    } finally {
      setLoading(false);
    }
  };

  const verifyChain = async () => {
    try {
      const result = await ghciApi.ledger.verifyIntegrity();
      setChainValid(result.valid);
    } catch (error) {
      console.error('Failed to verify chain:', error);
    }
  };

  const getComplianceColor = (
    status: ComplianceStatus
  ): 'success' | 'warning' | 'danger' | 'info' | 'default' => {
    switch (status) {
      case 'compliant':
        return 'success';
      case 'violation':
        return 'danger';
      case 'warning':
        return 'warning';
      case 'requires_review':
        return 'info';
      default:
        return 'default';
    }
  };

  const filteredReceipts = receipts.filter((receipt) => {
    if (!searchTerm) return true;
    const search = searchTerm.toLowerCase();
    return (
      receipt.receipt_id.toLowerCase().includes(search) ||
      receipt.decision_id.toLowerCase().includes(search) ||
      receipt.policies_checked.some((p) => p.toLowerCase().includes(search))
    );
  });

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  if (loading) {
    return (
      <div className="audit-ledger">
        <div className="loading">Loading audit ledger...</div>
      </div>
    );
  }

  return (
    <div className="audit-ledger">
      <div className="page-header">
        <div>
          <h1 className="page-title">Audit Ledger</h1>
          <p className="page-subtitle">Blockchain-style cryptographic audit trail</p>
        </div>
        <div className="header-actions">
          <Button variant="secondary" onClick={verifyChain}>
            🔐 Verify Chain
          </Button>
          <Button variant="secondary" onClick={loadLedger}>
            🔄 Refresh
          </Button>
        </div>
      </div>

      {/* Chain Status */}
      <Card className="chain-status-card">
        <div className="chain-status">
          <div className="status-icon">
            {chainValid ? '✅' : '❌'}
          </div>
          <div className="status-content">
            <h3>{chainValid ? 'Chain Integrity: Valid' : 'Chain Integrity: COMPROMISED'}</h3>
            <p>
              {chainValid
                ? 'All cryptographic hashes verified. No tampering detected.'
                : 'WARNING: Hash chain verification failed. Possible tampering detected!'}
            </p>
          </div>
          {!chainValid && (
            <Badge variant="danger" className="status-badge">
              SECURITY ALERT
            </Badge>
          )}
        </div>
      </Card>

      {/* Stats Grid */}
      {stats && (
        <div className="ledger-stats-grid">
          <Card className="stat-card">
            <div className="stat-number">{stats.total_receipts}</div>
            <div className="stat-label">Total Receipts</div>
          </Card>
          <Card className="stat-card">
            <div className="stat-number success">{stats.receipts_today}</div>
            <div className="stat-label">Today</div>
          </Card>
          <Card className="stat-card">
            <div className="stat-number danger">{stats.violations_detected}</div>
            <div className="stat-label">Violations</div>
          </Card>
          <Card className="stat-card">
            <div className={`stat-number ${stats.chain_integrity ? 'success' : 'danger'}`}>
              {stats.chain_integrity ? '✓' : '✗'}
            </div>
            <div className="stat-label">Chain Status</div>
          </Card>
        </div>
      )}

      {/* Controls */}
      <Card>
        <div className="ledger-controls">
          <input
            type="text"
            placeholder="🔍 Search receipts, decisions, or policies..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <div className="date-filters">
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
              className="date-input"
            />
            <span>to</span>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
              className="date-input"
            />
            <Button variant="secondary" onClick={loadLedger}>
              Apply
            </Button>
          </div>
        </div>

        {/* Receipt List */}
        <div className="receipt-list">
          {filteredReceipts.length === 0 ? (
            <div className="no-receipts">No audit receipts found</div>
          ) : (
            filteredReceipts.map((receipt, index) => (
              <div key={receipt.receipt_id} className="receipt-item">
                <div className="receipt-header">
                  <div className="receipt-info">
                    <div className="receipt-chain-link">
                      <span className="block-number">Block #{receipts.length - index}</span>
                      <span className="chain-icon">⛓️</span>
                    </div>
                    <div className="receipt-ids">
                      <div className="receipt-id">
                        <strong>Receipt:</strong>
                        <code onClick={() => copyToClipboard(receipt.receipt_id)}>
                          {receipt.receipt_id}
                        </code>
                      </div>
                      <div className="decision-id">
                        <strong>Decision:</strong>
                        <code onClick={() => copyToClipboard(receipt.decision_id)}>
                          {receipt.decision_id}
                        </code>
                      </div>
                    </div>
                  </div>
                  <div className="receipt-actions">
                    <Button variant="secondary" onClick={() => setSelectedReceipt(receipt)}>
                      View Details
                    </Button>
                  </div>
                </div>

                <div className="receipt-content">
                  <div className="receipt-field">
                    <label>Timestamp:</label>
                    <span>{new Date(receipt.timestamp).toLocaleString()}</span>
                  </div>
                  <div className="receipt-field">
                    <label>Policies Checked:</label>
                    <div className="policy-badges">
                      {receipt.policies_checked.map((policy) => (
                        <Badge key={policy} variant="info">
                          {policy}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div className="receipt-field">
                    <label>Compliance Results:</label>
                    <div className="compliance-results">
                      {receipt.compliance_results.map((result, idx) => (
                        <div key={idx} className="compliance-result">
                          <Badge variant={getComplianceColor(result.status)}>
                            {result.status}
                          </Badge>
                          <span>{result.policy_name}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {receipt.decision_outcome && (
                    <div className="receipt-field">
                      <label>Decision:</label>
                      <Badge
                        variant={
                          receipt.decision_outcome === 'approved' ? 'success' : 'danger'
                        }
                      >
                        {receipt.decision_outcome}
                      </Badge>
                    </div>
                  )}
                </div>

                <div className="receipt-footer">
                  <div className="hash-display">
                    <div className="hash-item">
                      <label>Content Hash:</label>
                      <code className="hash" onClick={() => copyToClipboard(receipt.content_hash)}>
                        {receipt.content_hash.substring(0, 16)}...
                      </code>
                    </div>
                    <div className="hash-link">→</div>
                    <div className="hash-item">
                      <label>Previous Hash:</label>
                      <code
                        className="hash"
                        onClick={() => copyToClipboard(receipt.previous_hash)}
                      >
                        {receipt.previous_hash
                          ? receipt.previous_hash.substring(0, 16) + '...'
                          : 'Genesis Block'}
                      </code>
                    </div>
                  </div>
                  {receipt.created_by && (
                    <div className="created-by">Created by: {receipt.created_by}</div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </Card>

      {/* Receipt Detail Modal */}
      {selectedReceipt && (
        <div className="modal-overlay" onClick={() => setSelectedReceipt(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Audit Receipt Details</h2>
              <button className="close-btn" onClick={() => setSelectedReceipt(null)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              <div className="detail-section">
                <h3>Receipt Information</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <label>Receipt ID:</label>
                    <code>{selectedReceipt.receipt_id}</code>
                  </div>
                  <div className="detail-item">
                    <label>Decision ID:</label>
                    <code>{selectedReceipt.decision_id}</code>
                  </div>
                  <div className="detail-item">
                    <label>Timestamp:</label>
                    <span>{new Date(selectedReceipt.timestamp).toLocaleString()}</span>
                  </div>
                  {selectedReceipt.model_id && (
                    <div className="detail-item">
                      <label>Model ID:</label>
                      <code>{selectedReceipt.model_id}</code>
                    </div>
                  )}
                </div>
              </div>

              <div className="detail-section">
                <h3>Cryptographic Verification</h3>
                <div className="hash-details">
                  <div className="hash-detail-item">
                    <label>Content Hash (SHA-256):</label>
                    <code className="full-hash">{selectedReceipt.content_hash}</code>
                  </div>
                  <div className="hash-detail-item">
                    <label>Previous Hash:</label>
                    <code className="full-hash">
                      {selectedReceipt.previous_hash || 'Genesis Block (No previous hash)'}
                    </code>
                  </div>
                  {selectedReceipt.signature && (
                    <div className="hash-detail-item">
                      <label>Digital Signature:</label>
                      <code className="full-hash">{selectedReceipt.signature}</code>
                    </div>
                  )}
                </div>
              </div>

              <div className="detail-section">
                <h3>Compliance Results</h3>
                {selectedReceipt.compliance_results.map((result, idx) => (
                  <div key={idx} className="compliance-detail">
                    <div className="compliance-header">
                      <Badge variant={getComplianceColor(result.status)}>{result.status}</Badge>
                      <strong>{result.policy_name}</strong>
                    </div>
                    <p>{result.message}</p>
                    <div className="compliance-meta">
                      <span>Policy ID: {result.policy_id}</span>
                      <span>Source: {result.regulation_source}</span>
                      {result.recommended_action && (
                        <Badge variant="warning">{result.recommended_action}</Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {selectedReceipt.feature_values && (
                <div className="detail-section">
                  <h3>Feature Values</h3>
                  <pre className="feature-display">
                    {JSON.stringify(selectedReceipt.feature_values, null, 2)}
                  </pre>
                </div>
              )}

              {selectedReceipt.created_by && (
                <div className="detail-section">
                  <h3>Audit Trail</h3>
                  <div className="detail-grid">
                    <div className="detail-item">
                      <label>Created By:</label>
                      <span>{selectedReceipt.created_by}</span>
                    </div>
                    {selectedReceipt.ip_address && (
                      <div className="detail-item">
                        <label>IP Address:</label>
                        <code>{selectedReceipt.ip_address}</code>
                      </div>
                    )}
                    {selectedReceipt.user_agent && (
                      <div className="detail-item">
                        <label>User Agent:</label>
                        <span className="user-agent">{selectedReceipt.user_agent}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

