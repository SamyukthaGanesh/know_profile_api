import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { ToggleSwitch } from '../../components/shared/ToggleSwitch';
import { Modal } from '../../components/shared/Modal';
import { api } from '../../services/api';
import { ConsentItem, ConsentReceipt } from '../../types/api';
import './ConsentWallet.css';

export const ConsentWallet: React.FC = () => {
  const [consents, setConsents] = useState<ConsentItem[]>([]);
  const [receipts, setReceipts] = useState<ConsentReceipt[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReceipt, setSelectedReceipt] = useState<ConsentReceipt | null>(null);
  const [showReceiptModal, setShowReceiptModal] = useState(false);

  useEffect(() => {
    loadConsents();
  }, []);

  const loadConsents = async () => {
    console.log('🔄 Loading consents...');
    try {
      const data = await api.getConsentStatus();
      console.log('✅ Consents loaded:', data.consents.length, 'items');
      console.log('📋 Consent data:', data.consents);
      setConsents(data.consents);
    } catch (error) {
      console.error('❌ Failed to load consents:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleConsent = async (consentId: string, newCheckedState: boolean) => {
    const newAction = newCheckedState ? 'grant' : 'revoke';
    const newStatus = newCheckedState ? 'granted' : 'revoked';
    
    try {
      const response = await api.updateConsent(consentId, newAction);
      
      // Update local state for THIS specific consent only
      setConsents(prevConsents => prevConsents.map(c => 
        c.consentId === consentId 
          ? { 
              ...c, 
              status: newStatus as 'granted' | 'revoked',
              revokedAt: !newCheckedState ? new Date().toISOString() : undefined,
              dataUsageCount: !newCheckedState ? 0 : c.dataUsageCount
            }
          : c
      ));
      
      // Add receipt
      if (response.receipt) {
        setReceipts([response.receipt, ...receipts]);
      }
      
      // Show notification
      console.log(`✅ Consent ${consentId} ${newAction}ed successfully!`);
    } catch (error) {
      console.error('Failed to update consent:', error);
      // Reload consents to get correct state
      loadConsents();
    }
  };

  const getCategoryIcon = (category: string) => {
    const icons: Record<string, string> = {
      fraud_detection: '🔍',
      loan_approval: '💳',
      personalization: '🎯',
      marketing: '📊',
      model_training: '🤖',
    };
    return icons[category] || '📋';
  };

  if (loading) {
    console.log('📊 Showing loading screen for consent wallet...');
    return <div className="loading">Loading consent wallet...</div>;
  }

  console.log('✅ Rendering consent wallet with', consents.length, 'consents');
  const activeConsents = consents.filter(c => c.status === 'granted').length;

  if (consents.length === 0) {
    console.log('⚠️ No consents to display!');
    return (
      <div className="consent-wallet-page">
        <Card>
          <CardHeader title="🔐 Consent Provenance Wallet" />
          <p style={{ padding: '20px', textAlign: 'center' }}>No consent data available.</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="consent-wallet-page">
      <Card>
        <CardHeader 
          title="🔐 Consent Provenance Wallet"
          badge={<Badge variant="success">{activeConsents}/{consents.length} Active Consents</Badge>}
        />

        <p className="consent-description">
          Control exactly how AI uses your data. Every permission creates a cryptographic 
          receipt you can verify.
        </p>

        <div className="consent-list">
          {consents.map((consent) => (
            <div key={consent.consentId} className="consent-item">
              <div className="consent-icon">
                {getCategoryIcon(consent.category)}
              </div>
              <div className="consent-info">
                <strong>{consent.serviceName}</strong>
                <p>{consent.serviceDescription}</p>
                {consent.status === 'granted' && consent.dataUsageCount > 0 && (
                  <span className="usage-info">
                    Used {consent.dataUsageCount} times
                    {consent.lastUsed && ` • Last used ${new Date(consent.lastUsed).toLocaleDateString()}`}
                  </span>
                )}
              </div>
              <ToggleSwitch
                checked={consent.status === 'granted'}
                onChange={(newCheckedState) => handleToggleConsent(consent.consentId, newCheckedState)}
              />
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <CardHeader 
          title="📜 Recent Consent Receipts"
          action={<Button variant="primary" size="small">Download All</Button>}
        />

        {receipts.length === 0 ? (
          <p className="no-receipts">No receipts yet. Toggle a consent to generate one!</p>
        ) : (
          <div className="receipts-list">
            {receipts.map((receipt) => (
              <div 
                key={receipt.receiptId} 
                className="receipt-item"
                onClick={() => {
                  setSelectedReceipt(receipt);
                  setShowReceiptModal(true);
                }}
              >
                <div className="receipt-header">
                  <strong>🔒 {receipt.receiptId}</strong>
                  <Badge variant={
                    receipt.blockchainStatus === 'verified' ? 'success' :
                    receipt.blockchainStatus === 'pending' ? 'warning' : 'danger'
                  }>
                    {receipt.blockchainStatus}
                  </Badge>
                </div>
                <div className="receipt-details">
                  <div>Service: {receipt.serviceName}</div>
                  <div>Action: Consent {receipt.action}</div>
                  <div>Timestamp: {new Date(receipt.timestamp).toLocaleString()}</div>
                  <div className="receipt-hash">Hash: {receipt.hash}</div>
                </div>
                {receipt.blockchainVerified && (
                  <div className="verified-badge">✓ Verified on Blockchain</div>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      <Modal
        isOpen={showReceiptModal}
        onClose={() => setShowReceiptModal(false)}
        title="Consent Receipt Details"
        size="medium"
      >
        {selectedReceipt && (
          <div className="receipt-modal-content">
            <div className="receipt-field">
              <label>Receipt ID:</label>
              <code>{selectedReceipt.receiptId}</code>
            </div>
            <div className="receipt-field">
              <label>Service:</label>
              <span>{selectedReceipt.serviceName}</span>
            </div>
            <div className="receipt-field">
              <label>Action:</label>
              <span>Consent {selectedReceipt.action}</span>
            </div>
            <div className="receipt-field">
              <label>Timestamp:</label>
              <span>{new Date(selectedReceipt.timestamp).toLocaleString()}</span>
            </div>
            <div className="receipt-field">
              <label>Cryptographic Hash:</label>
              <code className="hash-code">{selectedReceipt.hash}</code>
            </div>
            {selectedReceipt.blockNumber && (
              <div className="receipt-field">
                <label>Block Number:</label>
                <span>{selectedReceipt.blockNumber}</span>
              </div>
            )}
            {selectedReceipt.blockchainVerified && (
              <div className="verification-badge">
                ✓ Verified on Blockchain
              </div>
            )}
            <Button variant="primary" style={{ width: '100%', marginTop: '20px' }}>
              Download Receipt (PDF)
            </Button>
          </div>
        )}
      </Modal>
    </div>
  );
};

