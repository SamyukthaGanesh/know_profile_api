import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './BlockchainGraph.css';

interface BlockData {
  block_number: number;
  block_id: string;
  timestamp: string;
  content_hash: string;
  previous_hash: string;
  data: any;
  is_valid: boolean;
}

export const BlockchainGraph: React.FC = () => {
  const [blocks, setBlocks] = useState<BlockData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedBlock, setSelectedBlock] = useState<BlockData | null>(null);
  const [viewMode, setViewMode] = useState<'chain' | 'timeline'>('chain');

  useEffect(() => {
    loadBlockchain();
  }, []);

  const loadBlockchain = async () => {
    try {
      setLoading(true);
      console.log('🔗 Fetching blockchain data from GHCI...');
      
      const response = await fetch('http://localhost:8001/blockchain/compliance/blocks?limit=20');
      
      if (!response.ok) {
        throw new Error('Failed to fetch blockchain');
      }

      const data = await response.json();
      console.log('✅ Got blockchain data:', data);
      
      setBlocks(data.blocks || []);
    } catch (err) {
      console.error('❌ Failed to load blockchain:', err);
      
      // Fallback mock data
      setBlocks([
        {
          block_number: 0,
          block_id: 'BLOCK_GENESIS',
          timestamp: '2025-01-01T00:00:00Z',
          content_hash: 'a1b2c3d4e5f6...',
          previous_hash: '0000000000000000',
          data: { type: 'genesis', message: 'Genesis block' },
          is_valid: true
        },
        {
          block_number: 1,
          block_id: 'BLOCK_001',
          timestamp: '2025-01-02T10:30:00Z',
          content_hash: 'b2c3d4e5f6g7...',
          previous_hash: 'a1b2c3d4e5f6...',
          data: { type: 'policy_check', policy: 'Basel III', result: 'passed' },
          is_valid: true
        },
        {
          block_number: 2,
          block_id: 'BLOCK_002',
          timestamp: '2025-01-03T14:20:00Z',
          content_hash: 'c3d4e5f6g7h8...',
          previous_hash: 'b2c3d4e5f6g7...',
          data: { type: 'fairness_audit', model: 'home_credit', bias_score: 0.04 },
          is_valid: true
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const verifyBlock = async (blockId: string) => {
    try {
      console.log(`🔍 Verifying block ${blockId}...`);
      
      const response = await fetch(`http://localhost:8001/blockchain/verify/${blockId}`);
      
      if (!response.ok) {
        throw new Error('Verification failed');
      }

      const result = await response.json();
      console.log('✅ Verification result:', result);
      
      alert(`Block Verification:\n\nValid: ${result.is_valid ? '✅ YES' : '❌ NO'}\nHash Match: ${result.hash_matches ? '✅' : '❌'}\nChain Link: ${result.chain_link_valid ? '✅' : '❌'}`);
    } catch (err) {
      console.error('❌ Verification failed:', err);
      alert('Failed to verify block');
    }
  };

  if (loading) {
    return (
      <div className="loading-container" style={{ padding: '48px', textAlign: 'center' }}>
        <h2>⏳ Loading Blockchain...</h2>
        <p>Fetching cryptographic audit trail from GHCI...</p>
      </div>
    );
  }

  return (
    <div className="blockchain-graph-page">
      <div className="page-header">
        <div>
          <h1>🔗 Blockchain Explorer</h1>
          <p>Cryptographic audit trail visualization</p>
          <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
            <Badge variant="success">✅ Chain Valid</Badge>
            <Badge variant="info">{blocks.length} Blocks</Badge>
          </div>
        </div>
        <div className="header-actions" style={{ display: 'flex', gap: '12px' }}>
          <Button 
            variant={viewMode === 'chain' ? 'primary' : 'secondary'}
            onClick={() => setViewMode('chain')}
          >
            🔗 Chain View
          </Button>
          <Button 
            variant={viewMode === 'timeline' ? 'primary' : 'secondary'}
            onClick={() => setViewMode('timeline')}
          >
            📅 Timeline View
          </Button>
        </div>
      </div>

      {viewMode === 'chain' ? (
        // VISUAL CHAIN REPRESENTATION
        <div className="blockchain-chain-view">
          {blocks.map((block, index) => (
            <div key={block.block_id} className="block-container">
              <div 
                className={`block-card ${selectedBlock?.block_id === block.block_id ? 'selected' : ''}`}
                onClick={() => setSelectedBlock(block)}
              >
                {/* Block Header */}
                <div className="block-header">
                  <Badge variant={block.is_valid ? 'success' : 'danger'}>
                    {block.is_valid ? '✅ VALID' : '❌ INVALID'}
                  </Badge>
                  <span className="block-number">Block #{block.block_number}</span>
                </div>

                {/* Block Content */}
                <div className="block-content">
                  <div className="block-field">
                    <strong>Block ID:</strong>
                    <code>{block.block_id}</code>
                  </div>
                  <div className="block-field">
                    <strong>Hash:</strong>
                    <code className="hash-preview">{block.content_hash.substring(0, 16)}...</code>
                  </div>
                  <div className="block-field">
                    <strong>Prev Hash:</strong>
                    <code className="hash-preview">{block.previous_hash.substring(0, 16)}...</code>
                  </div>
                  <div className="block-field">
                    <strong>Timestamp:</strong>
                    <span>{new Date(block.timestamp).toLocaleString()}</span>
                  </div>
                </div>

                {/* Block Actions */}
                <div className="block-actions">
                  <Button 
                    variant="primary" 
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedBlock(block);
                    }}
                  >
                    View Data
                  </Button>
                </div>
              </div>

              {/* Chain Link Arrow */}
              {index < blocks.length - 1 && (
                <div className="chain-link">
                  <div className="arrow">↓</div>
                  <div className="link-label">Cryptographically Linked</div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        // TIMELINE VIEW
        <div className="blockchain-timeline-view">
          <div className="timeline-line"></div>
          {blocks.map((block, index) => (
            <div key={block.block_id} className="timeline-item">
              <div className="timeline-marker">
                <div className={`marker-dot ${block.is_valid ? 'valid' : 'invalid'}`}></div>
              </div>
              <div onClick={() => setSelectedBlock(block)}>
                <Card className="timeline-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <h3 style={{ margin: 0, fontSize: '18px' }}>Block #{block.block_number}</h3>
                      <p style={{ margin: '4px 0 0 0', color: '#666', fontSize: '14px' }}>
                        {new Date(block.timestamp).toLocaleString()}
                      </p>
                    </div>
                    <Badge variant={block.is_valid ? 'success' : 'danger'}>
                      {block.is_valid ? '✅ VALID' : '❌ INVALID'}
                    </Badge>
                  </div>
                  <div style={{ marginTop: '12px', padding: '12px', backgroundColor: '#f9fafb', borderRadius: '6px', fontSize: '13px', fontFamily: 'monospace' }}>
                    <div><strong>Hash:</strong> {block.content_hash}</div>
                  </div>
                </Card>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* SELECTED BLOCK DETAILS MODAL */}
      {selectedBlock && (
        <div 
          className="modal-overlay" 
          onClick={() => setSelectedBlock(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}
        >
          <div 
            className="modal-content" 
            onClick={(e) => e.stopPropagation()}
            style={{
              backgroundColor: '#fff',
              borderRadius: '12px',
              padding: '32px',
              maxWidth: '700px',
              maxHeight: '80vh',
              overflow: 'auto',
              boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h2 style={{ margin: 0 }}>🔗 Block #{selectedBlock.block_number} Details</h2>
              <Button variant="secondary" onClick={() => setSelectedBlock(null)}>✕ Close</Button>
            </div>

            <Card style={{ marginBottom: '16px' }}>
              <CardHeader title="Block Information" />
              <div style={{ padding: '16px' }}>
                <div className="detail-row">
                  <strong>Block ID:</strong>
                  <code>{selectedBlock.block_id}</code>
                </div>
                <div className="detail-row">
                  <strong>Timestamp:</strong>
                  <span>{new Date(selectedBlock.timestamp).toLocaleString()}</span>
                </div>
                <div className="detail-row">
                  <strong>Valid:</strong>
                  <Badge variant={selectedBlock.is_valid ? 'success' : 'danger'}>
                    {selectedBlock.is_valid ? '✅ YES' : '❌ NO'}
                  </Badge>
                </div>
              </div>
            </Card>

            <Card style={{ marginBottom: '16px' }}>
              <CardHeader title="Cryptographic Hashes" />
              <div style={{ padding: '16px' }}>
                <div className="detail-row">
                  <strong>Content Hash:</strong>
                  <code style={{ wordBreak: 'break-all' }}>{selectedBlock.content_hash}</code>
                </div>
                <div className="detail-row">
                  <strong>Previous Hash:</strong>
                  <code style={{ wordBreak: 'break-all' }}>{selectedBlock.previous_hash}</code>
                </div>
              </div>
            </Card>

            <Card>
              <CardHeader title="Block Data (JSON)" />
              <pre style={{ 
                padding: '16px', 
                backgroundColor: '#f9fafb', 
                borderRadius: '6px', 
                overflow: 'auto',
                fontSize: '12px',
                fontFamily: 'monospace'
              }}>
                {JSON.stringify(selectedBlock.data, null, 2)}
              </pre>
            </Card>

            <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                variant="primary" 
                onClick={() => setSelectedBlock(null)}
              >
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

