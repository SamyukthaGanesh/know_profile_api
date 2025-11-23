import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import './BlockchainExplorer.css';

interface Block {
  block_id: string;
  block_number: number;
  timestamp: string;
  previous_hash: string;
  current_hash: string;
  data: any;
  chain_type: 'compliance' | 'consent';
}

export const BlockchainExplorer: React.FC = () => {
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedBlock, setSelectedBlock] = useState<Block | null>(null);
  const [chainType, setChainType] = useState<'compliance' | 'consent'>('compliance');
  const [verifying, setVerifying] = useState(false);
  const [chainValid, setChainValid] = useState<boolean | null>(null);

  useEffect(() => {
    loadBlocks();
  }, [chainType]);

  const loadBlocks = async () => {
    setLoading(true);
    try {
      const endpoint = chainType === 'compliance' 
        ? 'http://localhost:8001/blockchain/compliance/blocks'
        : 'http://localhost:8001/blockchain/consent/blocks/all';
      
      const response = await fetch(endpoint);
      if (response.ok) {
        const data = await response.json();
        setBlocks(data.blocks || data || []);
        console.log(`✅ Loaded ${chainType} blockchain:`, data);
      } else {
        console.warn('⚠️ Using mock blockchain data');
        // Mock data
        setBlocks([
          {
            block_id: 'BLOCK_001',
            block_number: 1,
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            previous_hash: '0000000000000000',
            current_hash: 'a7f5c8d9e1b2a3c4',
            data: { event: 'genesis_block', description: 'Blockchain initialized' },
            chain_type: chainType
          }
        ]);
      }
    } catch (error) {
      console.error('Failed to load blocks:', error);
      setBlocks([]);
    } finally {
      setLoading(false);
    }
  };

  const verifyBlockchain = async () => {
    setVerifying(true);
    try {
      const response = await fetch('http://localhost:8001/compliance/audit/verify');
      if (response.ok) {
        const result = await response.json();
        setChainValid(result.valid || result.is_valid);
        alert(`🔐 Blockchain Verification\n\n${result.valid ? '✅ Chain is VALID' : '❌ Chain is INVALID'}\n\nTotal Blocks: ${result.total_blocks || blocks.length}`);
      } else {
        alert('⚠️ Verification service unavailable');
      }
    } catch (error) {
      console.error('Error verifying blockchain:', error);
      alert('⚠️ Could not verify blockchain');
    } finally {
      setVerifying(false);
    }
  };

  const viewBlockStats = async () => {
    try {
      const response = await fetch('http://localhost:8001/blockchain/stats/compliance-chain');
      if (response.ok) {
        const stats = await response.json();
        alert(`📊 Blockchain Statistics\n\nTotal Blocks: ${stats.total_blocks}\nChain Length: ${stats.chain_length}\nGenesis Time: ${stats.genesis_timestamp}\nLast Block: ${stats.latest_block_time}`);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  if (loading) {
    return <div className="loading">Loading blockchain...</div>;
  }

  return (
    <div className="blockchain-explorer">
      <div className="page-header">
        <div>
          <h1>🔗 Blockchain Explorer</h1>
          <p>Explore and verify cryptographic audit trail</p>
        </div>
        <div className="header-actions">
          {/* Actions removed per user request */}
        </div>
      </div>

      <div className="chain-selector">
        <Button 
          variant={chainType === 'compliance' ? 'primary' : 'secondary'}
          onClick={() => setChainType('compliance')}
        >
          📜 Compliance Chain
        </Button>
        <Button 
          variant={chainType === 'consent' ? 'primary' : 'secondary'}
          onClick={() => setChainType('consent')}
        >
          🔐 Consent Chain
        </Button>
      </div>

      <div className="explorer-content">
        <Card className="blocks-list-card">
          <CardHeader title={`🔗 ${chainType === 'compliance' ? 'Compliance' : 'Consent'} Blocks`} />
          <div className="blocks-list">
            {blocks.length === 0 ? (
              <div className="no-blocks">
                <p>No blocks found</p>
                <p>Blockchain is empty or not initialized</p>
              </div>
            ) : (
              blocks.map((block) => (
                <div
                  key={block.block_id}
                  className={`block-card ${selectedBlock?.block_id === block.block_id ? 'selected' : ''}`}
                  onClick={() => setSelectedBlock(block)}
                >
                  <div className="block-header">
                    <span className="block-number">Block #{block.block_number}</span>
                    <Badge variant="info">{block.chain_type}</Badge>
                  </div>
                  <div className="block-hash">
                    <span className="hash-label">Hash:</span>
                    <span className="hash-value">{block.current_hash?.slice(0, 16)}...</span>
                  </div>
                  <div className="block-time">
                    {new Date(block.timestamp).toLocaleString()}
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        <Card className="block-details-card">
          <CardHeader title="🔍 Block Details" />
          {selectedBlock ? (
            <div className="block-details">
              <div className="detail-section">
                <h3>Block Information</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="detail-label">Block ID:</span>
                    <span className="detail-value">{selectedBlock.block_id}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Block Number:</span>
                    <span className="detail-value">#{selectedBlock.block_number}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Timestamp:</span>
                    <span className="detail-value">
                      {new Date(selectedBlock.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Chain Type:</span>
                    <span className="detail-value">{selectedBlock.chain_type}</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3>Cryptographic Hashes</h3>
                <div className="hash-display">
                  <div className="hash-item">
                    <span className="hash-label">Previous Hash:</span>
                    <code className="hash-code">{selectedBlock.previous_hash}</code>
                  </div>
                  <div className="hash-item">
                    <span className="hash-label">Current Hash:</span>
                    <code className="hash-code">{selectedBlock.current_hash}</code>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3>Block Data</h3>
                <pre className="block-data-json">
                  {JSON.stringify(selectedBlock.data, null, 2)}
                </pre>
              </div>

              <div className="block-actions">
                <Button variant="secondary" onClick={() => {
                  navigator.clipboard.writeText(selectedBlock.current_hash);
                  alert('✅ Hash copied to clipboard!');
                }}>
                  📋 Copy Hash
                </Button>
                <Button variant="secondary" onClick={async () => {
                  try {
                    const response = await fetch(`http://localhost:8001/blockchain/verify/${selectedBlock.block_id}`);
                    if (response.ok) {
                      const result = await response.json();
                      alert(`🔐 Block Verification\n\n${result.valid ? '✅ Block is VALID' : '❌ Block is INVALID'}`);
                    }
                  } catch (error) {
                    alert('⚠️ Verification failed');
                  }
                }}>
                  🔐 Verify Block
                </Button>
              </div>
            </div>
          ) : (
            <div className="no-selection">
              <p>👈 Select a block to view details</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

