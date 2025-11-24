"""
Blockchain Visualization Endpoints
Provides blockchain-style visualization of audit and consent chains.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/blockchain", tags=["Blockchain Visualization"])


# ============================================================
# RESPONSE MODELS
# ============================================================

class BlockResponse(BaseModel):
    """Individual block in the chain"""
    block_number: int
    block_id: str
    timestamp: str
    content_hash: str
    previous_hash: str
    data: Dict[str, Any]
    is_valid: bool


class ChainResponse(BaseModel):
    """Complete chain information"""
    chain_type: str  # "compliance" or "consent"
    total_blocks: int
    chain_valid: bool
    blocks: List[BlockResponse]
    chain_metadata: Dict[str, Any]


class BlockVerificationResponse(BaseModel):
    """Block verification result"""
    block_id: str
    is_valid: bool
    hash_matches: bool
    chain_link_valid: bool
    verification_timestamp: str
    errors: List[str]


# ============================================================
# BLOCKCHAIN VISUALIZATION ENDPOINTS
# ============================================================

@router.get("/compliance/blocks", response_model=ChainResponse)
async def get_compliance_blockchain(
    limit: Optional[int] = 50,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None
):
    """
    Get compliance audit chain as blockchain visualization.
    
    Args:
        limit: Maximum number of blocks to return
        start_block: Starting block number
        end_block: Ending block number
        
    Returns:
        Blockchain visualization data
    """
    try:
        from core.compliance.audit_logger import AuditLogger
        
        # Initialize audit logger
        audit_logger = AuditLogger()
        
        # Get receipts (blocks)
        receipts = audit_logger.receipt_chain
        
        # Apply filters
        if start_block is not None:
            receipts = receipts[start_block:]
        if end_block is not None:
            receipts = receipts[:end_block]
        if limit:
            receipts = receipts[-limit:]  # Most recent
        
        # Verify chain
        chain_valid, errors = audit_logger.verify_chain()
        
        # Convert receipts to blocks
        blocks = []
        for i, receipt in enumerate(receipts):
            block = BlockResponse(
                block_number=i,
                block_id=receipt.receipt_id,
                timestamp=receipt.timestamp,
                content_hash=receipt.content_hash,
                previous_hash=receipt.previous_hash,
                data={
                    'decision_id': receipt.decision_id,
                    'policies_checked': receipt.policies_checked,
                    'is_compliant': all(r.compliant for r in receipt.compliance_results),
                    'violations_count': sum(1 for r in receipt.compliance_results if not r.compliant),
                    'model_id': receipt.model_id,
                    'created_by': receipt.created_by
                },
                is_valid=audit_logger.verify_receipt(receipt)
            )
            blocks.append(block)
        
        # Get statistics
        stats = audit_logger.get_statistics()
        
        return ChainResponse(
            chain_type="compliance",
            total_blocks=len(audit_logger.receipt_chain),
            chain_valid=chain_valid,
            blocks=blocks,
            chain_metadata={
                'total_decisions': stats['unique_decisions'],
                'compliant_blocks': stats['compliant_receipts'],
                'non_compliant_blocks': stats['non_compliant_receipts'],
                'earliest_block': stats.get('earliest_receipt'),
                'latest_block': stats.get('latest_receipt'),
                'chain_errors': errors if not chain_valid else []
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting compliance blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consent/blocks/{user_id}", response_model=ChainResponse)
async def get_consent_blockchain(
    user_id: str,
    limit: Optional[int] = 50
):
    """
    Get user's consent chain as blockchain visualization.
    
    Args:
        user_id: User identifier
        limit: Maximum number of blocks
        
    Returns:
        Blockchain visualization for user's consent history
    """
    try:
        from core.consent.consent_wallet import ConsentWallet
        
        # Load user's wallet
        wallet = ConsentWallet(user_id=user_id)
        
        # Get receipts (blocks)
        receipts = wallet.receipt_chain
        if limit:
            receipts = receipts[-limit:]
        
        # Verify wallet
        chain_valid, errors = wallet.verify_wallet()
        
        # Convert receipts to blocks
        blocks = []
        for i, receipt in enumerate(receipts):
            block = BlockResponse(
                block_number=i,
                block_id=receipt.receipt_id,
                timestamp=receipt.timestamp,
                content_hash=receipt.content_hash,
                previous_hash=receipt.previous_hash,
                data={
                    'action': receipt.action.value,
                    'consent_records': receipt.consent_records,
                    'action_details': receipt.action_details,
                    'initiated_by': receipt.initiated_by
                },
                is_valid=wallet.verify_receipt(receipt)
            )
            blocks.append(block)
        
        # Get summary
        summary = wallet.get_summary()
        
        return ChainResponse(
            chain_type="consent",
            total_blocks=len(wallet.receipt_chain),
            chain_valid=chain_valid,
            blocks=blocks,
            chain_metadata={
                'user_id': user_id,
                'total_consents': summary.total_consents,
                'active_consents': summary.active_consents,
                'revoked_consents': summary.revoked_consents,
                'chain_errors': errors if not chain_valid else []
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting consent blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verify/{block_id}", response_model=BlockVerificationResponse)
async def verify_block(
    block_id: str,
    chain_type: str = "compliance"
):
    """
    Verify specific block integrity.
    
    Args:
        block_id: Block/receipt ID to verify
        chain_type: "compliance" or "consent"
        
    Returns:
        Verification result
    """
    try:
        errors = []
        
        if chain_type == "compliance":
            from core.compliance.audit_logger import AuditLogger
            
            audit_logger = AuditLogger()
            receipt = audit_logger.get_receipt(block_id)
            
            if not receipt:
                raise HTTPException(status_code=404, detail=f"Block '{block_id}' not found")
            
            # Verify hash
            hash_matches = audit_logger.verify_receipt(receipt)
            
            # Verify chain link
            chain_link_valid = True
            receipt_idx = None
            
            for i, r in enumerate(audit_logger.receipt_chain):
                if r.receipt_id == block_id:
                    receipt_idx = i
                    break
            
            if receipt_idx is not None and receipt_idx > 0:
                expected_previous = audit_logger.receipt_chain[receipt_idx - 1].content_hash
                if receipt.previous_hash != expected_previous:
                    chain_link_valid = False
                    errors.append("Chain link broken - previous hash doesn't match")
        
        else:  # consent
            # For consent, would need user_id - simplified for now
            hash_matches = True
            chain_link_valid = True
            errors.append("Consent block verification requires user_id")
        
        is_valid = hash_matches and chain_link_valid
        
        return BlockVerificationResponse(
            block_id=block_id,
            is_valid=is_valid,
            hash_matches=hash_matches,
            chain_link_valid=chain_link_valid,
            verification_timestamp=datetime.now().isoformat(),
            errors=errors
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying block: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/compliance")
async def get_compliance_chain_graph():
    """
    Get blockchain graph data for visualization (nodes and edges).
    
    Returns data formatted for graph libraries like D3.js, vis.js, or cytoscape.
    """
    try:
        from core.compliance.audit_logger import AuditLogger
        
        audit_logger = AuditLogger()
        receipts = audit_logger.receipt_chain[-20:]  # Last 20 blocks
        
        nodes = []
        edges = []
        
        for i, receipt in enumerate(receipts):
            # Create node
            is_compliant = all(r.compliant for r in receipt.compliance_results)
            
            node = {
                'id': receipt.receipt_id,
                'label': f"Block {i}\n{receipt.decision_id}",
                'timestamp': receipt.timestamp,
                'hash': receipt.content_hash[:16] + "...",
                'compliant': is_compliant,
                'color': '#4CAF50' if is_compliant else '#F44336',
                'size': 30,
                'data': {
                    'decision_id': receipt.decision_id,
                    'policies_checked': len(receipt.policies_checked),
                    'violations': sum(1 for r in receipt.compliance_results if not r.compliant)
                }
            }
            nodes.append(node)
            
            # Create edge to previous block
            if i > 0:
                edge = {
                    'from': receipts[i-1].receipt_id,
                    'to': receipt.receipt_id,
                    'label': 'hash link',
                    'color': '#666',
                    'arrows': 'to'
                }
                edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_blocks': len(receipts),
                'chain_valid': audit_logger.verify_chain()[0]
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating compliance graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/consent/{user_id}")
async def get_consent_chain_graph(user_id: str):
    """
    Get user's consent blockchain graph.
    
    Args:
        user_id: User identifier
        
    Returns:
        Graph data for visualization
    """
    try:
        from core.consent.consent_wallet import ConsentWallet
        
        wallet = ConsentWallet(user_id=user_id)
        receipts = wallet.receipt_chain
        
        nodes = []
        edges = []
        
        for i, receipt in enumerate(receipts):
            # Create node
            node = {
                'id': receipt.receipt_id,
                'label': f"Block {i}\n{receipt.action.value}",
                'timestamp': receipt.timestamp,
                'hash': receipt.content_hash[:16] + "...",
                'action': receipt.action.value,
                'color': '#2196F3' if receipt.action.value == 'grant' else '#F44336',
                'size': 30,
                'data': {
                    'action': receipt.action.value,
                    'consents_affected': len(receipt.consent_records),
                    'initiated_by': receipt.initiated_by
                }
            }
            nodes.append(node)
            
            # Create edge
            if i > 0:
                edge = {
                    'from': receipts[i-1].receipt_id,
                    'to': receipt.receipt_id,
                    'label': 'hash link',
                    'color': '#666',
                    'arrows': 'to'
                }
                edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'user_id': user_id,
                'total_blocks': len(receipts),
                'chain_valid': wallet.verify_wallet()[0]
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating consent graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/compliance-chain")
async def get_compliance_chain_stats():
    """
    Get detailed statistics about compliance audit chain.
    """
    try:
        from core.compliance.audit_logger import AuditLogger
        
        audit_logger = AuditLogger()
        
        # Get all receipts
        receipts = audit_logger.receipt_chain
        
        # Calculate statistics
        total_blocks = len(receipts)
        
        # Block creation rate (blocks per day)
        if total_blocks > 1:
            first_timestamp = datetime.fromisoformat(receipts[0].timestamp)
            last_timestamp = datetime.fromisoformat(receipts[-1].timestamp)
            days_diff = (last_timestamp - first_timestamp).days + 1
            blocks_per_day = total_blocks / days_diff if days_diff > 0 else 0
        else:
            blocks_per_day = 0
        
        # Hash distribution (first char of hash)
        hash_distribution = {}
        for receipt in receipts:
            first_char = receipt.content_hash[0]
            hash_distribution[first_char] = hash_distribution.get(first_char, 0) + 1
        
        # Verify chain
        chain_valid, errors = audit_logger.verify_chain()
        
        return {
            'total_blocks': total_blocks,
            'chain_valid': chain_valid,
            'blocks_per_day': round(blocks_per_day, 2),
            'hash_distribution': hash_distribution,
            'chain_integrity': {
                'valid_blocks': sum(1 for r in receipts if audit_logger.verify_receipt(r)),
                'broken_links': len(errors),
                'errors': errors
            },
            'oldest_block': receipts[0].timestamp if receipts else None,
            'newest_block': receipts[-1].timestamp if receipts else None
        }
    
    except Exception as e:
        logger.error(f"Error getting chain stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explorer/block/{block_id}")
async def explore_block(
    block_id: str,
    chain_type: str = "compliance"
):
    """
    Block explorer - get detailed information about a specific block.
    
    Args:
        block_id: Block/receipt ID
        chain_type: "compliance" or "consent"
        
    Returns:
        Detailed block information
    """
    try:
        if chain_type == "compliance":
            from core.compliance.audit_logger import AuditLogger
            
            audit_logger = AuditLogger()
            receipt = audit_logger.get_receipt(block_id)
            
            if not receipt:
                raise HTTPException(status_code=404, detail=f"Block '{block_id}' not found")
            
            # Find block number
            block_number = None
            for i, r in enumerate(audit_logger.receipt_chain):
                if r.receipt_id == block_id:
                    block_number = i
                    break
            
            # Get next block
            next_block = None
            if block_number is not None and block_number < len(audit_logger.receipt_chain) - 1:
                next_block = audit_logger.receipt_chain[block_number + 1].receipt_id
            
            # Verify
            is_valid = audit_logger.verify_receipt(receipt)
            
            return {
                'block_number': block_number,
                'block_id': receipt.receipt_id,
                'timestamp': receipt.timestamp,
                'hashes': {
                    'current': receipt.content_hash,
                    'previous': receipt.previous_hash,
                    'signature': receipt.signature
                },
                'data': {
                    'decision_id': receipt.decision_id,
                    'policies_checked': receipt.policies_checked,
                    'compliance_results': [
                        {
                            'policy_id': r.policy.policy_id,
                            'policy_name': r.policy.name,
                            'compliant': r.compliant,
                            'message': r.message
                        }
                        for r in receipt.compliance_results
                    ],
                    'decision_outcome': receipt.decision_outcome,
                    'model_id': receipt.model_id
                },
                'metadata': {
                    'created_by': receipt.created_by,
                    'ip_address': receipt.ip_address,
                    'user_agent': receipt.user_agent
                },
                'chain_links': {
                    'previous_block': receipt.previous_hash[:16] + "..." if receipt.previous_hash else None,
                    'next_block': next_block
                },
                'verification': {
                    'is_valid': is_valid,
                    'verified_at': datetime.now().isoformat()
                }
            }
        
        else:
            raise HTTPException(status_code=400, detail="Consent block explorer requires user_id")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exploring block: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/compliance")
async def get_compliance_timeline(
    days: int = 7,
    include_violations_only: bool = False
):
    """
    Get compliance timeline view (chronological events).
    
    Args:
        days: Number of days to show
        include_violations_only: Only show blocks with violations
        
    Returns:
        Timeline data for visualization
    """
    try:
        from core.compliance.audit_logger import AuditLogger
        from datetime import timedelta
        
        audit_logger = AuditLogger()
        
        # Filter by date
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        receipts = [r for r in audit_logger.receipt_chain if r.timestamp >= cutoff]
        
        # Filter by violations if requested
        if include_violations_only:
            receipts = [
                r for r in receipts 
                if any(not cr.compliant for cr in r.compliance_results)
            ]
        
        # Create timeline events
        events = []
        for receipt in receipts:
            violations = [r for r in receipt.compliance_results if not r.compliant]
            
            event = {
                'timestamp': receipt.timestamp,
                'block_id': receipt.receipt_id,
                'decision_id': receipt.decision_id,
                'type': 'violation' if violations else 'compliant',
                'policies_checked': len(receipt.policies_checked),
                'violations_count': len(violations),
                'violations': [
                    {
                        'policy_id': v.policy.policy_id,
                        'policy_name': v.policy.name,
                        'message': v.message
                    }
                    for v in violations
                ] if violations else []
            }
            events.append(event)
        
        return {
            'events': events,
            'total_events': len(events),
            'time_range': {
                'start': cutoff,
                'end': datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting compliance timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))