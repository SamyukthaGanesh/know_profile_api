/**
 * GHCI API Service
 * Connects to the GHCI (AI Governance Framework) backend
 */

import type {
  Policy,
  PolicyListResponse,
  ComplianceCheckRequest,
  ComplianceCheckResponse,
  ExplainRequest,
  ExplainResponse,
  LedgerQueryRequest,
  LedgerQueryResponse,
  AuditReceipt,
  ComplianceDashboard,
  ModelHealthData,
  FairnessTrendData,
  DecisionReviewItem,
} from '../types/ghci';

// GHCI Backend URL (running on port 8001 to avoid conflict with TrustBank backend on 8000)
const GHCI_API_BASE_URL = 'http://localhost:8001';

/**
 * Fetch wrapper with error handling
 */
async function ghciFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${GHCI_API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`GHCI API Error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`GHCI API request failed: ${endpoint}`, error);
    throw error;
  }
}

// ============================================================================
// POLICY ENGINE API
// ============================================================================

export const policyApi = {
  /**
   * Get all policies
   */
  listPolicies: async (enabledOnly?: boolean): Promise<PolicyListResponse> => {
    const params = new URLSearchParams();
    if (enabledOnly !== undefined) {
      params.append('enabled_only', String(enabledOnly));
    }
    return ghciFetch<PolicyListResponse>(`/compliance/policies?${params}`);
  },

  /**
   * Get a specific policy by ID
   */
  getPolicy: async (policyId: string): Promise<Policy> => {
    return ghciFetch<Policy>(`/compliance/policies/${policyId}`);
  },

  /**
   * Enable or disable a policy
   */
  updatePolicyStatus: async (policyId: string, enabled: boolean): Promise<Policy> => {
    return ghciFetch<Policy>(`/compliance/policies/${policyId}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    });
  },

  /**
   * Check compliance for a decision
   */
  checkCompliance: async (request: ComplianceCheckRequest): Promise<ComplianceCheckResponse> => {
    return ghciFetch<ComplianceCheckResponse>('/compliance/check', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },
};

// ============================================================================
// AUDIT LOGGER (LEDGER) API
// ============================================================================

export const ledgerApi = {
  /**
   * Query audit ledger
   */
  queryLedger: async (request: LedgerQueryRequest): Promise<LedgerQueryResponse> => {
    const params = new URLSearchParams();
    if (request.start_date) params.append('start_date', request.start_date);
    if (request.end_date) params.append('end_date', request.end_date);
    if (request.policy_id) params.append('policy_id', request.policy_id);
    if (request.decision_id) params.append('decision_id', request.decision_id);
    if (request.limit) params.append('limit', String(request.limit));
    if (request.offset) params.append('offset', String(request.offset));

    return ghciFetch<LedgerQueryResponse>(`/audit/ledger?${params}`);
  },

  /**
   * Get a specific audit receipt
   */
  getReceipt: async (receiptId: string): Promise<AuditReceipt> => {
    return ghciFetch<AuditReceipt>(`/audit/receipt/${receiptId}`);
  },

  /**
   * Verify ledger integrity
   */
  verifyIntegrity: async (): Promise<{ valid: boolean; message: string }> => {
    return ghciFetch<{ valid: boolean; message: string }>('/audit/verify');
  },

  /**
   * Get blockchain visualization data
   */
  getBlockchainGraph: async (): Promise<any> => {
    return ghciFetch<any>('/blockchain/graph/compliance');
  },
};

// ============================================================================
// EXPLAINABILITY API
// ============================================================================

export const explainApi = {
  /**
   * Get SHAP explanation for a decision
   */
  explainDecision: async (request: ExplainRequest): Promise<ExplainResponse> => {
    return ghciFetch<ExplainResponse>('/explain', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  /**
   * Get natural language explanation
   */
  generateNaturalLanguage: async (
    decisionId: string,
    audienceType: 'user' | 'banker' | 'regulator'
  ): Promise<{ explanation: string }> => {
    return ghciFetch<{ explanation: string }>('/prompts/generate', {
      method: 'POST',
      body: JSON.stringify({
        decision_id: decisionId,
        audience_type: audienceType,
      }),
    });
  },
};

// ============================================================================
// DASHBOARD API
// ============================================================================

export const dashboardApi = {
  /**
   * Get model health data
   */
  getModelHealth: async (modelId?: string): Promise<any[]> => {
    const params = modelId ? `?model_id=${modelId}` : '';
    return ghciFetch<any[]>(`/dashboard/models/health${params}`);
  },

  /**
   * Get fairness trend data
   */
  getFairnessTrend: async (days: number = 7): Promise<any> => {
    return ghciFetch<any>(
      `/dashboard/charts/fairness-trend?days=${days}`
    );
  },

  /**
   * Get dashboard overview
   */
  getOverview: async (): Promise<any> => {
    return ghciFetch<any>('/dashboard/overview');
  },

  /**
   * Get compliance dashboard data
   */
  getComplianceDashboard: async (): Promise<{
    total_policies: number;
    active_policies: number;
    compliance_rate: number;
    recent_violations: Array<{
      decision_id: string;
      policy_id: string;
      policy_name: string;
      timestamp: string;
      message: string;
    }>;
    policies_by_regulation: Record<string, number>;
    audit_chain_status: string;
  }> => {
    return ghciFetch<any>('/dashboard/compliance');
  },

  /**
   * Get consent dashboard data
   */
  getConsentDashboard: async (): Promise<{
    total_users_with_consent: number;
    active_consents: number;
    revoked_consents: number;
    consent_rate: number;
    consents_by_purpose: Record<string, number>;
    recent_consent_actions: Array<{
      user_id: string;
      action: string;
      data_field: string;
      purpose: string;
      timestamp: string;
    }>;
  }> => {
    return ghciFetch<any>('/dashboard/consent');
  },
};

// ============================================================================
// DECISION REVIEW API (for Approvals Queue)
// ============================================================================

export const reviewApi = {
  /**
   * Get decisions pending review
   */
  getPendingReviews: async (): Promise<DecisionReviewItem[]> => {
    // This endpoint should be created in GHCI or we can build it from ledger data
    const ledgerData = await ledgerApi.queryLedger({
      limit: 100,
    });

    // Filter for decisions requiring review
    const pendingReviews: DecisionReviewItem[] = ledgerData.receipts
      .filter((receipt) =>
        receipt.compliance_results.some((r) => r.recommended_action === 'flag_for_review')
      )
      .map((receipt) => ({
        decision_id: receipt.decision_id,
        user_id: receipt.created_by || 'Unknown',
        timestamp: receipt.timestamp,
        decision_type: 'credit_decision',
        decision_outcome: receipt.decision_outcome || 'Unknown',
        confidence: 0.75, // Would come from model
        compliance_results: receipt.compliance_results,
        requires_review: true,
        review_reason: receipt.compliance_results
          .filter((r) => !r.compliant)
          .map((r) => r.message),
        has_explanation: true,
        top_features: Object.keys(receipt.feature_values || {}),
        review_status: 'pending' as const,
        audit_receipt_id: receipt.receipt_id,
      }));

    return pendingReviews;
  },

  /**
   * Approve a decision
   */
  approveDecision: async (
    decisionId: string,
    notes?: string
  ): Promise<{ success: boolean }> => {
    return ghciFetch<{ success: boolean }>(`/review/${decisionId}/approve`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    });
  },

  /**
   * Reject a decision
   */
  rejectDecision: async (decisionId: string, reason: string): Promise<{ success: boolean }> => {
    return ghciFetch<{ success: boolean }>(`/review/${decisionId}/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  },
};

// ============================================================================
// HEALTH CHECK
// ============================================================================

export const healthApi = {
  /**
   * Check GHCI API health
   */
  checkHealth: async (): Promise<{
    status: string;
    version: string;
    timestamp: string;
  }> => {
    return ghciFetch<{ status: string; version: string; timestamp: string }>('/health');
  },
};

// Export all APIs
export const ghciApi = {
  policy: policyApi,
  ledger: ledgerApi,
  explain: explainApi,
  dashboard: dashboardApi,
  review: reviewApi,
  health: healthApi,
};

export default ghciApi;

