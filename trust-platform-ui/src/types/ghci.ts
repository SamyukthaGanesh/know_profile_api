/**
 * GHCI (Governance, Human-in-the-loop, Compliance, Interpretability) Types
 * TypeScript interfaces for Policy Engine, Audit Logger, and SHAP Explainer
 */

// ============================================================================
// POLICY ENGINE TYPES
// ============================================================================

export enum PolicyType {
  CREDIT_RISK = 'credit_risk',
  FAIRNESS = 'fairness',
  DATA_PROTECTION = 'data_protection',
  CONSUMER_PROTECTION = 'consumer_protection',
  ANTI_DISCRIMINATION = 'anti_discrimination',
  CAPITAL_REQUIREMENT = 'capital_requirement',
  TRANSPARENCY = 'transparency',
  CUSTOM = 'custom',
}

export enum PolicyAction {
  DENY = 'deny',
  FLAG_FOR_REVIEW = 'flag_for_review',
  LOG_WARNING = 'log_warning',
  REQUIRE_EXPLANATION = 'require_explanation',
  BLOCK = 'block',
  ALERT = 'alert',
}

export enum ComplianceStatus {
  COMPLIANT = 'compliant',
  VIOLATION = 'violation',
  WARNING = 'warning',
  REQUIRES_REVIEW = 'requires_review',
  NOT_APPLICABLE = 'not_applicable',
}

export interface PolicyCondition {
  // Simple condition
  feature?: string;
  operator?: '>' | '<' | '>=' | '<=' | '==' | '!=' | 'in' | 'not_in';
  value?: any;
  
  // Compound condition
  logical_operator?: 'AND' | 'OR' | 'NOT';
  sub_conditions?: PolicyCondition[];
}

export interface Policy {
  policy_id: string;
  name: string;
  regulation_source: string; // e.g., "Basel III", "GDPR", "ECOA"
  policy_type: PolicyType;
  description: string;
  
  // The actual rule
  condition: PolicyCondition;
  action: PolicyAction;
  
  // Metadata
  version: string;
  effective_date?: string;
  expiry_date?: string;
  jurisdiction?: string;
  priority: number; // Higher priority checked first
  enabled: boolean;
  
  // Additional context
  rationale?: string;
  references?: string[];
  tags: string[];
  
  // Audit trail
  created_at: string;
  created_by?: string;
  updated_at?: string;
  updated_by?: string;
}

export interface ComplianceResult {
  policy_id: string;
  policy_name: string;
  regulation_source: string;
  status: ComplianceStatus;
  compliant: boolean;
  message: string;
  recommended_action?: string;
  timestamp: string;
}

// ============================================================================
// AUDIT LOGGER (LEDGER) TYPES
// ============================================================================

export interface AuditReceipt {
  receipt_id: string;
  decision_id: string;
  timestamp: string;
  
  // What was checked
  policies_checked: string[];
  compliance_results: ComplianceResult[];
  
  // Decision details
  decision_outcome?: string;
  feature_values?: Record<string, any>;
  model_id?: string;
  
  // Cryptographic verification
  content_hash: string; // SHA256 of all content
  previous_hash: string; // Hash of previous receipt (blockchain-style)
  signature?: string; // Digital signature
  
  // Metadata
  created_by?: string;
  ip_address?: string;
  user_agent?: string;
}

export interface LedgerStats {
  total_receipts: number;
  receipts_today: number;
  chain_integrity: boolean; // Whether hash chain is valid
  violations_detected: number;
  last_receipt_timestamp?: string;
}

// ============================================================================
// SHAP EXPLAINER TYPES
// ============================================================================

export interface SHAPExplanation {
  explanation_id: string;
  decision_id: string;
  model_id: string;
  timestamp: string;
  
  // SHAP values
  shap_values: Record<string, number>; // Feature -> SHAP value
  base_value: number; // Expected value
  prediction: number; // Model prediction
  
  // Feature importance
  feature_importance: Array<{
    feature: string;
    value: number;
    shap_value: number;
    importance: number; // Absolute SHAP value
  }>;
  
  // Top features
  top_positive_features: string[]; // Features pushing prediction up
  top_negative_features: string[]; // Features pushing prediction down
  
  // Metadata
  method: 'tree' | 'linear' | 'deep' | 'kernel';
  n_features: number;
}

export interface ExplanationResult {
  explanation_type: 'shap' | 'lime' | 'anchors' | 'integrated_gradients';
  explanation: SHAPExplanation | any;
  natural_language: string; // Human-readable explanation
  confidence: number;
  timestamp: string;
}

// ============================================================================
// DASHBOARD DATA TYPES
// ============================================================================

export interface ComplianceDashboard {
  total_checks: number;
  compliant_decisions: number;
  violations: number;
  warnings: number;
  compliance_rate: number;
  
  policy_violations: Array<{
    policy_id: string;
    policy_name: string;
    violation_count: number;
    last_violation: string;
  }>;
  
  recent_violations: Array<{
    violation_id: string;
    policy_id: string;
    policy_name: string;
    timestamp: string;
    severity: 'low' | 'medium' | 'high';
    status: 'pending' | 'resolved' | 'escalated';
  }>;
  
  timestamp: string;
}

export interface ModelHealthData {
  model_id: string;
  model_name: string;
  model_type: string;
  
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  
  data_drift_score: number;
  concept_drift_detected: boolean;
  
  predictions_today: number;
  predictions_total: number;
  
  last_retrained: string;
  next_retraining_scheduled: string;
  
  status: 'healthy' | 'degraded' | 'critical';
  alerts: string[];
}

export interface FairnessTrendData {
  metric_name: string;
  data_points: Array<{
    timestamp: string;
    value: number;
    threshold: number;
  }>;
  current_value: number;
  trend: 'improving' | 'stable' | 'degrading';
}

// ============================================================================
// API REQUEST/RESPONSE TYPES
// ============================================================================

export interface ComplianceCheckRequest {
  decision_id: string;
  feature_values: Record<string, any>;
  decision_outcome?: string;
  model_id?: string;
  user_id?: string;
}

export interface ComplianceCheckResponse {
  decision_id: string;
  compliant: boolean;
  compliance_rate: number;
  results: ComplianceResult[];
  audit_receipt: AuditReceipt;
  actions_required: PolicyAction[];
}

export interface ExplainRequest {
  decision_id: string;
  model_id: string;
  feature_values: Record<string, any>;
  prediction: number;
  method?: 'shap' | 'lime' | 'anchors';
}

export interface ExplainResponse {
  decision_id: string;
  explanation_id: string;
  method: string;
  explanation: SHAPExplanation;
  natural_language: string;
  visualization_data?: any;
  timestamp: string;
}

export interface PolicyListResponse {
  policies: Policy[];
  total: number;
  enabled: number;
  disabled: number;
}

export interface LedgerQueryRequest {
  start_date?: string;
  end_date?: string;
  policy_id?: string;
  decision_id?: string;
  limit?: number;
  offset?: number;
}

export interface LedgerQueryResponse {
  receipts: AuditReceipt[];
  total: number;
  stats: LedgerStats;
  chain_valid: boolean;
}

// ============================================================================
// ADMIN UI TYPES
// ============================================================================

export interface PolicyFilter {
  policy_type?: PolicyType;
  regulation_source?: string;
  enabled?: boolean;
  search?: string;
}

export interface LedgerFilter {
  date_range?: {
    start: string;
    end: string;
  };
  compliance_status?: ComplianceStatus;
  policy_id?: string;
  search?: string;
}

export interface DecisionReviewItem {
  decision_id: string;
  user_id: string;
  timestamp: string;
  
  // Decision details
  decision_type: string;
  decision_outcome: string;
  confidence: number;
  
  // Compliance
  compliance_results: ComplianceResult[];
  requires_review: boolean;
  review_reason: string[];
  
  // Explanation
  has_explanation: boolean;
  top_features: string[];
  
  // Review status
  review_status: 'pending' | 'approved' | 'rejected' | 'escalated';
  reviewed_by?: string;
  reviewed_at?: string;
  review_notes?: string;
  
  // Audit
  audit_receipt_id: string;
}

