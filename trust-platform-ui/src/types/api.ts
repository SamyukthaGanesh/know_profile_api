// API Types based on specification

// ============================================
// USER DASHBOARD TYPES
// ============================================

export interface UserDashboardResponse {
  user: {
    userId: string;
    name: string;
    lastLogin: string;
    profilePictureUrl?: string;
  };
  trustScore: {
    overall: number;
    components: {
      accuracy: number;
      fairness: number;
      transparency: number;
      privacy: number;
      explainability: number;
      compliance: number;
    };
    lastUpdated: string;
    trend: 'up' | 'down' | 'stable';
  };
  activeLoanApplication?: {
    applicationId: string;
    amount: number;
    currency: string;
    status: 'approved' | 'denied' | 'under_review' | 'pending';
    submittedAt: string;
    confidence: number;
    riskLevel: 'low' | 'medium' | 'high';
  };
  quickStats: {
    activeConsents: number;
    fairnessRating: number;
    pendingActions: number;
  };
}

export interface ExplanationFactor {
  featureName: string;
  humanReadableName: string;
  description: string;
  value: number | string;
  impact: 'positive' | 'negative';
  impactStrength: 'strong' | 'moderate' | 'weak';
  impactPercentage: number;
  shapValue: number;
}

export interface ExplanationResponse {
  applicationId: string;
  decision: {
    outcome: 'approved' | 'denied' | 'under_review';
    confidence: number;
    timestamp: string;
  };
  explanation: {
    summary: string;
    literacyLevel: 'beginner' | 'intermediate' | 'advanced';
    mainReasons: string;
    factors: ExplanationFactor[];
    nextSteps: string | string[];
  };
  educationalContent: Array<{
    topic: string;
    content: string;
    learnMoreUrl?: string;
    relevance: 'high' | 'medium' | 'low';
  }>;
  improvementSuggestions: Array<{
    priority: number;
    action: string;
    description: string;
    expectedImpact: string;
    timelineMonths: number;
    difficulty: 'easy' | 'medium' | 'hard';
  }>;
}

export interface ConsentItem {
  consentId: string;
  serviceName: string;
  serviceDescription: string;
  category: 'fraud_detection' | 'loan_approval' | 'personalization' | 'marketing' | 'model_training';
  status: 'granted' | 'revoked' | 'expired';
  grantedAt?: string;
  revokedAt?: string;
  expiresAt?: string;
  dataUsageCount: number;
  lastUsed?: string;
}

export interface ConsentReceipt {
  receiptId: string;
  consentId: string;
  serviceName: string;
  action: 'granted' | 'revoked';
  timestamp: string;
  hash: string;
  blockchainStatus: 'pending' | 'verified' | 'failed';
  blockNumber?: number;
  transactionId?: string;
  cryptographicHash?: string;
  blockchainVerified?: boolean;
  previousHash?: string;
  signature?: string;
}

export interface UserFairnessReport {
  userId: string;
  overallFairness: {
    score: number;
    rating: 'excellent' | 'good' | 'fair' | 'needs_improvement';
  };
  comparisons: {
    gender: FairnessComparison;
    ageGroup: FairnessComparison;
    incomeLevel: FairnessComparison;
  };
  metrics: {
    approvalRateParity: number;
    demographicBias: number;
    consistencyScore: number;
    fairnessGrade: string;
  };
  recentAdjustments: Array<{
    date: string;
    adjustment: string;
    impact: string;
  }>;
}

export interface FairnessComparison {
  userGroup: string;
  groupApprovalRate: number;
  baselineRate: number;
  difference: number;
  fairnessStatus: 'fair' | 'monitoring' | 'investigating';
}

export interface AuditEntry {
  entryId: string;
  timestamp: string;
  action: string;
  category: 'decision' | 'consent' | 'access' | 'update';
  details: string;
  hash: string;
  previousHash: string;
  verified: boolean;
  metadata?: {
    modelVersion?: string;
    confidence?: number;
    userId?: string;
  };
}

// ============================================
// ADMIN DASHBOARD TYPES
// ============================================

export interface AdminOverviewResponse {
  systemHealth: {
    overall: number;
    components: {
      models: number;
      infrastructure: number;
      data: number;
      compliance: number;
    };
    trend: {
      direction: 'up' | 'down' | 'stable';
      change: number;
      period: string;
    };
  };
  metrics: {
    decisionsToday: number;
    decisionsYesterday: number;
    pendingApprovals: number;
    biasScore: number;
    driftScore: number;
    errorRate: number;
  };
  alerts: AdminAlert[];
  realtimeMetrics: {
    approvalRate: number;
    denialRate: number;
    manualReviewRate: number;
    avgLatencyMs: number;
    throughputPerMin: number;
  };
}

export interface AdminAlert {
  alertId: string;
  severity: 'critical' | 'warning' | 'info';
  type: 'model_drift' | 'fairness' | 'regulatory' | 'performance' | 'fairness_violation' | 'anomaly' | 'regulatory_update';
  title: string;
  message: string;
  timestamp: string;
  requiresAction: boolean;
  actionUrl?: string;
  detectedAt?: string;
  acknowledgedAt?: string;
  resolvedAt?: string;
  status?: 'active' | 'acknowledged' | 'resolved';
  metrics?: any;
  suggestedActions?: Array<{
    action: string;
    description: string;
    automated: boolean;
  }>;
}

export interface ModelHealth {
  modelId: string;
  name: string;
  version: string;
  status: 'healthy' | 'warning' | 'critical';
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    auc: number;
  };
  drift: {
    score: number;
    features: Array<{
      featureName: string;
      driftPercentage: number;
    }>;
    trend: 'increasing' | 'stable' | 'decreasing';
  };
  lastUpdated: string;
  lastRetrained: string;
  decisionsToday: number;
}

export interface FairnessMetrics {
  overallFairness: {
    score: number;
    grade: string;
    trend: 'improving' | 'stable' | 'degrading';
  };
  protectedGroups: Array<{
    attribute: string;
    groups: Array<{
      groupName: string;
      size: number;
      approvalRate: number;
      baselineDiff: number;
      status: 'fair' | 'monitoring' | 'violation';
      pValue: number;
      statisticalSignificance: boolean;
    }>;
  }>;
  dynamicOptimizer: {
    status: 'active' | 'paused' | 'disabled';
    mode: 'optimizing' | 'monitoring' | 'learning';
    episodes: number;
    lastAdjustment: {
      timestamp: string;
      action: string;
      impact: string;
      success: boolean;
    };
    recentActions: Array<{
      timestamp: string;
      action: string;
      result: string;
      biasReduction: number;
      accuracyImpact: number;
    }>;
  };
}

export interface ApprovalItem {
  approvalId: string;
  type: 'model_update' | 'policy_change' | 'fairness_adjustment' | 'rollback';
  priority: 'critical' | 'high' | 'medium' | 'low';
  submittedBy: {
    userId: string;
    name: string;
    team: string;
  };
  submittedAt: string;
  deadline?: string;
  details: {
    title: string;
    description: string;
    changes: any;
    impact: {
      users: number;
      accuracy?: number;
      fairness?: number;
      compliance?: string[];
    };
  };
  validations: {
    fairnessTest: 'passed' | 'failed' | 'pending';
    backtesting: 'passed' | 'failed' | 'pending';
    regulatoryReview: 'passed' | 'failed' | 'pending';
    securityScan: 'passed' | 'failed' | 'pending';
  };
}

export interface RegulatoryStatus {
  complianceScores: {
    gdpr: number;
    euAiAct: number;
    ccpa: number;
    rbi: number;
    iso23053: number;
    basel: number;
  };
  requirements: Array<{
    regulation: string;
    requirement: string;
    status: 'compliant' | 'in_progress' | 'non_compliant';
    deadline?: string;
    evidence?: string[];
    lastAudit?: string;
    nextAudit?: string;
  }>;
  regulatoryCompanion: {
    status: 'active' | 'inactive';
    lastUpdate: string;
    parsedRegulations: Array<{
      regulationId: string;
      name: string;
      parsedAt: string;
      machineReadable: {
        requirements: Array<{
          id: string;
          type: string;
          description: string;
          appliesTo: string[];
          implementation: any;
          deadline: string;
        }>;
      };
    }>;
  };
}

export interface LedgerBlock {
  blockNumber: number;
  hash: string;
  previousHash: string;
  timestamp: string;
  type: 'model_decision' | 'consent_change' | 'policy_update' | 'model_deployment';
  data: {
    decisionId?: string;
    modelVersion?: string;
    outcome?: string;
    confidence?: number;
    shapHash?: string;
    userId?: string;
    action?: string;
  };
  signature: string;
  verified: boolean;
}

export interface HumanLoopCase {
  caseId: string;
  type: 'loan_application' | 'fraud_alert' | 'dispute';
  aiConfidence: number;
  reasonForReview: 'low_confidence' | 'edge_case' | 'customer_dispute' | 'random_sample';
  priority: 'high' | 'medium' | 'low';
  assignedTo?: {
    userId: string;
    name: string;
  };
  status: 'pending' | 'in_review' | 'decided';
  submittedAt: string;
  deadline?: string;
  data: any;
  aiAnalysis: {
    recommendation: string;
    confidence: number;
    topFactors: Array<{
      factor: string;
      impact: number;
    }>;
    shapValues?: any;
  };
}

export interface ReviewerStats {
  reviewerId: string;
  name: string;
  casesReviewed: number;
  accuracy: number;
  avgProcessingTime: number;
  specialization?: string[];
}

