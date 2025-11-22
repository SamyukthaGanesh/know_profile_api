// Mock API Service Layer
import * as types from '../types/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Mock data generators
const mockUserDashboard: types.UserDashboardResponse = {
  user: {
    userId: 'USR-001',
    name: 'Sarah Johnson',
    lastLogin: new Date().toISOString(),
    profilePictureUrl: undefined,
  },
  trustScore: {
    overall: 85,
    components: {
      accuracy: 92,
      fairness: 94,
      transparency: 88,
      privacy: 90,
      explainability: 82,
      compliance: 95,
    },
    lastUpdated: new Date().toISOString(),
    trend: 'up',
  },
  activeLoanApplication: {
    applicationId: 'APP-2024-001',
    amount: 2500000,
    currency: 'INR',
    status: 'under_review',
    submittedAt: new Date(Date.now() - 86400000).toISOString(),
    confidence: 0.78,
    riskLevel: 'medium',
  },
  quickStats: {
    activeConsents: 12,
    fairnessRating: 94,
    pendingActions: 3,
  },
};

const mockExplanation: types.ExplanationResponse = {
  applicationId: 'APP-2024-001',
  decision: {
    outcome: 'under_review',
    confidence: 0.78,
    timestamp: new Date().toISOString(),
  },
  explanation: {
    summary: 'Your loan application is currently under review. The AI model has identified several key factors affecting your application.',
    literacyLevel: 'beginner',
    mainReasons: 'Your stable income and good job history are positive factors, but your credit score and debt-to-income ratio need improvement.',
    factors: [
      {
        featureName: 'credit_score',
        humanReadableName: 'Credit Score',
        description: 'Your monthly income of ₹85,000 is good and meets our requirements',
        value: 620,
        impact: 'negative',
        impactStrength: 'strong',
        impactPercentage: -25,
        shapValue: -0.25,
      },
      {
        featureName: 'income',
        humanReadableName: 'Stable Income',
        description: 'Your monthly income of ₹85,000 is good and meets our requirements',
        value: 85000,
        impact: 'positive',
        impactStrength: 'moderate',
        impactPercentage: 15,
        shapValue: 0.15,
      },
      {
        featureName: 'dti_ratio',
        humanReadableName: 'High Monthly Debts',
        description: "You're using 48% of your income for debts. We prefer under 40%.",
        value: 0.48,
        impact: 'negative',
        impactStrength: 'moderate',
        impactPercentage: -18,
        shapValue: -0.18,
      },
      {
        featureName: 'employment_years',
        humanReadableName: 'Good Job History',
        description: "You've been at your job for 3+ years. This shows stability!",
        value: 3.5,
        impact: 'positive',
        impactStrength: 'weak',
        impactPercentage: 8,
        shapValue: 0.08,
      },
    ],
    nextSteps: 'Consider improving your credit score by paying bills on time and reducing your monthly debt obligations.',
  },
  educationalContent: [
    {
      topic: 'Credit Score',
      content: 'A number that shows how well you\'ve paid back loans before. Higher is better!',
      relevance: 'high',
    },
    {
      topic: 'Debt-to-Income Ratio',
      content: 'How much of your paycheck goes to paying debts each month.',
      relevance: 'high',
    },
  ],
  improvementSuggestions: [
    {
      priority: 1,
      action: 'Improve Your Credit Score',
      description: 'Pay all bills on time for the next 3 months. This alone can boost your score by 50 points!',
      expectedImpact: 'High - Could increase approval chances by 45%',
      timelineMonths: 3,
      difficulty: 'easy',
    },
    {
      priority: 2,
      action: 'Reduce Your Monthly Debts',
      description: 'Try to pay off ₹2,00,000 of your current loans. This will improve your debt-to-income ratio.',
      expectedImpact: 'Medium - Would improve approval chances by 30%',
      timelineMonths: 4,
      difficulty: 'medium',
    },
    {
      priority: 3,
      action: 'Build an Emergency Fund',
      description: 'Save ₹2,50,000 (3 months of expenses). This shows financial stability to lenders.',
      expectedImpact: 'Low - Demonstrates financial responsibility',
      timelineMonths: 8,
      difficulty: 'hard',
    },
  ],
};

// API Service Functions
export const api = {
  // User Dashboard APIs
  async getUserDashboard(): Promise<types.UserDashboardResponse> {
    // In production, replace with actual API call
    return new Promise((resolve) => {
      setTimeout(() => resolve(mockUserDashboard), 500);
    });
  },

  async getExplanation(applicationId: string, literacyLevel: string = 'beginner'): Promise<types.ExplanationResponse> {
    return new Promise((resolve) => {
      setTimeout(() => resolve({...mockExplanation, explanation: {...mockExplanation.explanation, literacyLevel: literacyLevel as any}}), 500);
    });
  },

  async getConsentStatus(): Promise<{ consents: types.ConsentItem[]; statistics: any }> {
    const consents: types.ConsentItem[] = [
      {
        consentId: 'C-001',
        serviceName: 'Fraud Detection AI',
        serviceDescription: 'Analyze transactions for suspicious activity',
        category: 'fraud_detection',
        status: 'granted',
        grantedAt: new Date(Date.now() - 30 * 86400000).toISOString(),
        dataUsageCount: 245,
        lastUsed: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        consentId: 'C-002',
        serviceName: 'Loan Approval Model',
        serviceDescription: 'Use my data for credit decisions',
        category: 'loan_approval',
        status: 'granted',
        grantedAt: new Date(Date.now() - 60 * 86400000).toISOString(),
        dataUsageCount: 12,
        lastUsed: new Date(Date.now() - 86400000).toISOString(),
      },
      {
        consentId: 'C-003',
        serviceName: 'Personalization Engine',
        serviceDescription: 'Customize offers based on my preferences',
        category: 'personalization',
        status: 'granted',
        grantedAt: new Date(Date.now() - 45 * 86400000).toISOString(),
        dataUsageCount: 89,
        lastUsed: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        consentId: 'C-004',
        serviceName: 'Marketing Analytics',
        serviceDescription: 'Include my anonymized data in marketing analysis',
        category: 'marketing',
        status: 'revoked',
        grantedAt: new Date(Date.now() - 90 * 86400000).toISOString(),
        revokedAt: new Date(Date.now() - 15 * 86400000).toISOString(),
        dataUsageCount: 34,
      },
      {
        consentId: 'C-005',
        serviceName: 'Model Training',
        serviceDescription: 'Use my data to improve AI models',
        category: 'model_training',
        status: 'revoked',
        dataUsageCount: 0,
      },
    ];

    return new Promise((resolve) => {
      setTimeout(() => resolve({
        consents,
        statistics: {
          totalActive: consents.filter(c => c.status === 'granted').length,
          totalRevoked: consents.filter(c => c.status === 'revoked').length,
          totalDataUsage: consents.reduce((sum, c) => sum + c.dataUsageCount, 0),
        },
      }), 500);
    });
  },

  async updateConsent(consentId: string, action: 'grant' | 'revoke'): Promise<{ receipt: types.ConsentReceipt }> {
    const receipt: types.ConsentReceipt = {
      receiptId: `CR-${Date.now()}`,
      consentId,
      serviceName: 'Service',
      action: action === 'grant' ? 'granted' : 'revoked',
      timestamp: new Date().toISOString(),
      hash: `0x${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`,
      blockchainStatus: 'verified',
      blockNumber: Math.floor(Math.random() * 100000),
      cryptographicHash: `0x${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`,
      blockchainVerified: true,
      previousHash: `0x${Math.random().toString(36).substring(2, 15)}`,
      signature: `0x${Math.random().toString(36).substring(2, 15)}`,
    };

    return new Promise((resolve) => {
      setTimeout(() => resolve({ receipt }), 500);
    });
  },

  async getFairnessReport(): Promise<types.UserFairnessReport> {
    return new Promise((resolve) => {
      setTimeout(() => resolve({
        userId: 'USR-001',
        overallFairness: {
          score: 96,
          rating: 'excellent',
        },
        comparisons: {
          gender: {
            userGroup: 'Female',
            groupApprovalRate: 72.3,
            baselineRate: 72.1,
            difference: 0.2,
            fairnessStatus: 'fair',
          },
          ageGroup: {
            userGroup: '26-40',
            groupApprovalRate: 73.8,
            baselineRate: 72.1,
            difference: 1.7,
            fairnessStatus: 'fair',
          },
          incomeLevel: {
            userGroup: 'Middle (50k-100k)',
            groupApprovalRate: 71.5,
            baselineRate: 72.1,
            difference: -0.6,
            fairnessStatus: 'fair',
          },
        },
        metrics: {
          approvalRateParity: 94,
          demographicBias: 0.02,
          consistencyScore: 98,
          fairnessGrade: 'A+',
        },
        recentAdjustments: [
          {
            date: new Date(Date.now() - 7200000).toISOString(),
            adjustment: 'Reduced age bias for 18-25 group',
            impact: 'Bias reduced from 0.04 to 0.02',
          },
        ],
      }), 500);
    });
  },

  async getAuditTrail(page: number = 1): Promise<{ entries: types.AuditEntry[]; pagination: any }> {
    const entries: types.AuditEntry[] = [
      {
        entryId: 'AE-001',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        action: 'Loan Application Submitted',
        category: 'decision',
        details: 'Amount: ₹25,00,000',
        hash: '0x7f3a9b2c4e1d5f6a',
        previousHash: '0x6e2a8b1c3d4e5f6g',
        verified: true,
        metadata: {
          modelVersion: 'v2.1',
          confidence: 0.78,
        },
      },
      {
        entryId: 'AE-002',
        timestamp: new Date(Date.now() - 295000).toISOString(),
        action: 'Consent Verified',
        category: 'consent',
        details: 'All required consents active',
        hash: '0x8g4b0c3d5e2f6g7h',
        previousHash: '0x7f3a9b2c4e1d5f6a',
        verified: true,
      },
      {
        entryId: 'AE-003',
        timestamp: new Date(Date.now() - 290000).toISOString(),
        action: 'AI Model Executed',
        category: 'decision',
        details: 'Model v2.1, Confidence: 78%',
        hash: '0x9h5c1d4e6f3g7h8i',
        previousHash: '0x8g4b0c3d5e2f6g7h',
        verified: true,
        metadata: {
          modelVersion: 'v2.1',
          confidence: 0.78,
        },
      },
    ];

    return new Promise((resolve) => {
      setTimeout(() => resolve({
        entries,
        pagination: {
          page,
          pageSize: 10,
          total: 50,
        },
      }), 500);
    });
  },

  // Admin Dashboard APIs
  async getAdminOverview(): Promise<types.AdminOverviewResponse> {
    return new Promise((resolve) => {
      setTimeout(() => resolve({
        systemHealth: {
          overall: 98.5,
          components: {
            models: 95,
            infrastructure: 99,
            data: 97,
            compliance: 100,
          },
          trend: {
            direction: 'up',
            change: 2.1,
            period: 'week',
          },
        },
        metrics: {
          decisionsToday: 12847,
          decisionsYesterday: 12324,
          pendingApprovals: 3,
          biasScore: 0.018,
          driftScore: 5.2,
          errorRate: 0.025,
        },
        alerts: [
          {
            alertId: 'ALT-001',
            severity: 'critical',
            type: 'model_drift',
            title: 'Model Drift Detected',
            message: 'Loan approval model v2.1 showing 5.2% drift in predictions. Manual review recommended.',
            timestamp: new Date(Date.now() - 120000).toISOString(),
            requiresAction: true,
          },
          {
            alertId: 'ALT-002',
            severity: 'warning',
            type: 'fairness',
            title: 'Fairness Threshold Alert',
            message: 'Age group 18-25 approval rate 8% below baseline. Auto-adjustment pending approval.',
            timestamp: new Date(Date.now() - 900000).toISOString(),
            requiresAction: true,
          },
          {
            alertId: 'ALT-003',
            severity: 'info',
            type: 'regulatory',
            title: 'New Regulation: EU AI Act Update',
            message: 'Article 15 requires additional transparency measures by Jan 2025.',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            requiresAction: true,
          },
        ],
        realtimeMetrics: {
          approvalRate: 73,
          denialRate: 22,
          manualReviewRate: 5,
          avgLatencyMs: 45,
          throughputPerMin: 15600,
        },
      }), 500);
    });
  },

  async getModelHealth(): Promise<{ models: types.ModelHealth[]; performanceMetrics: any }> {
    const models: types.ModelHealth[] = [
      {
        modelId: 'M-001',
        name: 'Loan Approval',
        version: 'v2.1',
        status: 'warning',
        metrics: {
          accuracy: 87.3,
          precision: 85.2,
          recall: 89.1,
          f1Score: 87.1,
          auc: 0.893,
        },
        drift: {
          score: 5.2,
          features: [
            { featureName: 'credit_score', driftPercentage: 3.2 },
            { featureName: 'income', driftPercentage: 1.8 },
          ],
          trend: 'increasing',
        },
        lastUpdated: new Date(Date.now() - 7200000).toISOString(),
        lastRetrained: new Date(Date.now() - 604800000).toISOString(),
        decisionsToday: 8542,
      },
      {
        modelId: 'M-002',
        name: 'Fraud Detection',
        version: 'v3.4',
        status: 'healthy',
        metrics: {
          accuracy: 94.2,
          precision: 93.8,
          recall: 94.5,
          f1Score: 94.1,
          auc: 0.965,
        },
        drift: {
          score: 0.8,
          features: [],
          trend: 'stable',
        },
        lastUpdated: new Date(Date.now() - 3600000).toISOString(),
        lastRetrained: new Date(Date.now() - 1209600000).toISOString(),
        decisionsToday: 3245,
      },
    ];

    return new Promise((resolve) => {
      setTimeout(() => resolve({
        models,
        performanceMetrics: {
          latency: { p50: 35, p95: 65, p99: 95 },
          throughput: { current: 15600, capacity: 20000, utilizationPercent: 78 },
          errorRate: { rate: 0.025, threshold: 0.1, errors24h: 32 },
          cacheHitRate: 92,
        },
      }), 500);
    });
  },

  async getFairnessMetrics(): Promise<types.FairnessMetrics> {
    return new Promise((resolve) => {
      setTimeout(() => resolve({
        overallFairness: {
          score: 94,
          grade: 'A+',
          trend: 'improving',
        },
        protectedGroups: [
          {
            attribute: 'gender',
            groups: [
              {
                groupName: 'Female',
                size: 12450,
                approvalRate: 72.3,
                baselineDiff: 0.2,
                status: 'fair',
                pValue: 0.15,
                statisticalSignificance: false,
              },
              {
                groupName: 'Male',
                size: 13200,
                approvalRate: 72.1,
                baselineDiff: 0,
                status: 'fair',
                pValue: 1.0,
                statisticalSignificance: false,
              },
            ],
          },
          {
            attribute: 'age',
            groups: [
              {
                groupName: '18-25',
                size: 4200,
                approvalRate: 64.5,
                baselineDiff: -7.6,
                status: 'monitoring',
                pValue: 0.03,
                statisticalSignificance: true,
              },
              {
                groupName: '26-40',
                size: 15800,
                approvalRate: 73.8,
                baselineDiff: 1.7,
                status: 'fair',
                pValue: 0.12,
                statisticalSignificance: false,
              },
            ],
          },
        ],
        dynamicOptimizer: {
          status: 'active',
          mode: 'optimizing',
          episodes: 1247,
          lastAdjustment: {
            timestamp: new Date(Date.now() - 720000).toISOString(),
            action: 'Reduced age bias for 18-25 group',
            impact: 'Bias reduced from 0.04 to 0.02',
            success: true,
          },
          recentActions: [
            {
              timestamp: new Date(Date.now() - 720000).toISOString(),
              action: 'Threshold adjustment for age 18-25',
              result: 'SUCCESS',
              biasReduction: 2.1,
              accuracyImpact: 0,
            },
            {
              timestamp: new Date(Date.now() - 1440000).toISOString(),
              action: 'Gender reweighting',
              result: 'ROLLBACK',
              biasReduction: 0,
              accuracyImpact: -3,
            },
          ],
        },
      }), 500);
    });
  },

  async getApprovalsQueue(): Promise<{ pendingApprovals: types.ApprovalItem[]; statistics: any }> {
    const pendingApprovals: types.ApprovalItem[] = [
      {
        approvalId: 'APV-001',
        type: 'model_update',
        priority: 'high',
        submittedBy: {
          userId: 'U-ML-001',
          name: 'ML Team',
          team: 'Data Science',
        },
        submittedAt: new Date(Date.now() - 3600000).toISOString(),
        details: {
          title: 'Model Update: Loan Approval v2.2',
          description: 'Updated feature weights, improved accuracy by 3.2%',
          changes: { features_updated: 12, accuracy_improvement: 3.2 },
          impact: {
            users: 25000,
            accuracy: 3.2,
            fairness: 0,
          },
        },
        validations: {
          fairnessTest: 'passed',
          backtesting: 'passed',
          regulatoryReview: 'pending',
          securityScan: 'passed',
        },
      },
      {
        approvalId: 'APV-002',
        type: 'policy_change',
        priority: 'medium',
        submittedBy: {
          userId: 'U-LEG-001',
          name: 'Legal Team',
          team: 'Legal',
        },
        submittedAt: new Date(Date.now() - 7200000).toISOString(),
        details: {
          title: 'Policy Change: GDPR Consent Update',
          description: 'Updated consent collection for AI model training',
          changes: {},
          impact: {
            users: 45000,
          },
        },
        validations: {
          fairnessTest: 'passed',
          backtesting: 'passed',
          regulatoryReview: 'passed',
          securityScan: 'passed',
        },
      },
      {
        approvalId: 'APV-003',
        type: 'fairness_adjustment',
        priority: 'high',
        submittedBy: {
          userId: 'SYSTEM',
          name: 'AI System (Auto)',
          team: 'Automated',
        },
        submittedAt: new Date(Date.now() - 1800000).toISOString(),
        details: {
          title: 'Fairness Adjustment: Age Group 18-25',
          description: 'Reduce bias by adjusting decision threshold',
          changes: { threshold_adjustment: 0.05 },
          impact: {
            users: 12000,
            fairness: 8,
          },
        },
        validations: {
          fairnessTest: 'passed',
          backtesting: 'passed',
          regulatoryReview: 'passed',
          securityScan: 'passed',
        },
      },
    ];

    return new Promise((resolve) => {
      setTimeout(() => resolve({
        pendingApprovals,
        statistics: {
          total: 3,
          critical: 0,
          avgProcessingTime: 4.5,
          overdueCount: 0,
        },
      }), 500);
    });
  },

  async getRegulatoryStatus(): Promise<types.RegulatoryStatus> {
    return new Promise((resolve) => {
      setTimeout(() => resolve({
        complianceScores: {
          gdpr: 100,
          euAiAct: 98,
          ccpa: 100,
          rbi: 100,
          iso23053: 85,
          basel: 95,
        },
        requirements: [
          {
            regulation: 'EU AI Act',
            requirement: 'High-risk system documentation',
            status: 'compliant',
            lastAudit: new Date(Date.now() - 2592000000).toISOString(),
            nextAudit: new Date(Date.now() + 2592000000).toISOString(),
          },
          {
            regulation: 'EU AI Act',
            requirement: 'Transparency obligations',
            status: 'in_progress',
            deadline: '2025-01-15',
          },
        ],
        regulatoryCompanion: {
          status: 'active',
          lastUpdate: new Date(Date.now() - 7200000).toISOString(),
          parsedRegulations: [
            {
              regulationId: 'EU-AI-ACT-ART15',
              name: 'EU AI Act Article 15',
              parsedAt: new Date(Date.now() - 7200000).toISOString(),
              machineReadable: {
                requirements: [
                  {
                    id: 'REQ-001',
                    type: 'transparency_notice',
                    description: 'Users must be informed when interacting with AI systems',
                    appliesTo: ['high_risk_systems', 'credit_scoring'],
                    implementation: {
                      user_notification: true,
                      ai_disclosure: 'mandatory',
                      explanation_level: 'detailed',
                    },
                    deadline: '2025-01-15',
                  },
                ],
              },
            },
          ],
        },
      }), 500);
    });
  },

  async getLedgerBlocks(page: number = 1): Promise<{ blocks: types.LedgerBlock[]; integrity: any }> {
    const blocks: types.LedgerBlock[] = [
      {
        blockNumber: 48291,
        hash: '0x8f4b0c3d5e2f6g7h8i9j0k1l2m3n4o5p',
        previousHash: '0x6e2a8b1c3d4e5f6g7h8i9j0k',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        type: 'model_decision',
        data: {
          decisionId: 'DEC-2024-112233',
          modelVersion: 'loan_approval_v2.1',
          outcome: 'APPROVED',
          confidence: 0.893,
          shapHash: '0x7f3a9b2c4e1d5f6a8b9c0d1e',
        },
        signature: '0xSIGN123',
        verified: true,
      },
      {
        blockNumber: 48290,
        hash: '0x6e2a8b1c3d4e5f6g7h8i9j0k1l2m3n4o',
        previousHash: '0x5d1a7b0c2d3e4f5g6h7i8j9k',
        timestamp: new Date(Date.now() - 310000).toISOString(),
        type: 'consent_change',
        data: {
          userId: 'USR-445566',
          action: 'CONSENT_GRANTED',
        },
        signature: '0xSIGN122',
        verified: true,
      },
    ];

    return new Promise((resolve) => {
      setTimeout(() => resolve({
        blocks,
        integrity: {
          chainValid: true,
          lastVerification: new Date().toISOString(),
          totalBlocks: 48291,
          corruptedBlocks: 0,
        },
      }), 500);
    });
  },

  async getHumanLoopCases(): Promise<{ cases: types.HumanLoopCase[]; reviewerStats: types.ReviewerStats[] }> {
    const cases: types.HumanLoopCase[] = [
      {
        caseId: 'LN-2024-8821',
        type: 'loan_application',
        aiConfidence: 52,
        reasonForReview: 'low_confidence',
        priority: 'high',
        assignedTo: {
          userId: 'REV-001',
          name: 'John D.',
        },
        status: 'in_review',
        submittedAt: new Date(Date.now() - 3600000).toISOString(),
        data: {},
        aiAnalysis: {
          recommendation: 'APPROVE_WITH_CONDITIONS',
          confidence: 0.52,
          topFactors: [
            { factor: 'Credit Score', impact: -0.15 },
            { factor: 'Income Variance', impact: -0.22 },
          ],
        },
      },
      {
        caseId: 'FR-2024-3342',
        type: 'fraud_alert',
        aiConfidence: 78,
        reasonForReview: 'customer_dispute',
        priority: 'medium',
        assignedTo: {
          userId: 'REV-002',
          name: 'Sarah M.',
        },
        status: 'in_review',
        submittedAt: new Date(Date.now() - 7200000).toISOString(),
        data: {},
        aiAnalysis: {
          recommendation: 'FLAG_FOR_REVIEW',
          confidence: 0.78,
          topFactors: [
            { factor: 'Transaction Pattern', impact: 0.35 },
          ],
        },
      },
    ];

    const reviewerStats: types.ReviewerStats[] = [
      {
        reviewerId: 'REV-001',
        name: 'John D.',
        casesReviewed: 142,
        accuracy: 94,
        avgProcessingTime: 1.2,
      },
      {
        reviewerId: 'REV-002',
        name: 'Sarah M.',
        casesReviewed: 128,
        accuracy: 92,
        avgProcessingTime: 1.5,
      },
      {
        reviewerId: 'REV-003',
        name: 'Mike R.',
        casesReviewed: 156,
        accuracy: 96,
        avgProcessingTime: 1.0,
      },
    ];

    return new Promise((resolve) => {
      setTimeout(() => resolve({ cases, reviewerStats }), 500);
    });
  },
};

