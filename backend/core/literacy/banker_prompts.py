"""
Banker-Specific Prompt Templates
Generates prompts for explaining AI decisions to banking professionals.
Covers both technical analysts and business executives.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from .prompt_generator import (
    PromptGenerator,
    ExplanationContext,
    AudienceType,
    ExplanationType
)

logger = logging.getLogger(__name__)


@dataclass
class BankerProfile:
    """Profile information about the banker/analyst"""
    banker_id: str
    role: str  # 'technical_analyst', 'business_executive', 'compliance_officer', 'risk_manager'
    department: str  # 'credit', 'fraud', 'risk', 'compliance', 'operations'
    expertise_level: str  # 'junior', 'senior', 'executive'
    
    # Context
    portfolio_size: Optional[int] = None
    responsibility_area: Optional[str] = None
    regulatory_focus: Optional[List[str]] = None


class BankerPromptGenerator:
    """
    Generates professional prompts for banking staff.
    Tailored for technical analysis, business strategy, and compliance.
    """
    
    def __init__(self):
        """Initialize banker prompt generator"""
        self.base_generator = PromptGenerator()
        self.templates = self._load_banker_templates()
    
    def generate_technical_analysis_prompt(
        self,
        context: ExplanationContext,
        banker_profile: BankerProfile,
        include_model_details: bool = True,
        include_risk_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Generate prompt for technical analysis of a decision.
        
        Args:
            context: Explanation context with decision details
            banker_profile: Banker profile information
            include_model_details: Whether to include model architecture details
            include_risk_metrics: Whether to include risk analysis
            
        Returns:
            Complete prompt dictionary
        """
        context.audience = AudienceType.BANKER_TECHNICAL
        
        # Generate base prompt
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.DECISION,
            include_examples=False  # Bankers don't need examples
        )
        
        # Add technical sections
        base_prompt['technical_context'] = self._build_technical_context(
            banker_profile, context
        )
        
        if include_model_details:
            base_prompt['model_details'] = self._create_model_details_section(context)
        
        if include_risk_metrics:
            base_prompt['risk_analysis'] = self._create_risk_analysis_section(context)
        
        # Add performance metrics
        base_prompt['performance_metrics'] = self._create_performance_section(context)
        
        # Add statistical significance
        base_prompt['statistical_analysis'] = self._create_statistical_section(context)
        
        base_prompt['formatted_prompt'] = self._format_technical_prompt(
            base_prompt, context
        )
        
        return base_prompt
    
    def generate_business_analysis_prompt(
        self,
        context: ExplanationContext,
        banker_profile: BankerProfile,
        include_roi_analysis: bool = True,
        include_strategic_implications: bool = True
    ) -> Dict[str, Any]:
        """
        Generate prompt for business analysis.
        
        Args:
            context: Explanation context
            banker_profile: Banker profile
            include_roi_analysis: Whether to include ROI analysis
            include_strategic_implications: Whether to include strategic impact
            
        Returns:
            Prompt dictionary for business analysis
        """
        context.audience = AudienceType.BANKER_BUSINESS
        
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.DECISION,
            include_examples=False
        )
        
        # Add business sections
        base_prompt['business_context'] = self._build_business_context(
            banker_profile, context
        )
        
        if include_roi_analysis:
            base_prompt['roi_analysis'] = self._create_roi_section(context)
        
        if include_strategic_implications:
            base_prompt['strategic_impact'] = self._create_strategic_section(context)
        
        # Add KPI mapping
        base_prompt['kpi_mapping'] = self._create_kpi_mapping(context)
        
        # Add competitive analysis
        base_prompt['competitive_context'] = self._create_competitive_section(context)
        
        base_prompt['formatted_prompt'] = self._format_business_prompt(
            base_prompt, context
        )
        
        return base_prompt
    
    def generate_compliance_report_prompt(
        self,
        context: ExplanationContext,
        banker_profile: BankerProfile,
        regulatory_framework: str = 'Basel III',
        audit_requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate prompt for compliance and regulatory reporting.
        
        Args:
            context: Explanation context
            banker_profile: Banker profile
            regulatory_framework: Applicable regulatory framework
            audit_requirements: Specific audit requirements
            
        Returns:
            Prompt dictionary for compliance reporting
        """
        context.audience = AudienceType.REGULATOR
        
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.COMPLIANCE,
            include_examples=False
        )
        
        # Add compliance sections
        base_prompt['compliance_context'] = {
            'regulatory_framework': regulatory_framework,
            'audit_requirements': audit_requirements or [],
            'documentation_level': 'comprehensive'
        }
        
        base_prompt['governance_section'] = self._create_governance_section(context)
        base_prompt['audit_trail'] = self._create_audit_trail_section(context)
        base_prompt['fairness_documentation'] = self._create_fairness_docs_section(context)
        
        base_prompt['formatted_prompt'] = self._format_compliance_prompt(
            base_prompt, context, regulatory_framework
        )
        
        return base_prompt
    
    def generate_risk_assessment_prompt(
        self,
        context: ExplanationContext,
        banker_profile: BankerProfile,
        risk_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate prompt for risk assessment analysis.
        
        Args:
            context: Explanation context
            banker_profile: Banker profile
            risk_categories: Specific risk categories to analyze
            
        Returns:
            Prompt dictionary for risk assessment
        """
        context.audience = AudienceType.BANKER_TECHNICAL
        
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.DECISION,
            include_examples=False
        )
        
        # Add risk sections
        base_prompt['risk_context'] = {
            'categories': risk_categories or ['credit', 'operational', 'reputational', 'model'],
            'assessment_framework': 'quantitative and qualitative'
        }
        
        base_prompt['risk_quantification'] = self._create_risk_quantification_section(context)
        base_prompt['mitigation_strategies'] = self._create_mitigation_section(context)
        base_prompt['monitoring_plan'] = self._create_monitoring_section(context)
        
        base_prompt['formatted_prompt'] = self._format_risk_prompt(
            base_prompt, context
        )
        
        return base_prompt
    
    def generate_portfolio_analysis_prompt(
        self,
        decisions: List[ExplanationContext],
        banker_profile: BankerProfile,
        analysis_type: str = 'aggregate'
    ) -> Dict[str, Any]:
        """
        Generate prompt for portfolio-level analysis.
        
        Args:
            decisions: List of decision contexts
            banker_profile: Banker profile
            analysis_type: Type of analysis ('aggregate', 'segmented', 'trend')
            
        Returns:
            Prompt dictionary for portfolio analysis
        """
        # Use first decision as template
        context = decisions[0] if decisions else None
        
        if not context:
            raise ValueError("At least one decision context required")
        
        context.audience = AudienceType.BANKER_BUSINESS
        
        prompt = {
            'system': self._get_portfolio_system_prompt(),
            'portfolio_context': {
                'total_decisions': len(decisions),
                'analysis_type': analysis_type,
                'time_period': 'specified by data',
                'portfolio_size': banker_profile.portfolio_size
            },
            'analysis_requirements': self._create_portfolio_requirements(analysis_type),
            'aggregation_methods': self._create_aggregation_section(),
            'trend_analysis': self._create_trend_section(),
            'formatted_prompt': ''
        }
        
        prompt['formatted_prompt'] = self._format_portfolio_prompt(prompt, decisions)
        
        return prompt
    
    def _build_technical_context(
        self,
        banker_profile: BankerProfile,
        context: ExplanationContext
    ) -> Dict[str, Any]:
        """Build technical context for bankers"""
        return {
            'role': banker_profile.role,
            'department': banker_profile.department,
            'expertise': banker_profile.expertise_level,
            'model_name': context.model_name,
            'model_version': context.model_version,
            'technical_depth': 'comprehensive'
        }
    
    def _build_business_context(
        self,
        banker_profile: BankerProfile,
        context: ExplanationContext
    ) -> Dict[str, Any]:
        """Build business context for executives"""
        return {
            'role': banker_profile.role,
            'department': banker_profile.department,
            'responsibility_area': banker_profile.responsibility_area,
            'strategic_focus': 'business_impact',
            'metrics_focus': 'kpis_and_roi'
        }
    
    def _create_model_details_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create model details section"""
        return {
            'instruction': 'Provide comprehensive model details',
            'include': [
                'Model architecture and algorithm',
                'Training data characteristics',
                'Feature engineering approach',
                'Hyperparameters and configuration',
                'Model performance metrics',
                'Validation methodology'
            ],
            'format': 'technical_specification'
        }
    
    def _create_risk_analysis_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create risk analysis section"""
        return {
            'instruction': 'Analyze risks associated with this decision',
            'risk_types': [
                'Credit risk',
                'Operational risk',
                'Model risk',
                'Reputational risk'
            ],
            'include_quantification': True,
            'include_mitigation': True
        }
    
    def _create_performance_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create performance metrics section"""
        return {
            'instruction': 'Provide model performance metrics',
            'metrics': [
                'Accuracy, Precision, Recall, F1-Score',
                'AUC-ROC, AUC-PR',
                'Confusion matrix',
                'Calibration metrics',
                'Performance by segment'
            ],
            'benchmark_comparison': True
        }
    
    def _create_statistical_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create statistical analysis section"""
        return {
            'instruction': 'Provide statistical analysis',
            'include': [
                'Confidence intervals',
                'Statistical significance tests',
                'Effect sizes',
                'Feature correlation analysis',
                'Uncertainty quantification'
            ]
        }
    
    def _create_roi_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create ROI analysis section"""
        return {
            'instruction': 'Analyze return on investment',
            'components': [
                'Cost savings from automation',
                'Revenue impact',
                'Efficiency gains',
                'Risk reduction value',
                'Customer satisfaction impact'
            ],
            'time_horizon': ['immediate', 'annual', '3-year'],
            'include_assumptions': True
        }
    
    def _create_strategic_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create strategic implications section"""
        return {
            'instruction': 'Analyze strategic implications',
            'areas': [
                'Competitive advantage',
                'Market positioning',
                'Customer experience impact',
                'Operational efficiency',
                'Scalability considerations',
                'Future opportunities'
            ]
        }
    
    def _create_kpi_mapping(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create KPI mapping section"""
        return {
            'instruction': 'Map technical metrics to business KPIs',
            'mappings': {
                'Accuracy': 'Decision quality rate',
                'Recall': 'Customer capture rate',
                'Precision': 'False positive cost',
                'Processing time': 'Operational efficiency',
                'Fairness metrics': 'Regulatory compliance score'
            }
        }
    
    def _create_competitive_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create competitive analysis section"""
        return {
            'instruction': 'Analyze competitive position',
            'include': [
                'Industry benchmarks',
                'Best practices comparison',
                'Innovation assessment',
                'Market differentiation'
            ]
        }
    
    def _create_governance_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create governance section"""
        return {
            'instruction': 'Document governance framework',
            'include': [
                'Model approval process',
                'Oversight committee',
                'Review frequency',
                'Change management',
                'Escalation procedures',
                'Accountability structure'
            ]
        }
    
    def _create_audit_trail_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create audit trail section"""
        return {
            'instruction': 'Provide complete audit trail',
            'include': [
                'Decision timestamp and ID',
                'Model version used',
                'Input data snapshot',
                'Feature values and transformations',
                'Model output and confidence',
                'Override history (if any)',
                'Reviewer actions'
            ],
            'format': 'chronological_log'
        }
    
    def _create_fairness_docs_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create fairness documentation section"""
        return {
            'instruction': 'Document fairness testing and monitoring',
            'include': [
                'Fairness metrics used',
                'Protected attributes considered',
                'Testing methodology',
                'Results and interpretation',
                'Mitigation actions taken',
                'Ongoing monitoring plan'
            ]
        }
    
    def _create_risk_quantification_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create risk quantification section"""
        return {
            'instruction': 'Quantify risks in financial terms',
            'metrics': [
                'Expected loss',
                'Value at Risk (VaR)',
                'Probability of default',
                'Loss given default',
                'Exposure at default'
            ],
            'scenarios': ['base', 'stressed', 'adverse']
        }
    
    def _create_mitigation_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create risk mitigation section"""
        return {
            'instruction': 'Recommend risk mitigation strategies',
            'categories': [
                'Model improvements',
                'Process enhancements',
                'Control mechanisms',
                'Monitoring systems',
                'Contingency plans'
            ]
        }
    
    def _create_monitoring_section(self, context: ExplanationContext) -> Dict[str, Any]:
        """Create monitoring plan section"""
        return {
            'instruction': 'Define ongoing monitoring plan',
            'include': [
                'Key metrics to track',
                'Monitoring frequency',
                'Alert thresholds',
                'Reporting cadence',
                'Review triggers',
                'Escalation criteria'
            ]
        }
    
    def _create_portfolio_requirements(self, analysis_type: str) -> Dict[str, Any]:
        """Create portfolio analysis requirements"""
        requirements = {
            'aggregate': {
                'metrics': ['Total volume', 'Approval rates', 'Average confidence', 'Distribution analysis'],
                'segments': ['By product', 'By region', 'By customer segment', 'By decision outcome']
            },
            'segmented': {
                'metrics': ['Segment-specific approval rates', 'Performance by segment', 'Risk profiles'],
                'comparisons': ['Cross-segment', 'Time-based', 'Benchmark']
            },
            'trend': {
                'metrics': ['Time series analysis', 'Seasonality', 'Drift detection', 'Growth rates'],
                'forecasting': True
            }
        }
        
        return requirements.get(analysis_type, requirements['aggregate'])
    
    def _create_aggregation_section(self) -> Dict[str, Any]:
        """Create aggregation methods section"""
        return {
            'instruction': 'Aggregate individual decisions into portfolio insights',
            'methods': [
                'Weighted averages',
                'Percentile analysis',
                'Cohort analysis',
                'Segment stratification'
            ]
        }
    
    def _create_trend_section(self) -> Dict[str, Any]:
        """Create trend analysis section"""
        return {
            'instruction': 'Identify and analyze trends',
            'include': [
                'Time series patterns',
                'Seasonality detection',
                'Anomaly identification',
                'Predictive indicators'
            ]
        }
    
    def _get_portfolio_system_prompt(self) -> str:
        """Get system prompt for portfolio analysis"""
        return """You are a senior banking analyst providing portfolio-level analysis 
        of AI-driven decisions. Focus on aggregate patterns, trends, and strategic insights. 
        Provide actionable recommendations for portfolio optimization and risk management."""
    
    def _format_technical_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext
    ) -> str:
        """Format technical analysis prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

TECHNICAL CONTEXT:
{self._format_dict(prompt_dict['technical_context'])}

DECISION DETAILS:
Decision ID: {context.decision_id}
Type: {context.decision_type}
Outcome: {context.outcome}
Confidence: {context.confidence:.3f}
Model: {context.model_name} v{context.model_version}

TOP CONTRIBUTING FACTORS (SHAP Values):
{chr(10).join(f'{i+1}. {f["name"]}: SHAP={f.get("shap_value", "N/A"):.4f}, Value={f.get("value", "N/A")}' 
              for i, f in enumerate(context.top_factors[:10]))}

MODEL DETAILS REQUIRED:
{self._format_dict(prompt_dict['model_details'])}

PERFORMANCE METRICS REQUIRED:
{self._format_dict(prompt_dict['performance_metrics'])}

RISK ANALYSIS REQUIRED:
{self._format_dict(prompt_dict['risk_analysis'])}

STATISTICAL ANALYSIS REQUIRED:
{self._format_dict(prompt_dict['statistical_analysis'])}

OUTPUT FORMAT:
Provide comprehensive technical report with:
1. Executive Summary (technical)
2. Model Architecture & Performance
3. Decision Analysis with SHAP interpretation
4. Risk Assessment
5. Statistical Validation
6. Recommendations for model improvement

Include all relevant metrics, confidence intervals, and statistical tests.
"""
        
        return formatted.strip()
    
    def _format_business_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext
    ) -> str:
        """Format business analysis prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

BUSINESS CONTEXT:
{self._format_dict(prompt_dict['business_context'])}

DECISION SUMMARY:
Type: {context.decision_type}
Outcome: {context.outcome}
Confidence: {context.confidence:.1%}

KEY FACTORS:
{chr(10).join(f'- {f["name"]}' for f in context.top_factors[:5])}

ROI ANALYSIS REQUIRED:
{self._format_dict(prompt_dict['roi_analysis'])}

STRATEGIC IMPACT REQUIRED:
{self._format_dict(prompt_dict['strategic_impact'])}

KPI MAPPING:
{self._format_dict(prompt_dict['kpi_mapping'])}

COMPETITIVE CONTEXT:
{self._format_dict(prompt_dict['competitive_context'])}

OUTPUT FORMAT:
Provide executive business report with:
1. Executive Summary (business impact)
2. ROI Analysis with financial projections
3. Strategic Implications
4. KPI Impact Assessment
5. Competitive Positioning
6. Recommendations for business optimization

Focus on business value, not technical details. Use business language and metrics.
"""
        
        return formatted.strip()
    
    def _format_compliance_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext,
        regulatory_framework: str
    ) -> str:
        """Format compliance reporting prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

COMPLIANCE CONTEXT:
Regulatory Framework: {regulatory_framework}
Audit Requirements: {', '.join(prompt_dict['compliance_context']['audit_requirements'])}

DECISION DETAILS:
Decision ID: {context.decision_id}
Type: {context.decision_type}
Outcome: {context.outcome}
Timestamp: [To be filled]

GOVERNANCE DOCUMENTATION REQUIRED:
{self._format_dict(prompt_dict['governance_section'])}

AUDIT TRAIL REQUIRED:
{self._format_dict(prompt_dict['audit_trail'])}

FAIRNESS DOCUMENTATION REQUIRED:
{self._format_dict(prompt_dict['fairness_documentation'])}

OUTPUT FORMAT:
Provide compliance report suitable for regulatory submission:
1. Executive Summary (compliance status)
2. Governance Framework Documentation
3. Complete Audit Trail
4. Fairness Testing Results
5. Risk Assessment and Mitigation
6. Ongoing Monitoring Plan
7. Attestations and Certifications

Ensure all documentation meets {regulatory_framework} requirements.
Format for regulatory review and audit purposes.
"""
        
        return formatted.strip()
    
    def _format_risk_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext
    ) -> str:
        """Format risk assessment prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

RISK ASSESSMENT CONTEXT:
Decision: {context.decision_type}
Outcome: {context.outcome}
Risk Categories: {', '.join(prompt_dict['risk_context']['categories'])}

RISK QUANTIFICATION REQUIRED:
{self._format_dict(prompt_dict['risk_quantification'])}

MITIGATION STRATEGIES REQUIRED:
{self._format_dict(prompt_dict['mitigation_strategies'])}

MONITORING PLAN REQUIRED:
{self._format_dict(prompt_dict['monitoring_plan'])}

OUTPUT FORMAT:
Provide comprehensive risk assessment report:
1. Executive Summary (risk overview)
2. Quantitative Risk Analysis (with financial impact)
3. Qualitative Risk Assessment
4. Scenario Analysis (base, stressed, adverse)
5. Mitigation Recommendations
6. Ongoing Monitoring Plan
7. Risk Dashboard Metrics

Include risk scores, financial quantification, and actionable recommendations.
"""
        
        return formatted.strip()
    
    def _format_portfolio_prompt(
        self,
        prompt_dict: Dict[str, Any],
        decisions: List[ExplanationContext]
    ) -> str:
        """Format portfolio analysis prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

PORTFOLIO CONTEXT:
Total Decisions: {prompt_dict['portfolio_context']['total_decisions']}
Analysis Type: {prompt_dict['portfolio_context']['analysis_type']}
Portfolio Size: {prompt_dict['portfolio_context']['portfolio_size']}

ANALYSIS REQUIREMENTS:
{self._format_dict(prompt_dict['analysis_requirements'])}

AGGREGATION METHODS:
{self._format_dict(prompt_dict['aggregation_methods'])}

TREND ANALYSIS:
{self._format_dict(prompt_dict['trend_analysis'])}

PORTFOLIO SUMMARY:
[Individual decision details would be provided here]

OUTPUT FORMAT:
Provide portfolio-level analysis report:
1. Executive Summary (portfolio health)
2. Aggregate Performance Metrics
3. Segment Analysis
4. Trend Identification
5. Risk Concentration Analysis
6. Optimization Recommendations
7. Strategic Insights

Focus on portfolio-level patterns, not individual decisions.
"""
        
        return formatted.strip()
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for readable prompt"""
        if isinstance(d, dict):
            return '\n'.join(f"- {k}: {v}" for k, v in d.items())
        return str(d)
    
    def _load_banker_templates(self) -> Dict[str, str]:
        """Load banker-specific templates"""
        # In production, these would be loaded from a database
        return {
            'technical_header': "Technical Analysis Report",
            'business_header': "Business Impact Analysis",
            'compliance_header': "Regulatory Compliance Report"
        }


def create_banker_prompt_generator() -> BankerPromptGenerator:
    """Convenience function to create banker prompt generator"""
    return BankerPromptGenerator()