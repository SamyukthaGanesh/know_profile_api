"""
Prompt Generator for AI Literacy Assistant
Generates structured prompts for LLM-based explanations without calling the LLM.
Model-agnostic and data-agnostic prompt generation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class AudienceType(Enum):
    """Target audience for explanations"""
    USER_BEGINNER = "user_beginner"
    USER_INTERMEDIATE = "user_intermediate" 
    USER_ADVANCED = "user_advanced"
    BANKER_TECHNICAL = "banker_technical"
    BANKER_BUSINESS = "banker_business"
    REGULATOR = "regulator"
    AUDITOR = "auditor"


class ExplanationType(Enum):
    """Type of explanation needed"""
    DECISION = "decision"
    FAIRNESS = "fairness"
    DRIFT = "drift"
    COMPLIANCE = "compliance"
    IMPROVEMENT = "improvement"


@dataclass
class ExplanationContext:
    """Context for generating explanations"""
    # Decision details
    decision_id: str
    decision_type: str  # loan, fraud_detection, credit_score, etc.
    outcome: str  # approved, denied, flagged, etc.
    confidence: float
    
    # Model information
    model_name: str
    model_version: str
    
    # Feature information
    top_factors: List[Dict[str, Any]]  # From SHAP/LIME
    feature_values: Dict[str, Any]
    feature_importance: Dict[str, float]
    
    # User/requester information
    audience: AudienceType
    language: str = "en"
    
    # Additional context
    fairness_metrics: Optional[Dict[str, Any]] = None
    historical_decisions: Optional[List[Dict]] = None
    regulatory_requirements: Optional[List[str]] = None
    industry_benchmarks: Optional[Dict[str, Any]] = None


class PromptGenerator:
    """
    Generates structured prompts for LLM-based explanations.
    Does NOT call any LLM - only creates the prompts.
    """
    
    def __init__(self):
        """Initialize prompt generator with templates"""
        self.templates = self._load_templates()
        self.instruction_sets = self._load_instruction_sets()
        
    def generate_explanation_prompt(
        self,
        context: ExplanationContext,
        explanation_type: ExplanationType = ExplanationType.DECISION,
        include_examples: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete prompt for explanation.
        
        Args:
            context: Explanation context with all necessary information
            explanation_type: Type of explanation to generate
            include_examples: Whether to include few-shot examples
            max_length: Maximum response length in tokens/words
            
        Returns:
            Dictionary containing the complete prompt structure
        """
        # Select appropriate instruction set based on audience
        instructions = self._get_instructions(context.audience, explanation_type)
        
        # Build the system prompt
        system_prompt = self._build_system_prompt(context.audience)
        
        # Build the main prompt
        main_prompt = self._build_main_prompt(context, explanation_type)
        
        # Add examples if requested
        examples = []
        if include_examples:
            examples = self._get_examples(context.audience, explanation_type)
        
        # Add constraints and guidelines
        constraints = self._get_constraints(context.audience, max_length)
        
        # Combine into final prompt structure
        prompt = {
            "system": system_prompt,
            "instructions": instructions,
            "context": self._format_context(context),
            "task": main_prompt,
            "examples": examples,
            "constraints": constraints,
            "expected_output_format": self._get_output_format(context.audience),
            "metadata": {
                "audience": context.audience.value,
                "explanation_type": explanation_type.value,
                "decision_id": context.decision_id,
                "timestamp": None  # Will be set when actually used
            }
        }
        
        # Add the complete formatted prompt as a single string option
        prompt["formatted_prompt"] = self._format_complete_prompt(prompt)
        
        return prompt
    
    def _build_system_prompt(self, audience: AudienceType) -> str:
        """Build system prompt based on audience"""
        system_prompts = {
            AudienceType.USER_BEGINNER: """You are a friendly and patient financial advisor explaining AI decisions to someone with no technical background. 
Use simple language, avoid jargon, and relate concepts to everyday experiences. 
Be empathetic and supportive, especially for negative decisions.""",
            
            AudienceType.USER_INTERMEDIATE: """You are a knowledgeable financial advisor explaining AI decisions to someone with basic financial literacy.
You can use some technical terms but always explain them. 
Provide actionable insights and practical recommendations.""",
            
            AudienceType.USER_ADVANCED: """You are a technical expert explaining AI decisions to someone with strong financial and technical knowledge.
Be precise, include technical details, and provide comprehensive analysis.
Include metrics, statistical significance, and methodological details.""",
            
            AudienceType.BANKER_TECHNICAL: """You are an AI systems expert providing technical explanations to banking professionals.
Focus on model performance, risk metrics, and technical implementation details.
Include SHAP values, feature importance, and statistical analysis.""",
            
            AudienceType.BANKER_BUSINESS: """You are a business analyst explaining AI decisions to banking executives.
Focus on business impact, risk management, and strategic implications.
Translate technical metrics into business KPIs and ROI.""",
            
            AudienceType.REGULATOR: """You are a compliance expert explaining AI decisions to regulatory authorities.
Emphasize transparency, fairness, auditability, and regulatory compliance.
Include all relevant metrics, testing procedures, and governance measures.""",
            
            AudienceType.AUDITOR: """You are an audit specialist providing detailed explanations for audit purposes.
Focus on decision traceability, data lineage, and model governance.
Include all technical details necessary for audit trail."""
        }
        
        return system_prompts.get(audience, system_prompts[AudienceType.USER_INTERMEDIATE])
    
    def _build_main_prompt(self, context: ExplanationContext, explanation_type: ExplanationType) -> str:
        """Build the main task prompt"""
        decision_info = f"""
        Decision Type: {context.decision_type}
        Outcome: {context.outcome}
        Confidence: {context.confidence:.1%}
        Model: {context.model_name} (Version: {context.model_version})
        """
        
        # Format top factors
        factors_text = self._format_factors(context.top_factors, context.audience)
        
        prompts = {
            ExplanationType.DECISION: f"""Explain the following AI decision:
{decision_info}

Top Contributing Factors:
{factors_text}

Provide a clear explanation of why this decision was made, what factors were most important, 
and what this means for the recipient.""",
            
            ExplanationType.FAIRNESS: f"""Analyze the fairness of the following AI decision:
{decision_info}

Fairness Metrics:
{json.dumps(context.fairness_metrics, indent=2) if context.fairness_metrics else 'Not provided'}

Explain whether this decision was fair, how it compares to similar cases, 
and any potential bias concerns.""",
            
            ExplanationType.IMPROVEMENT: f"""Provide improvement recommendations for the following decision:
{decision_info}

Current Feature Values:
{json.dumps(context.feature_values, indent=2)}

Top Factors:
{factors_text}

Suggest specific, actionable steps to improve the outcome, including timelines and expected impact.""",
            
            ExplanationType.DRIFT: f"""Analyze model drift for the following decision:
{decision_info}

Historical Context:
{json.dumps(context.historical_decisions, indent=2) if context.historical_decisions else 'Not provided'}

Explain any detected drift, its potential impact, and recommended actions.""",
            
            ExplanationType.COMPLIANCE: f"""Provide compliance analysis for the following decision:
{decision_info}

Regulatory Requirements:
{', '.join(context.regulatory_requirements) if context.regulatory_requirements else 'Standard banking regulations'}

Confirm compliance status, identify any concerns, and provide documentation requirements."""
        }
        
        return prompts.get(explanation_type, prompts[ExplanationType.DECISION])
    
    def _format_factors(self, factors: List[Dict[str, Any]], audience: AudienceType) -> str:
        """Format factors based on audience"""
        if not factors:
            return "No factors provided"
        
        formatted = []
        for i, factor in enumerate(factors[:5], 1):  # Top 5 factors
            if audience in [AudienceType.USER_BEGINNER, AudienceType.USER_INTERMEDIATE]:
                # Simplified format
                impact = factor.get('impact', 0)
                # Handle both string and numeric impact values
                if isinstance(impact, str):
                    impact_text = 'Positive' if impact.lower() == 'positive' else 'Negative'
                else:
                    impact_text = 'Positive' if impact > 0 else 'Negative'
                
                formatted.append(
                    f"{i}. {factor.get('name', 'Unknown')}: "
                    f"{impact_text} impact "
                    f"(importance: {abs(factor.get('importance', 0)):.1%})"
                )
            else:
                # Technical format
                impact = factor.get('impact', 0)
                # Handle both string and numeric impact values for technical format
                if isinstance(impact, str):
                    impact_value = impact
                else:
                    impact_value = f"{impact:.4f}"
                
                formatted.append(
                    f"{i}. {factor.get('name', 'Unknown')}: "
                    f"SHAP={factor.get('shap_value', 0):.4f}, "
                    f"Value={factor.get('value', 'N/A')}, "
                    f"Impact={impact_value}"
                )
        
        return "\n".join(formatted)
    
    def _format_context(self, context: ExplanationContext) -> Dict[str, Any]:
        """Format context for prompt"""
        return {
            "decision": {
                "id": context.decision_id,
                "type": context.decision_type,
                "outcome": context.outcome,
                "confidence": context.confidence
            },
            "model": {
                "name": context.model_name,
                "version": context.model_version
            },
            "features": {
                "top_factors": context.top_factors[:5] if context.top_factors else [],
                "importance_scores": dict(sorted(
                    context.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]) if context.feature_importance else {}
            },
            "audience": context.audience.value,
            "language": context.language
        }
    
    def _get_instructions(self, audience: AudienceType, explanation_type: ExplanationType) -> List[str]:
        """Get specific instructions based on audience and explanation type"""
        base_instructions = [
            "Provide accurate and truthful explanations",
            "Base explanations on the provided data and metrics",
            "Maintain appropriate tone for the audience",
            "Structure the response clearly"
        ]
        
        audience_instructions = {
            AudienceType.USER_BEGINNER: [
                "Use simple, everyday language",
                "Avoid technical jargon",
                "Use analogies and examples",
                "Be encouraging and supportive",
                "Keep sentences short and clear"
            ],
            AudienceType.USER_INTERMEDIATE: [
                "Balance technical accuracy with clarity",
                "Define technical terms when first used",
                "Provide actionable insights",
                "Include relevant context"
            ],
            AudienceType.USER_ADVANCED: [
                "Include technical details and metrics",
                "Provide statistical analysis",
                "Reference specific model components",
                "Include confidence intervals and p-values"
            ],
            AudienceType.BANKER_TECHNICAL: [
                "Focus on model performance metrics",
                "Include technical implementation details",
                "Provide risk analysis",
                "Reference industry standards"
            ],
            AudienceType.BANKER_BUSINESS: [
                "Translate technical metrics to business impact",
                "Focus on ROI and KPIs",
                "Include competitive analysis",
                "Provide strategic recommendations"
            ],
            AudienceType.REGULATOR: [
                "Emphasize compliance and governance",
                "Include all relevant regulations",
                "Provide audit trail information",
                "Focus on fairness and transparency"
            ]
        }
        
        type_instructions = {
            ExplanationType.DECISION: [
                "Explain the primary factors",
                "Clarify the decision logic",
                "Provide clear reasoning"
            ],
            ExplanationType.FAIRNESS: [
                "Analyze demographic parity",
                "Check for disparate impact",
                "Compare to baseline rates"
            ],
            ExplanationType.IMPROVEMENT: [
                "Provide specific actions",
                "Include realistic timelines",
                "Estimate impact of changes"
            ]
        }
        
        return base_instructions + audience_instructions.get(audience, []) + type_instructions.get(explanation_type, [])
    
    def _get_examples(self, audience: AudienceType, explanation_type: ExplanationType) -> List[Dict[str, str]]:
        """Get few-shot examples for the prompt"""
        # Examples would be loaded from a database or file
        # This is a simplified version
        if audience == AudienceType.USER_BEGINNER and explanation_type == ExplanationType.DECISION:
            return [{
                "input": "Loan denied with 78% confidence. Main factor: credit score 620.",
                "output": "I understand this may be disappointing. Your loan wasn't approved this time, "
                         "mainly because your credit score of 620 is below what we typically need (700+). "
                         "Think of a credit score like a grade for how well you've managed borrowed money - "
                         "the higher the better! The good news is you can improve this by paying bills on time "
                         "and reducing credit card balances. Many people improve their scores by 50-80 points "
                         "in just 3-6 months with consistent effort."
            }]
        
        # Add more examples for different combinations
        return []
    
    def _get_constraints(self, audience: AudienceType, max_length: Optional[int]) -> Dict[str, Any]:
        """Get constraints for the response"""
        constraints = {
            "tone": self._get_tone(audience),
            "complexity": self._get_complexity_level(audience),
            "format": self._get_format_requirements(audience)
        }
        
        if max_length:
            constraints["max_length"] = max_length
        
        # Audience-specific constraints
        if audience in [AudienceType.USER_BEGINNER, AudienceType.USER_INTERMEDIATE]:
            constraints["avoid"] = ["complex mathematics", "technical jargon", "statistical formulas"]
            constraints["include"] = ["clear summary", "next steps", "encouragement"]
        elif audience in [AudienceType.BANKER_TECHNICAL, AudienceType.REGULATOR]:
            constraints["include"] = ["technical metrics", "statistical significance", "confidence intervals"]
            constraints["format"]["citations"] = True
        
        return constraints
    
    def _get_tone(self, audience: AudienceType) -> str:
        """Get appropriate tone for audience"""
        tones = {
            AudienceType.USER_BEGINNER: "friendly, patient, encouraging",
            AudienceType.USER_INTERMEDIATE: "professional, helpful, informative",
            AudienceType.USER_ADVANCED: "technical, precise, comprehensive",
            AudienceType.BANKER_TECHNICAL: "technical, analytical, detailed",
            AudienceType.BANKER_BUSINESS: "strategic, results-oriented, executive",
            AudienceType.REGULATOR: "formal, compliant, thorough",
            AudienceType.AUDITOR: "precise, detailed, evidence-based"
        }
        return tones.get(audience, "professional")
    
    def _get_complexity_level(self, audience: AudienceType) -> str:
        """Get complexity level for audience"""
        if audience == AudienceType.USER_BEGINNER:
            return "simple"
        elif audience == AudienceType.USER_INTERMEDIATE:
            return "moderate"
        else:
            return "advanced"
    
    def _get_format_requirements(self, audience: AudienceType) -> Dict[str, Any]:
        """Get format requirements for audience"""
        if audience in [AudienceType.USER_BEGINNER, AudienceType.USER_INTERMEDIATE]:
            return {
                "structure": "paragraph",
                "lists": "bulleted for key points",
                "sections": ["summary", "main_explanation", "next_steps"]
            }
        else:
            return {
                "structure": "sections with headers",
                "lists": "numbered with sub-points",
                "sections": ["executive_summary", "technical_analysis", "metrics", "recommendations"],
                "include_tables": True
            }
    
    def _get_output_format(self, audience: AudienceType) -> Dict[str, str]:
        """Define expected output format"""
        if audience in [AudienceType.USER_BEGINNER, AudienceType.USER_INTERMEDIATE, AudienceType.USER_ADVANCED]:
            return {
                "type": "narrative",
                "sections": {
                    "summary": "Brief overview of the decision",
                    "factors": "Main factors explained clearly",
                    "meaning": "What this means for you",
                    "next_steps": "Actionable recommendations"
                }
            }
        else:
            return {
                "type": "structured",
                "sections": {
                    "executive_summary": "High-level overview",
                    "technical_details": "Model metrics and analysis",
                    "risk_assessment": "Risk factors and mitigation",
                    "compliance": "Regulatory compliance status",
                    "recommendations": "Strategic recommendations"
                }
            }
    
    def _format_complete_prompt(self, prompt_dict: Dict[str, Any]) -> str:
        """Format the complete prompt as a single string"""
        formatted = f"""
SYSTEM: {prompt_dict['system']}

INSTRUCTIONS:
{chr(10).join(f'- {inst}' for inst in prompt_dict['instructions'])}

CONTEXT:
{json.dumps(prompt_dict['context'], indent=2)}

TASK:
{prompt_dict['task']}

CONSTRAINTS:
{json.dumps(prompt_dict['constraints'], indent=2)}

EXPECTED OUTPUT FORMAT:
{json.dumps(prompt_dict['expected_output_format'], indent=2)}
"""
        
        if prompt_dict['examples']:
            formatted += "\n\nEXAMPLES:\n"
            for i, example in enumerate(prompt_dict['examples'], 1):
                formatted += f"\nExample {i}:\nInput: {example['input']}\nOutput: {example['output']}\n"
        
        return formatted.strip()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates - would be from database in production"""
        return {}
    
    def _load_instruction_sets(self) -> Dict[str, List[str]]:
        """Load instruction sets - would be from database in production"""
        return {}


def create_prompt_generator() -> PromptGenerator:
    """Convenience function to create a prompt generator"""
    return PromptGenerator()