"""
User-Specific Prompt Templates
Generates prompts for explaining AI decisions to end-users at different literacy levels.
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
class UserProfile:
    """Profile information about the user"""
    user_id: str
    literacy_level: str  # 'beginner', 'intermediate', 'advanced'
    preferred_language: str = 'en'
    preferred_format: str = 'narrative'  # 'narrative', 'bullet_points', 'visual'
    
    # Context
    previous_decisions: Optional[List[Dict]] = None
    customer_segment: Optional[str] = None  # 'new', 'existing', 'premium'
    interaction_history: Optional[Dict] = None


class UserPromptGenerator:
    """
    Generates user-friendly prompts for explaining AI decisions.
    Tailored for different user literacy levels and contexts.
    """
    
    def __init__(self):
        """Initialize user prompt generator"""
        self.base_generator = PromptGenerator()
        self.templates = self._load_user_templates()
    
    def generate_decision_explanation_prompt(
        self,
        context: ExplanationContext,
        user_profile: UserProfile,
        include_improvement_tips: bool = True,
        include_comparison: bool = False
    ) -> Dict[str, Any]:
        """
        Generate prompt for explaining a decision to a user.
        
        Args:
            context: Explanation context with decision details
            user_profile: User profile information
            include_improvement_tips: Whether to include actionable tips
            include_comparison: Whether to compare with similar cases
            
        Returns:
            Complete prompt dictionary
        """
        # Map literacy level to audience type
        audience = self._map_literacy_to_audience(user_profile.literacy_level)
        context.audience = audience
        context.language = user_profile.preferred_language
        
        # Generate base prompt
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.DECISION,
            include_examples=True
        )
        
        # Add user-specific customizations
        base_prompt['user_context'] = self._build_user_context(user_profile)
        base_prompt['tone_guidelines'] = self._get_tone_guidelines(user_profile, context)
        
        # Add improvement tips section if requested
        if include_improvement_tips:
            base_prompt['improvement_section'] = self._create_improvement_section(
                context, user_profile
            )
        
        # Add comparison section if requested
        if include_comparison:
            base_prompt['comparison_section'] = self._create_comparison_section(
                context, user_profile
            )
        
        # Add empathy and support guidelines
        base_prompt['empathy_guidelines'] = self._get_empathy_guidelines(
            context, user_profile
        )
        
        # Create the complete formatted prompt
        base_prompt['formatted_prompt'] = self._format_user_prompt(base_prompt, context)
        
        return base_prompt
    
    def generate_improvement_prompt(
        self,
        context: ExplanationContext,
        user_profile: UserProfile,
        time_horizon: str = '3-6 months'
    ) -> Dict[str, Any]:
        """
        Generate prompt for improvement recommendations.
        
        Args:
            context: Explanation context
            user_profile: User profile
            time_horizon: Expected time to see improvements
            
        Returns:
            Prompt dictionary with improvement recommendations
        """
        audience = self._map_literacy_to_audience(user_profile.literacy_level)
        context.audience = audience
        
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.IMPROVEMENT,
            include_examples=True
        )
        
        # Add improvement-specific sections
        base_prompt['action_plan'] = {
            'time_horizon': time_horizon,
            'priority_levels': ['immediate', 'short_term', 'long_term'],
            'success_metrics': self._define_success_metrics(context)
        }
        
        base_prompt['motivation'] = self._create_motivation_section(user_profile)
        
        base_prompt['formatted_prompt'] = self._format_improvement_prompt(
            base_prompt, context, time_horizon
        )
        
        return base_prompt
    
    def generate_fairness_explanation_prompt(
        self,
        context: ExplanationContext,
        user_profile: UserProfile,
        fairness_concern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate prompt for explaining fairness to users.
        
        Args:
            context: Explanation context
            user_profile: User profile
            fairness_concern: Specific fairness concern raised by user
            
        Returns:
            Prompt dictionary for fairness explanation
        """
        audience = self._map_literacy_to_audience(user_profile.literacy_level)
        context.audience = audience
        
        base_prompt = self.base_generator.generate_explanation_prompt(
            context,
            explanation_type=ExplanationType.FAIRNESS,
            include_examples=True
        )
        
        # Add fairness-specific sections
        base_prompt['fairness_context'] = {
            'user_concern': fairness_concern,
            'explanation_approach': 'accessible',
            'avoid_technical_jargon': user_profile.literacy_level == 'beginner'
        }
        
        base_prompt['transparency_section'] = self._create_transparency_section(
            context, user_profile
        )
        
        base_prompt['formatted_prompt'] = self._format_fairness_prompt(
            base_prompt, context, fairness_concern
        )
        
        return base_prompt
    
    def _map_literacy_to_audience(self, literacy_level: str) -> AudienceType:
        """Map user literacy level to audience type"""
        mapping = {
            'beginner': AudienceType.USER_BEGINNER,
            'intermediate': AudienceType.USER_INTERMEDIATE,
            'advanced': AudienceType.USER_ADVANCED
        }
        return mapping.get(literacy_level.lower(), AudienceType.USER_INTERMEDIATE)
    
    def _build_user_context(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Build user context for personalization"""
        return {
            'literacy_level': user_profile.literacy_level,
            'customer_segment': user_profile.customer_segment,
            'preferred_format': user_profile.preferred_format,
            'language': user_profile.preferred_language,
            'has_history': user_profile.previous_decisions is not None
        }
    
    def _get_tone_guidelines(
        self,
        user_profile: UserProfile,
        context: ExplanationContext
    ) -> Dict[str, str]:
        """Get tone guidelines based on decision outcome"""
        
        if context.outcome.lower() in ['denied', 'rejected', 'flagged']:
            # Negative outcome - be extra empathetic
            if user_profile.literacy_level == 'beginner':
                return {
                    'primary_tone': 'empathetic and supportive',
                    'approach': 'focus on actionable next steps',
                    'avoid': 'technical jargon, blame language, hopelessness',
                    'emphasize': 'improvement possibilities, timeline, resources'
                }
            else:
                return {
                    'primary_tone': 'professional and constructive',
                    'approach': 'explain rationale clearly with improvement path',
                    'avoid': 'overly simplified language',
                    'emphasize': 'specific factors, data-driven recommendations'
                }
        else:
            # Positive outcome
            return {
                'primary_tone': 'positive and informative',
                'approach': 'explain why decision was favorable',
                'avoid': 'complacency',
                'emphasize': 'factors that led to approval, maintaining good standing'
            }
    
    def _create_improvement_section(
        self,
        context: ExplanationContext,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Create improvement recommendations section"""
        
        # Analyze which factors can be improved
        improvable_factors = []
        for factor in context.top_factors[:5]:
            if self._is_improvable_factor(factor.get('name', '')):
                improvable_factors.append(factor)
        
        section = {
            'instruction': 'Provide specific, actionable steps to improve the outcome',
            'factors_to_address': [f['name'] for f in improvable_factors],
            'format': 'step_by_step' if user_profile.literacy_level == 'beginner' else 'prioritized_list',
            'include_timeline': True,
            'include_resources': True,
            'realistic_expectations': True
        }
        
        return section
    
    def _is_improvable_factor(self, factor_name: str) -> bool:
        """Check if a factor is improvable by the user"""
        # Some factors can't be changed (e.g., age, past history)
        non_improvable = ['age', 'birth', 'historical', 'past']
        
        factor_lower = factor_name.lower()
        return not any(term in factor_lower for term in non_improvable)
    
    def _create_comparison_section(
        self,
        context: ExplanationContext,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Create comparison with similar cases section"""
        
        return {
            'instruction': 'Compare this decision with similar cases to provide context',
            'comparison_type': 'similar_profiles',
            'include_statistics': user_profile.literacy_level != 'beginner',
            'anonymize': True,
            'focus': 'what makes successful applications different'
        }
    
    def _get_empathy_guidelines(
        self,
        context: ExplanationContext,
        user_profile: UserProfile
    ) -> List[str]:
        """Get empathy guidelines for negative outcomes"""
        
        if context.outcome.lower() in ['denied', 'rejected', 'flagged']:
            return [
                'Acknowledge the user\'s disappointment',
                'Emphasize that this is not a permanent situation',
                'Provide clear path forward',
                'Avoid language that sounds judgmental or dismissive',
                'Frame improvements as achievable goals',
                'Offer encouragement and support resources'
            ]
        else:
            return [
                'Congratulate the user appropriately',
                'Explain why they qualified',
                'Provide tips for maintaining good standing',
                'Be professional without being condescending'
            ]
    
    def _create_motivation_section(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Create motivational section for improvement"""
        
        return {
            'instruction': 'Provide motivation and encouragement',
            'elements': [
                'Success stories of others who improved',
                'Typical timeline for seeing results',
                'Small wins to celebrate along the way',
                'Support resources available'
            ],
            'tone': 'encouraging but realistic'
        }
    
    def _create_transparency_section(
        self,
        context: ExplanationContext,
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Create transparency section for fairness"""
        
        return {
            'instruction': 'Explain how fairness is ensured in the decision process',
            'include': [
                'What factors were considered',
                'What factors were NOT considered (e.g., protected attributes)',
                'How the model is monitored for bias',
                'Recourse options if user believes decision was unfair'
            ],
            'complexity': 'simple' if user_profile.literacy_level == 'beginner' else 'detailed'
        }
    
    def _define_success_metrics(self, context: ExplanationContext) -> List[str]:
        """Define success metrics for improvement"""
        
        metrics = [
            'Improvement in key factors identified',
            'Increased approval probability',
            'Better financial health indicators'
        ]
        
        return metrics
    
    def _format_user_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext
    ) -> str:
        """Format complete user-facing prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

USER CONTEXT:
- Literacy Level: {prompt_dict['user_context']['literacy_level']}
- Customer Segment: {prompt_dict['user_context']['customer_segment']}
- Preferred Format: {prompt_dict['user_context']['preferred_format']}

TONE GUIDELINES:
{self._format_dict(prompt_dict['tone_guidelines'])}

EMPATHY GUIDELINES:
{chr(10).join(f'- {guideline}' for guideline in prompt_dict['empathy_guidelines'])}

MAIN TASK:
{prompt_dict['task']}

EXPLANATION CONTEXT:
Decision: {context.outcome} ({context.confidence:.1%} confidence)
Top Factors:
{chr(10).join(f'- {f["name"]}: {f.get("impact", "N/A")}' for f in context.top_factors[:5])}

"""
        
        if 'improvement_section' in prompt_dict:
            formatted += f"""
IMPROVEMENT RECOMMENDATIONS:
{self._format_dict(prompt_dict['improvement_section'])}
"""
        
        if 'comparison_section' in prompt_dict:
            formatted += f"""
COMPARISON CONTEXT:
{self._format_dict(prompt_dict['comparison_section'])}
"""
        
        formatted += f"""
OUTPUT FORMAT:
{self._format_dict(prompt_dict['expected_output_format'])}

CONSTRAINTS:
- Maximum length: {prompt_dict['constraints'].get('max_length', 'No limit')}
- Tone: {prompt_dict['constraints']['tone']}
- Complexity: {prompt_dict['constraints']['complexity']}
- Avoid: {', '.join(prompt_dict['constraints'].get('avoid', []))}
"""
        
        return formatted.strip()
    
    def _format_improvement_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext,
        time_horizon: str
    ) -> str:
        """Format improvement-focused prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

TASK: Provide Improvement Recommendations
Decision: {context.outcome}
Time Horizon: {time_horizon}

CURRENT SITUATION:
{chr(10).join(f'- {f["name"]}: {f.get("value", "N/A")}' for f in context.top_factors[:5])}

ACTION PLAN REQUIREMENTS:
Time Horizon: {prompt_dict['action_plan']['time_horizon']}
Priority Levels: {', '.join(prompt_dict['action_plan']['priority_levels'])}

MOTIVATION SECTION:
{self._format_dict(prompt_dict['motivation'])}

OUTPUT FORMAT: Provide a clear action plan with:
1. Immediate actions (next 30 days)
2. Short-term goals (1-3 months)
3. Long-term improvements (3-6 months)

Each action should include:
- Specific step to take
- Expected impact
- Resources/support available
- How to track progress
"""
        
        return formatted.strip()
    
    def _format_fairness_prompt(
        self,
        prompt_dict: Dict[str, Any],
        context: ExplanationContext,
        fairness_concern: Optional[str]
    ) -> str:
        """Format fairness explanation prompt"""
        
        formatted = f"""
SYSTEM: {prompt_dict['system']}

TASK: Address Fairness Concerns
Decision: {context.outcome}
"""
        
        if fairness_concern:
            formatted += f"User's Concern: {fairness_concern}\n"
        
        formatted += f"""
FAIRNESS CONTEXT:
{self._format_dict(prompt_dict['fairness_context'])}

TRANSPARENCY REQUIREMENTS:
{self._format_dict(prompt_dict['transparency_section'])}

OUTPUT FORMAT: Provide clear explanation that:
1. Addresses the user's specific concern
2. Explains what factors were and were NOT considered
3. Describes fairness monitoring processes
4. Provides recourse options if applicable

Tone: {prompt_dict['constraints']['tone']}
Avoid technical jargon: {prompt_dict['fairness_context']['avoid_technical_jargon']}
"""
        
        return formatted.strip()
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for readable prompt"""
        return '\n'.join(f"- {k}: {v}" for k, v in d.items())
    
    def _load_user_templates(self) -> Dict[str, str]:
        """Load user-specific templates"""
        # In production, these would be loaded from a database
        return {
            'beginner_greeting': "Let me explain this in simple terms...",
            'intermediate_greeting': "Here's what happened with your application...",
            'advanced_greeting': "Here's a detailed analysis of your decision..."
        }


def create_user_prompt_generator() -> UserPromptGenerator:
    """Convenience function to create user prompt generator"""
    return UserPromptGenerator()