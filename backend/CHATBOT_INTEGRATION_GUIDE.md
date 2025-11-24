# 🤖 Chatbot Integration Guide - AI Decision Explanations

**For: LLM-Powered Customer Service Chatbot**  
**Use Case: Explaining loan denials, approvals, and other AI decisions**

---

## 📋 What We Can Provide Your Chatbot

Your AI governance framework provides **rich, structured data** that your chatbot can use to give customers detailed explanations about AI decisions.

---

## 🎯 Available Data for Loan Decisions

### 1. **Decision Explanation Data Structure**

```json
{
  "status": "rejected",
  "confidence": "78.3%",
  "decision_id": "LOAN_REJ_CUST_12345",
  "timestamp": "2025-11-22T15:30:00Z",
  
  "primary_reasons": [
    {
      "factor": "EXT_SOURCE_2",
      "explanation": "Your credit score contributed to this decision",
      "impact_score": 0.045,
      "your_value": -0.63,
      "impact_type": "negative"
    },
    {
      "factor": "DAYS_EMPLOYED", 
      "explanation": "Your employment history contributed to this decision",
      "impact_score": 0.032,
      "your_value": 500,
      "impact_type": "negative"
    },
    {
      "factor": "AMT_INCOME_TOTAL",
      "explanation": "Your income level contributed to this decision", 
      "impact_score": 0.028,
      "your_value": 35000,
      "impact_type": "negative"
    }
  ],
  
  "improvement_suggestions": [
    "Consider improving your credit score by paying bills on time and reducing debt",
    "Ensure employment history is well-documented and stable",
    "Provide additional documentation of stable income sources"
  ],
  
  "compliance_info": {
    "regulatory_requirements": [
      "This decision complies with Fair Credit Reporting Act (FCRA)",
      "Equal Credit Opportunity Act (ECOA) guidelines followed"
    ],
    "audit_trail_id": "AUDIT_abc123_1732287000",
    "policies_applied": ["BASEL_III_CREDIT_001", "FAIR_LENDING_AGE_001"]
  },
  
  "next_steps": [
    "Review the factors that affected your application",
    "Consider improving the highlighted areas",
    "You may reapply after addressing these factors", 
    "Contact our customer service for personalized guidance"
  ],
  
  "appeal_process": {
    "can_appeal": true,
    "appeal_deadline": "30 days from decision date",
    "required_documents": ["Updated income verification", "Credit report improvements"]
  }
}
```

---

## 🔌 API Endpoints for Your Chatbot

### **Get Decision Explanation**
```bash
POST /explain
{
  "model_name": "home_credit_model",
  "features": {
    "AMT_INCOME_TOTAL": 35000,
    "EXT_SOURCE_2": -0.63,
    "DAYS_EMPLOYED": 500,
    "AMT_CREDIT": 25000
  },
  "method": "shap",
  "user_id": "CUST_12345"
}
```

### **Generate User-Friendly Prompt** 
```bash
POST /prompts/generate
{
  "decision_id": "LOAN_REJ_CUST_12345",
  "decision_type": "loan_application",
  "outcome": "denied",
  "confidence": 0.783,
  "top_factors": [
    {"name": "credit_score", "importance": 0.045, "value": -0.63},
    {"name": "employment", "importance": 0.032, "value": 500}
  ],
  "audience_type": "user_beginner",
  "user_literacy_level": "beginner",
  "include_improvement_tips": true
}
```

### **Check Compliance Status**
```bash
POST /compliance/check
{
  "decision_id": "LOAN_REJ_CUST_12345",
  "features": {
    "AMT_INCOME_TOTAL": 35000,
    "AGE": 25,
    "CODE_GENDER": "M"
  }
}
```

---

## 🎭 Different Conversation Scenarios

### **Scenario 1: Customer asks "Why was my loan denied?"**

**Your chatbot can respond with:**
```
I understand your concern about the loan decision. Let me explain what happened:

Your application was declined with 78.3% confidence based on our AI model. The main factors were:

1. **Credit Score**: Your credit profile was a significant factor in this decision
2. **Employment History**: Your current employment duration affected the assessment  
3. **Income Level**: Your reported income relative to the loan amount was considered

**What you can do:**
• Improve your credit score by paying bills on time and reducing debt
• Provide additional employment documentation
• Consider a smaller loan amount

You have the right to appeal this decision within 30 days. Would you like me to help you with the appeal process?
```

### **Scenario 2: Customer asks "What can I do to get approved next time?"**

**Your chatbot can provide specific, actionable advice:**
```
Based on your application analysis, here are specific steps to improve your approval chances:

**Priority Actions:**
1. **Credit Score** (Highest Impact): 
   - Pay down existing credit card balances
   - Make all payments on time for the next 6 months
   - Consider a secured credit card if needed

2. **Employment Stability**:
   - Stay in your current job for at least 12 more months
   - Gather documentation of stable employment history

3. **Income Documentation**:
   - Provide recent pay stubs and tax returns
   - Include any additional income sources

**Timeline**: With these improvements, you could reapply in 6-12 months with better chances.

Would you like specific resources for credit improvement?
```

### **Scenario 3: Customer asks "Is this decision fair/legal?"**

**Your chatbot can provide compliance information:**
```
I understand your concern about fairness. Here's what I can tell you:

**Regulatory Compliance:**
✅ This decision followed Fair Credit Reporting Act (FCRA) guidelines
✅ Equal Credit Opportunity Act (ECOA) requirements were met  
✅ No discrimination based on age, gender, or protected characteristics

**Decision Transparency:**
• Our AI model is regularly audited for bias
• Decision factors are based on creditworthiness, not personal characteristics
• You have the right to know the specific reasons (which I've provided)

**Your Rights:**
• Right to free credit report to verify information
• Right to appeal within 30 days
• Right to file complaints with regulatory agencies if you believe there was discrimination

Audit Trail ID: AUDIT_abc123_1732287000 (for your records)

Would you like information about the appeals process?
```

---

## 💬 Chatbot Integration Examples

### **React/JavaScript Example**
```javascript
// In your chatbot component
const explainLoanDecision = async (customerId, decisionId) => {
  try {
    // Get explanation from your API
    const response = await fetch('/api/ai-governance/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        decision_id: decisionId,
        customer_id: customerId
      })
    });
    
    const explanation = await response.json();
    
    // Format for chatbot display
    const chatMessage = formatExplanationForChat(explanation);
    
    // Send to LLM for natural language response
    const llmResponse = await callLLM({
      system: "You are a helpful customer service agent explaining loan decisions.",
      user: `Customer asks: "Why was my loan denied?" 
             
             Data: ${JSON.stringify(explanation)}
             
             Provide a clear, empathetic explanation.`
    });
    
    return llmResponse;
    
  } catch (error) {
    console.error('Error explaining decision:', error);
    return "I apologize, but I'm having trouble accessing your decision details. Please contact our customer service team.";
  }
};

const formatExplanationForChat = (explanation) => {
  return {
    status: explanation.status,
    confidence: explanation.confidence,
    reasons: explanation.primary_reasons.map(r => r.explanation),
    suggestions: explanation.improvement_suggestions,
    canAppeal: explanation.appeal_process.can_appeal,
    nextSteps: explanation.next_steps
  };
};
```

### **Python/Flask Chatbot Backend Example**
```python
from ai_governance_framework.core.explainability.shap_explainer import SHAPExplainer
from ai_governance_framework.core.literacy.user_prompts import UserPromptGenerator

def handle_loan_explanation_request(customer_id, decision_id):
    """Handle customer request for loan decision explanation"""
    
    # Get explanation data
    explanation_data = get_decision_explanation(decision_id)
    
    # Generate user-friendly prompt
    prompt_generator = UserPromptGenerator()
    user_prompt = prompt_generator.generate_decision_explanation_prompt(
        context=explanation_data,
        include_improvement_tips=True
    )
    
    # Send to LLM (OpenAI, Claude, etc.)
    llm_response = call_llm({
        "system": "You are an empathetic customer service agent helping customers understand loan decisions.",
        "user": user_prompt['formatted_prompt']
    })
    
    # Add interactive elements
    response = {
        "message": llm_response,
        "quick_actions": [
            {"text": "How can I improve my credit?", "action": "credit_help"},
            {"text": "I want to appeal", "action": "appeal_process"},
            {"text": "Check my application status", "action": "status_check"}
        ],
        "can_appeal": explanation_data.get('appeal_process', {}).get('can_appeal', False),
        "audit_id": explanation_data.get('compliance_info', {}).get('audit_trail_id')
    }
    
    return response
```

---

## 🚀 Implementation Steps for Your Teammate

### **Step 1: Connect to Your API**
```javascript
// Base API configuration
const AI_GOVERNANCE_API = 'http://localhost:8000';

const getDecisionExplanation = async (decisionId) => {
  const response = await fetch(`${AI_GOVERNANCE_API}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision_id: decisionId })
  });
  return response.json();
};
```

### **Step 2: Format Data for LLM**
```javascript
const formatForLLM = (explanation) => {
  return `
Customer loan decision explanation:
Status: ${explanation.status}
Confidence: ${explanation.confidence}

Main factors:
${explanation.primary_reasons.map(r => `- ${r.explanation}`).join('\n')}

Improvement suggestions:
${explanation.improvement_suggestions.map(s => `- ${s}`).join('\n')}

Compliance: Decision follows ${explanation.compliance_info.regulatory_requirements.join(', ')}
`;
};
```

### **Step 3: Integrate with Your LLM**
```javascript
const generateChatResponse = async (customerQuery, explanation) => {
  const prompt = `
System: You are a helpful, empathetic customer service agent.

Customer asks: "${customerQuery}"

Decision data: ${formatForLLM(explanation)}

Provide a clear, helpful response that:
1. Acknowledges their concern
2. Explains the decision clearly
3. Offers specific next steps
4. Maintains a supportive tone
`;

  // Send to your LLM (OpenAI, Claude, etc.)
  const response = await yourLLMService.complete(prompt);
  return response;
};
```

---

## 🎯 Key Benefits for Your Chatbot

✅ **Rich Data**: Detailed explanations with specific factor analysis  
✅ **Compliance-Ready**: All decisions include regulatory compliance info  
✅ **Actionable Advice**: Specific improvement suggestions for customers  
✅ **Appeal Support**: Built-in appeal process information  
✅ **Audit Trail**: Every decision is traceable for customer protection  
✅ **Multiple Audiences**: Different explanation styles for different customer types  
✅ **Real-time**: API responses in milliseconds for chatbot speed  

---

## 📞 What to Tell Your Teammate

**"Our AI governance framework provides everything your chatbot needs to explain loan decisions:"**

1. **Structured explanation data** - JSON format with reasons, confidence scores, and improvement tips
2. **Ready-to-use API endpoints** - Just HTTP requests, returns clean JSON
3. **LLM-friendly prompts** - Pre-formatted prompts optimized for natural language generation  
4. **Compliance information** - Regulatory details for customer protection
5. **Appeal process data** - Built-in appeal workflow information
6. **Real-time performance** - Fast enough for chatbot conversations

**Integration is simple**: Make HTTP requests to our API, get structured data, format for your LLM, and respond to customers with detailed, compliant explanations.

**Sample conversation flow**: Customer asks → Your chatbot calls our API → Gets explanation data → Sends to LLM → LLM generates natural response → Customer gets helpful, detailed explanation

---

**Your teammate now has everything needed to integrate AI decision explanations into their chatbot! 🚀**