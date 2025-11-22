import React, { useState, useRef, useEffect } from 'react';
import { Button } from '../shared/Button';
import './AIChatbot.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface AIChatbotProps {
  apiEndpoint?: string; // Your teammate can pass their API endpoint
  onSendMessage?: (message: string) => Promise<string>; // Custom handler for sending messages
}

export const AIChatbot: React.FC<AIChatbotProps> = ({ apiEndpoint, onSendMessage }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hi! I\'m your TrustBank AI assistant. How can I help you today?',
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages([...messages, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      let botResponse: string;

      // Use custom handler if provided, otherwise use default logic
      if (onSendMessage) {
        botResponse = await onSendMessage(inputValue);
      } else if (apiEndpoint) {
        // Call your teammate's API
        const response = await fetch(apiEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: inputValue })
        });
        const data = await response.json();
        botResponse = data.response || 'I\'m not sure how to help with that.';
      } else {
        // Default mock response
        botResponse = getMockResponse(inputValue);
      }

      // Simulate typing delay
      setTimeout(() => {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: botResponse,
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages((prev) => [...prev, botMessage]);
        setIsTyping(false);
      }, 1000);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages((prev) => [...prev, errorMessage]);
      setIsTyping(false);
    }
  };

  const getMockResponse = (input: string): string => {
    const lowerInput = input.toLowerCase();

    if (lowerInput.includes('credit score') || lowerInput.includes('score')) {
      return 'Your credit score is calculated based on payment history, credit utilization, length of credit history, and other factors. I can help you understand what affects your score!';
    }

    if (lowerInput.includes('loan') || lowerInput.includes('borrow')) {
      return 'Based on your profile, you have a good chance of loan approval! Your current DTI ratio and credit score are within healthy ranges. Would you like to know more about loan options?';
    }

    if (lowerInput.includes('dti') || lowerInput.includes('debt')) {
      return 'DTI (Debt-to-Income) ratio is the percentage of your monthly income that goes toward debt payments. A ratio below 30% is considered healthy. Your current DTI is in good shape!';
    }

    if (lowerInput.includes('savings') || lowerInput.includes('save')) {
      return 'I recommend maintaining an emergency fund of 3-6 months of expenses. You\'re currently earning 3.5% interest on your savings. Consider exploring FDs or mutual funds for better returns!';
    }

    if (lowerInput.includes('hello') || lowerInput.includes('hi')) {
      return 'Hello! I\'m here to help with questions about your financial profile, credit score, loans, and more. What would you like to know?';
    }

    if (lowerInput.includes('help') || lowerInput.includes('what can you')) {
      return 'I can help you with:\n\n• Understanding your credit score\n• Explaining your DTI ratio\n• Loan eligibility questions\n• Investment advice\n• Transaction queries\n• General banking questions\n\nWhat would you like to know more about?';
    }

    return 'That\'s a great question! I\'m here to help with your banking needs. Could you provide more details or try asking about your credit score, loans, savings, or transactions?';
  };

  const quickActions = [
    'What\'s my credit score?',
    'Explain my DTI ratio',
    'Am I eligible for a loan?',
    'How can I improve my score?'
  ];

  const handleQuickAction = (action: string) => {
    setInputValue(action);
  };

  return (
    <>
      {/* Chat Toggle Button */}
      <button
        className={`chatbot-toggle ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle AI Chatbot"
      >
        {isOpen ? (
          <span className="toggle-icon">✕</span>
        ) : (
          <>
            <span className="toggle-icon">💬</span>
            <span className="toggle-badge">AI</span>
          </>
        )}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="chatbot-window">
          {/* Header */}
          <div className="chatbot-header">
            <div className="chatbot-header-info">
              <span className="chatbot-avatar">🤖</span>
              <div>
                <h3 className="chatbot-title">TrustBank AI Assistant</h3>
                <p className="chatbot-status">
                  <span className="status-dot"></span>
                  Online
                </p>
              </div>
            </div>
            <button
              className="chatbot-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div className="chatbot-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.sender === 'user' ? 'message-user' : 'message-bot'}`}
              >
                {message.sender === 'bot' && (
                  <span className="message-avatar">🤖</span>
                )}
                <div className="message-bubble">
                  <p className="message-text">{message.text}</p>
                  <span className="message-time">
                    {message.timestamp.toLocaleTimeString('en-US', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>
                {message.sender === 'user' && (
                  <span className="message-avatar user-avatar">👤</span>
                )}
              </div>
            ))}

            {isTyping && (
              <div className="message message-bot">
                <span className="message-avatar">🤖</span>
                <div className="message-bubble typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Actions */}
          {messages.length === 1 && (
            <div className="chatbot-quick-actions">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  className="quick-action-btn"
                  onClick={() => handleQuickAction(action)}
                >
                  {action}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="chatbot-input">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your message..."
              disabled={isTyping}
            />
            <button
              className="send-button"
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isTyping}
              aria-label="Send message"
            >
              <span className="send-icon">➤</span>
            </button>
          </div>

          {/* Footer */}
          <div className="chatbot-footer">
            <span>Powered by TrustBank AI</span>
          </div>
        </div>
      )}
    </>
  );
};

