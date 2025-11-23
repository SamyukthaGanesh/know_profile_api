import React, { useState, useRef, useEffect } from 'react';
import './ChatWidget.css';
import './AdminChatStyles.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface AdminRegulationChatProps {
  userId: string;
}

const AdminRegulationChat: React.FC<AdminRegulationChatProps> = ({ userId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Format message content with markdown-like syntax
  const formatMessage = (content: string) => {
    return content
      .split('\n')
      .map((line, idx) => {
        // Headers (## text)
        if (line.startsWith('## ')) {
          return <h3 key={idx} className="msg-heading">{line.substring(3)}</h3>;
        }
        // Bold (**text**)
        if (line.includes('**')) {
          const parts = line.split('**');
          return (
            <p key={idx}>
              {parts.map((part, i) => 
                i % 2 === 1 ? <strong key={i}>{part}</strong> : part
              )}
            </p>
          );
        }
        // Bullet points (• text)
        if (line.startsWith('•')) {
          return <li key={idx} className="msg-bullet">{line.substring(1).trim()}</li>;
        }
        // Emoji actions
        if (line.match(/^[✅❌⚠️🔒⬆️⬇️↔️]/)) {
          return <p key={idx} className="msg-action">{line}</p>;
        }
        // Regular text
        if (line.trim()) {
          return <p key={idx}>{line}</p>;
        }
        return <br key={idx} />;
      });
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8002/regulation_chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          user_query: inputValue,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from regulation chat');
      }

      const data = await response.json();
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Chat Button */}
      <button
        className={`chat-widget-button admin-chat-button ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle regulation chat"
      >
        {isOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <line x1="18" y1="6" x2="6" y2="18" strokeWidth="2" strokeLinecap="round"/>
            <line x1="6" y1="6" x2="18" y2="18" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="chat-widget-window admin-chat-window">
          <div className="chat-widget-header admin-chat-header">
            <div className="chat-header-content">
              <div className="chat-header-icon">📜</div>
              <div>
                <h3>Regulation Assistant</h3>
                <p>Ask about regulatory compliance and policies</p>
              </div>
            </div>
          </div>

          <div className="chat-widget-messages">
            {messages.length === 0 && (
              <div className="chat-welcome">
                <p>👋 Hi Admin! I'm your Regulatory Compliance Assistant.</p>
                <p>I can help you with:</p>
                <ul>
                  <li>GDPR and data protection requirements</li>
                  <li>Basel III capital requirements</li>
                  <li>ECOA fair lending practices</li>
                  <li>FCRA credit reporting rules</li>
                  <li>Internal policy interpretations</li>
                </ul>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.role}`}>
                <div className="message-content">
                  {formatMessage(msg.content)}
                </div>
                <div className="message-timestamp">
                  {msg.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="chat-message assistant">
                <div className="message-content typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="chat-widget-input">
            <input
              type="text"
              placeholder="Ask about regulations..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button 
              onClick={handleSendMessage} 
              disabled={isLoading || !inputValue.trim()}
              className="send-button"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default AdminRegulationChat;

