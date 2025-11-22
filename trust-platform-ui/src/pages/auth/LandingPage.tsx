import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../../components/shared/Button';
import './LandingPage.css';

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      {/* Header */}
      <header className="landing-header">
        <div className="header-content">
          <div className="logo-section">
            <span className="logo-icon">🏦</span>
            <span className="logo-text">TrustBank AI</span>
          </div>
          <div className="header-actions">
            <Button variant="secondary" onClick={() => navigate('/login')}>
              Login
            </Button>
            <Button variant="primary" onClick={() => navigate('/signup')}>
              Get Started
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            AI-Powered Banking
            <br />
            <span className="hero-highlight">Built on Trust</span>
          </h1>
          <p className="hero-subtitle">
            Experience next-generation banking with transparent AI, real-time insights,
            and complete control over your financial data.
          </p>
          <div className="hero-actions">
            <Button variant="primary" size="large" onClick={() => navigate('/signup')}>
              Open Your Account
            </Button>
            <Button variant="secondary" size="large" onClick={() => navigate('/login')}>
              Sign In
            </Button>
          </div>

          {/* Quick Stats */}
          <div className="hero-stats">
            <div className="stat-item">
              <div className="stat-value">94.3%</div>
              <div className="stat-label">AI Accuracy</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">&lt;0.03</div>
              <div className="stat-label">Bias Score</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">100%</div>
              <div className="stat-label">Transparent</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">24/7</div>
              <div className="stat-label">AI Support</div>
            </div>
          </div>
        </div>

        {/* Hero Image/Animation */}
        <div className="hero-visual">
          <div className="visual-card card-1">
            <div className="card-icon">💳</div>
            <div className="card-text">Smart Credit Scoring</div>
          </div>
          <div className="visual-card card-2">
            <div className="card-icon">🤖</div>
            <div className="card-text">AI-Powered Insights</div>
          </div>
          <div className="visual-card card-3">
            <div className="card-icon">🔒</div>
            <div className="card-text">Bank-Grade Security</div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <h2 className="section-title">Why Choose TrustBank AI?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3 className="feature-title">Know Your Profile</h3>
            <p className="feature-description">
              Get instant insights into your credit score, debt-to-income ratio, and financial health
              with AI-powered analysis.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">⚖️</div>
            <h3 className="feature-title">Fair & Unbiased</h3>
            <p className="feature-description">
              Our RL-optimized fairness monitor ensures every decision is free from bias and
              discrimination.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h3 className="feature-title">Explainable AI</h3>
            <p className="feature-description">
              Understand exactly why AI made each decision with SHAP-based explanations in plain
              English.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🛡️</div>
            <h3 className="feature-title">Your Data, Your Control</h3>
            <p className="feature-description">
              Complete consent management with blockchain-verified receipts and granular privacy
              controls.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3 className="feature-title">Real-Time Insights</h3>
            <p className="feature-description">
              Track your spending, assets, and financial goals with beautiful visualizations and
              smart recommendations.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">💬</div>
            <h3 className="feature-title">AI Assistant</h3>
            <p className="feature-description">
              Get instant answers to your banking questions with our intelligent 24/7 AI chatbot.
            </p>
          </div>
        </div>
      </section>

      {/* Trust Section */}
      <section className="trust-section">
        <h2 className="section-title">Built on Transparency & Trust</h2>
        <div className="trust-content">
          <div className="trust-item">
            <div className="trust-icon">🔐</div>
            <h3>Bank-Grade Security</h3>
            <p>256-bit encryption, multi-factor authentication, and continuous monitoring</p>
          </div>
          <div className="trust-item">
            <div className="trust-icon">📜</div>
            <h3>Regulatory Compliant</h3>
            <p>Fully compliant with RBI, GDPR, and international banking regulations</p>
          </div>
          <div className="trust-item">
            <div className="trust-icon">✅</div>
            <h3>Human-in-the-Loop</h3>
            <p>Critical decisions reviewed by experienced banking professionals</p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <h2 className="cta-title">Ready to Experience Banking of Tomorrow?</h2>
        <p className="cta-subtitle">Join thousands of customers who trust us with their financial future</p>
        <Button variant="primary" size="large" onClick={() => navigate('/signup')}>
          Create Free Account
        </Button>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>TrustBank AI</h4>
            <p>Next-generation banking powered by transparent AI</p>
          </div>
          <div className="footer-section">
            <h4>Product</h4>
            <ul>
              <li>Personal Banking</li>
              <li>Credit Scoring</li>
              <li>AI Insights</li>
              <li>Admin Dashboard</li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Company</h4>
            <ul>
              <li>About Us</li>
              <li>Careers</li>
              <li>Press</li>
              <li>Contact</li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Legal</h4>
            <ul>
              <li>Privacy Policy</li>
              <li>Terms of Service</li>
              <li>Security</li>
              <li>Compliance</li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2024 TrustBank AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

