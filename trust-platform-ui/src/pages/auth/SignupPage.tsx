import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { Button } from '../../components/shared/Button';
import './AuthPages.css';

export const SignupPage: React.FC = () => {
  const navigate = useNavigate();
  const { signup, error } = useAuth();
  const [formData, setFormData] = useState({
    userId: '',
    email: '',
    password: '',
    confirmPassword: '',
    name: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [localError, setLocalError] = useState('');
  const [acceptedTerms, setAcceptedTerms] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const validateForm = () => {
    if (!formData.userId || !formData.email || !formData.password || !formData.name) {
      setLocalError('Please fill in all fields');
      return false;
    }

    if (formData.userId.length < 4) {
      setLocalError('User ID must be at least 4 characters');
      return false;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      setLocalError('Please enter a valid email address');
      return false;
    }

    if (formData.password.length < 6) {
      setLocalError('Password must be at least 6 characters');
      return false;
    }

    if (formData.password !== formData.confirmPassword) {
      setLocalError('Passwords do not match');
      return false;
    }

    if (!acceptedTerms) {
      setLocalError('Please accept the terms and conditions');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError('');

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);

    try {
      await signup(formData.userId, formData.email, formData.password, formData.name);
      navigate('/user/dashboard');
    } catch (err: any) {
      setLocalError(err.message || 'Signup failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        {/* Left Side - Branding */}
        <div className="auth-branding">
          <div className="branding-content">
            <div className="brand-logo">
              <span className="brand-icon">🏦</span>
              <span className="brand-name">TrustBank AI</span>
            </div>
            <h1 className="brand-tagline">
              Start Your Journey to
              <br />
              <span className="highlight">Smarter Banking</span>
            </h1>
            <p className="brand-description">
              Join thousands of customers who trust us with their financial future. Experience
              AI-powered banking with complete transparency and control.
            </p>

            <div className="brand-stats">
              <div className="stat-box">
                <div className="stat-number">50K+</div>
                <div className="stat-label">Active Users</div>
              </div>
              <div className="stat-box">
                <div className="stat-number">94.3%</div>
                <div className="stat-label">AI Accuracy</div>
              </div>
              <div className="stat-box">
                <div className="stat-number">4.9★</div>
                <div className="stat-label">User Rating</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side - Signup Form */}
        <div className="auth-form-container">
          <div className="auth-form-content">
            <button className="back-button" onClick={() => navigate('/')}>
              ← Back to Home
            </button>

            <h2 className="form-title">Create Account</h2>
            <p className="form-subtitle">Get started with your free TrustBank AI account</p>

            {(localError || error) && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {localError || error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="name">Full Name</label>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="John Doe"
                    disabled={isLoading}
                    autoComplete="name"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="userId">User ID</label>
                  <input
                    id="userId"
                    name="userId"
                    type="text"
                    value={formData.userId}
                    onChange={handleChange}
                    placeholder="johndoe123"
                    disabled={isLoading}
                    autoComplete="username"
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="email">Email Address</label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="john@example.com"
                  disabled={isLoading}
                  autoComplete="email"
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="password">Password</label>
                  <input
                    id="password"
                    name="password"
                    type="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="Min. 6 characters"
                    disabled={isLoading}
                    autoComplete="new-password"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="confirmPassword">Confirm Password</label>
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type="password"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    placeholder="Re-enter password"
                    disabled={isLoading}
                    autoComplete="new-password"
                  />
                </div>
              </div>

              <label className="checkbox-label full-width">
                <input
                  type="checkbox"
                  checked={acceptedTerms}
                  onChange={(e) => setAcceptedTerms(e.target.checked)}
                  disabled={isLoading}
                />
                <span>
                  I agree to the <Link to="/terms" className="link">Terms of Service</Link> and{' '}
                  <Link to="/privacy" className="link">Privacy Policy</Link>
                </span>
              </label>

              <Button
                type="submit"
                variant="primary"
                size="large"
                disabled={isLoading}
                className="submit-button"
              >
                {isLoading ? 'Creating Account...' : 'Create Account'}
              </Button>
            </form>

            <div className="form-footer">
              <p>
                Already have an account?{' '}
                <Link to="/login" className="link">
                  Sign In
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

