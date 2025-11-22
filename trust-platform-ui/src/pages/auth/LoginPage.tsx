import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { Button } from '../../components/shared/Button';
import './AuthPages.css';

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login, error } = useAuth();
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [localError, setLocalError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError('');

    if (!userId || !password) {
      setLocalError('Please fill in all fields');
      return;
    }

    setIsLoading(true);

    try {
      await login(userId, password);
      navigate('/user/dashboard');
    } catch (err: any) {
      setLocalError(err.message || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoLogin = async (role: 'admin' | 'user') => {
    setIsLoading(true);
    setLocalError('');

    try {
      if (role === 'admin') {
        await login('admin', 'password');
        navigate('/admin/overview');
      } else {
        await login('user1', 'password');
        navigate('/user/dashboard');
      }
    } catch (err: any) {
      setLocalError(err.message || 'Login failed');
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
              Welcome Back to
              <br />
              <span className="highlight">Intelligent Banking</span>
            </h1>
            <p className="brand-description">
              Sign in to access your AI-powered financial insights, personalized recommendations,
              and complete control over your banking experience.
            </p>

            <div className="brand-features">
              <div className="brand-feature">
                <span className="feature-icon">✓</span>
                <span>Real-time credit insights</span>
              </div>
              <div className="brand-feature">
                <span className="feature-icon">✓</span>
                <span>AI-powered recommendations</span>
              </div>
              <div className="brand-feature">
                <span className="feature-icon">✓</span>
                <span>Complete data transparency</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side - Login Form */}
        <div className="auth-form-container">
          <div className="auth-form-content">
            <button className="back-button" onClick={() => navigate('/')}>
              ← Back to Home
            </button>

            <h2 className="form-title">Sign In</h2>
            <p className="form-subtitle">Enter your credentials to access your account</p>

            {(localError || error) && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {localError || error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-group">
                <label htmlFor="userId">User ID or Email</label>
                <input
                  id="userId"
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Enter your user ID or email"
                  disabled={isLoading}
                  autoComplete="username"
                />
              </div>

              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  disabled={isLoading}
                  autoComplete="current-password"
                />
              </div>

              <div className="form-options">
                <label className="checkbox-label">
                  <input type="checkbox" />
                  <span>Remember me</span>
                </label>
                <Link to="/forgot-password" className="forgot-link">
                  Forgot password?
                </Link>
              </div>

              <Button
                type="submit"
                variant="primary"
                size="large"
                disabled={isLoading}
                className="submit-button"
              >
                {isLoading ? 'Signing In...' : 'Sign In'}
              </Button>
            </form>

            {/* Demo Accounts */}
            <div className="demo-section">
              <div className="divider">
                <span>Or try demo accounts</span>
              </div>

              <div className="demo-buttons">
                <button
                  onClick={() => handleDemoLogin('user')}
                  disabled={isLoading}
                  className="demo-button user-demo"
                >
                  <span className="demo-icon">👤</span>
                  <div>
                    <div className="demo-title">User Demo</div>
                    <div className="demo-subtitle">userId: user1</div>
                  </div>
                </button>

                <button
                  onClick={() => handleDemoLogin('admin')}
                  disabled={isLoading}
                  className="demo-button admin-demo"
                >
                  <span className="demo-icon">👔</span>
                  <div>
                    <div className="demo-title">Admin Demo</div>
                    <div className="demo-subtitle">userId: admin</div>
                  </div>
                </button>
              </div>
            </div>

            <div className="form-footer">
              <p>
                Don't have an account?{' '}
                <Link to="/signup" className="link">
                  Create Account
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

