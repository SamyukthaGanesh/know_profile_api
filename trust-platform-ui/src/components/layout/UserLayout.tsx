import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { AIChatbot } from '../chatbot/AIChatbot';
import './UserLayout.css';

interface UserLayoutProps {
  children: React.ReactNode;
}

export const UserLayout: React.FC<UserLayoutProps> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [showMenu, setShowMenu] = useState(false);

  const navItems = [
    { path: '/user/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/user/profile', label: 'Your Profile', icon: '👤' },
    { path: '/user/explanations', label: 'AI Explanations', icon: '🧠' },
    { path: '/user/consent', label: 'Consent Wallet', icon: '🔐' },
  ];

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="user-layout">
      <header className="user-header">
        <div className="logo">
          <span className="logo-icon">🏦</span>
          <span className="logo-text">TrustBank AI</span>
        </div>
        <nav className="nav">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-btn ${location.pathname === item.path ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="user-menu">
          {user?.role === 'admin' && (
            <Link to="/admin/overview" className="admin-link">
              👔 Admin View
            </Link>
          )}
          <div className="user-profile" onClick={() => setShowMenu(!showMenu)}>
            <span className="user-avatar">👤</span>
            <span className="user-name">{user?.name || 'User'}</span>
            <span className="dropdown-arrow">▾</span>
            {showMenu && (
              <div className="user-dropdown">
                <div className="dropdown-header">
                  <div className="dropdown-user-name">{user?.name}</div>
                  <div className="dropdown-user-email">{user?.email}</div>
                  <div className="dropdown-user-id">ID: {user?.userId}</div>
                </div>
                <div className="dropdown-divider"></div>
                <button className="dropdown-item" onClick={() => navigate('/user/profile')}>
                  <span>👤</span> Your Profile
                </button>
                <button className="dropdown-item" onClick={() => navigate('/user/consent')}>
                  <span>🔐</span> Privacy Settings
                </button>
                <div className="dropdown-divider"></div>
                <button className="dropdown-item logout" onClick={handleLogout}>
                  <span>🚪</span> Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </header>
      <main className="user-content">{children}</main>
      
      {/* AI Chatbot Widget */}
      <AIChatbot />
    </div>
  );
};

