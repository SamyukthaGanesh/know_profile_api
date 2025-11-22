import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import './AdminLayout.css';

interface AdminLayoutProps {
  children: React.ReactNode;
}

export const AdminLayout: React.FC<AdminLayoutProps> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const navSections = [
    {
      title: 'AI GOVERNANCE',
      items: [
        { path: '/admin/overview', label: 'Overview', icon: '📊' },
        { path: '/admin/models', label: 'Model Health', icon: '🏥' },
        { path: '/admin/fairness', label: 'Fairness Monitor', icon: '⚖️' },
        { path: '/admin/approvals', label: 'Approvals Queue', icon: '✅' },
      ],
    },
    {
      title: 'COMPLIANCE',
      items: [
        { path: '/admin/regulatory', label: 'Regulatory Dashboard', icon: '📋' },
        { path: '/admin/audit', label: 'Audit & Ledgers', icon: '🔍' },
      ],
    },
    {
      title: 'OPERATIONS',
      items: [
        { path: '/admin/alerts', label: 'Alert Center', icon: '🚨' },
        { path: '/admin/human-loop', label: 'Human-in-Loop', icon: '👤' },
        { path: '/admin/explainability', label: 'Explainability Lab', icon: '🧠' },
      ],
    },
  ];

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="admin-layout">
      <aside className="admin-sidebar">
        <div className="admin-logo">
          <span className="admin-logo-icon">🏦</span>
          <span>TrustBank Admin</span>
        </div>

        {navSections.map((section) => (
          <div key={section.title}>
            <div className="nav-section">{section.title}</div>
            {section.items.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
              >
                <span className="nav-icon">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
        ))}

        <div className="sidebar-footer">
          <Link to="/user/dashboard" className="nav-item">
            <span className="nav-icon">👤</span>
            User View
          </Link>
          <div className="admin-user-info">
            <div className="admin-user-name">{user?.name || 'Admin'}</div>
            <div className="admin-user-email">{user?.email}</div>
          </div>
          <button className="logout-btn" onClick={handleLogout}>
            <span className="nav-icon">🚪</span>
            Logout
          </button>
        </div>
      </aside>

      <main className="admin-main">{children}</main>
    </div>
  );
};

