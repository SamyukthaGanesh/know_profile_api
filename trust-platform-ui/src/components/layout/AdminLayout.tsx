import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import AdminRegulationChat from '../chatbot/AdminRegulationChat';
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
      title: 'DASHBOARD',
      items: [
        { path: '/admin/overview', label: 'Overview', icon: '📊' },
        { path: '/admin/models', label: 'Model Health', icon: '🏥' },
        { path: '/admin/fairness', label: 'Fairness Monitor', icon: '⚖️' },
      ],
    },
    {
      title: 'GOVERNANCE',
      items: [
        { path: '/admin/policies', label: 'Policy Manager', icon: '📜' },
        { path: '/admin/audit-ledger', label: 'Audit Ledger', icon: '🔐' },
        { path: '/admin/blockchain', label: 'Blockchain Explorer', icon: '🔗' },
      { path: '/admin/blockchain-graph', label: 'Blockchain Graph', icon: '📊' },
        { path: '/admin/data', label: 'Data Management', icon: '🗄️' },
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
      
      {/* Admin Regulation Chat Widget */}
      <AdminRegulationChat userId={user?.userId || 'admin'} />
    </div>
  );
};

