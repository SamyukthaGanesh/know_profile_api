import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { UserLayout } from './components/layout/UserLayout';
import { AdminLayout } from './components/layout/AdminLayout';

// Auth Pages
import { LandingPage } from './pages/auth/LandingPage';
import { LoginPage } from './pages/auth/LoginPage';
import { SignupPage } from './pages/auth/SignupPage';

// User Pages
import { UserDashboard } from './pages/user/Dashboard';
import { Explanations } from './pages/user/Explanations';
import { ConsentWallet } from './pages/user/ConsentWallet';
import { KnowYourProfile } from './pages/user/KnowYourProfile';

// Admin Pages
import { AdminOverview } from './pages/admin/Overview';
import { ModelHealth } from './pages/admin/ModelHealth';
import { FairnessMonitor } from './pages/admin/FairnessMonitor';
import { ApprovalsQueue } from './pages/admin/ApprovalsQueue';

import './App.css';

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode; adminOnly?: boolean }> = ({ 
  children, 
  adminOnly = false 
}) => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (adminOnly && user.role !== 'admin') {
    return <Navigate to="/user/dashboard" replace />;
  }

  return <>{children}</>;
};

// Public Route Component (redirect to dashboard if already logged in)
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  if (user) {
    return <Navigate to={user.role === 'admin' ? '/admin/overview' : '/user/dashboard'} replace />;
  }

  return <>{children}</>;
};

const App: React.FC = () => {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<PublicRoute><LandingPage /></PublicRoute>} />
          <Route path="/login" element={<PublicRoute><LoginPage /></PublicRoute>} />
          <Route path="/signup" element={<PublicRoute><SignupPage /></PublicRoute>} />

          {/* Protected User Routes */}
          <Route
            path="/user/*"
            element={
              <ProtectedRoute>
                <UserLayout>
                  <Routes>
                    <Route path="dashboard" element={<UserDashboard />} />
                    <Route path="profile" element={<KnowYourProfile />} />
                    <Route path="explanations" element={<Explanations />} />
                    <Route path="consent" element={<ConsentWallet />} />
                    <Route path="fairness" element={<PlaceholderPage title="Fairness Monitor" />} />
                    <Route path="audit" element={<PlaceholderPage title="Audit Trail" />} />
                    <Route path="*" element={<Navigate to="/user/dashboard" replace />} />
                  </Routes>
                </UserLayout>
              </ProtectedRoute>
            }
          />

          {/* Protected Admin Routes */}
          <Route
            path="/admin/*"
            element={
              <ProtectedRoute adminOnly>
                <AdminLayout>
                  <Routes>
                    <Route path="overview" element={<AdminOverview />} />
                    <Route path="models" element={<ModelHealth />} />
                    <Route path="fairness" element={<FairnessMonitor />} />
                    <Route path="approvals" element={<ApprovalsQueue />} />
                    <Route path="regulatory" element={<PlaceholderPage title="Regulatory Dashboard" dark />} />
                    <Route path="audit" element={<PlaceholderPage title="Audit & Ledgers" dark />} />
                    <Route path="alerts" element={<PlaceholderPage title="Alert Center" dark />} />
                    <Route path="human-loop" element={<PlaceholderPage title="Human-in-Loop Cases" dark />} />
                    <Route path="explainability" element={<PlaceholderPage title="Explainability Lab" dark />} />
                    <Route path="*" element={<Navigate to="/admin/overview" replace />} />
                  </Routes>
                </AdminLayout>
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
};

// Placeholder component for pages not yet implemented
const PlaceholderPage: React.FC<{ title: string; dark?: boolean }> = ({ title, dark = false }) => {
  return (
    <div className={`placeholder-page ${dark ? 'dark' : ''}`}>
      <div className="placeholder-content">
        <h1>{title}</h1>
        <p>This page is coming soon. The full implementation includes:</p>
        <ul>
          <li>Real-time data visualization</li>
          <li>Interactive controls and filters</li>
          <li>Detailed analytics and reporting</li>
          <li>Advanced AI features</li>
        </ul>
      </div>
    </div>
  );
};

export default App;

