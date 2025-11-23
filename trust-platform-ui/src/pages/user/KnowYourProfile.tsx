import React, { useState, useEffect } from 'react';
import { Card, CardHeader } from '../../components/shared/Card';
import { Button } from '../../components/shared/Button';
import { Badge } from '../../components/shared/Badge';
import { ProgressBar } from '../../components/shared/ProgressBar';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import './KnowYourProfile.css';

interface ProfileData {
  user_id: string;
  credit_score: number;
  dti_ratio: number;
  annual_interest_inr: number;
  loan_probability: number;
  top_spend: Record<string, number>;
  tips: string[];
  ai_trust_score: number;
  shap_reasons: string[];
  annual_income_inr: number;
  savings_balance_inr: number;
  fd_balance_inr: number;
  rd_balance_inr: number;
  mf_value_inr: number;
  demat_value_inr: number;
  total_assets_inr: number;
  consent: {
    model_training: boolean;
    data_sharing: boolean;
    personalized_offers: boolean;
  };
}

interface Transaction {
  date: string;
  category: string;
  amount_inr: number;
  merchant: string;
  method: string;
}

// Force localhost:8000 for backend API
const API_BASE = 'http://localhost:8000';
console.log('🔧 KnowYourProfile using API_BASE:', API_BASE);

export const KnowYourProfile: React.FC = () => {
  const [userId, setUserId] = useState('U1000');
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [transactionTab, setTransactionTab] = useState<'table' | 'visual'>('visual');

  useEffect(() => {
    // Always load profile on mount with default user
    console.log('🔄 KnowYourProfile mounted, loading profile for:', userId);
    loadProfile();
  }, []); // Load on mount

  const loadProfile = async () => {
    console.log('🔄 Loading profile for user:', userId);
    console.log('🔧 API_BASE:', API_BASE);
    setLoading(true);
    setError('');
    try {
      // Fetch profile
      const profileUrl = `${API_BASE}/generate_profile/${userId}`;
      console.log('📡 Fetching profile from:', profileUrl);
      
      const profileRes = await fetch(profileUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        mode: 'cors',
      });
      
      console.log('📊 Profile response status:', profileRes.status, profileRes.statusText);
      
      if (!profileRes.ok) {
        console.error('❌ Profile fetch failed:', profileRes.status, profileRes.statusText);
        throw new Error(`Failed to fetch profile: ${profileRes.status} ${profileRes.statusText}`);
      }
      
      const profileData = await profileRes.json();
      console.log('✅ Profile loaded:', profileData);
      setProfile(profileData);

      // Fetch transactions
      const txUrl = `${API_BASE}/get_transactions/${userId}?limit=10`;
      console.log('📡 Fetching transactions from:', txUrl);
      
      const txRes = await fetch(txUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        mode: 'cors',
      });
      
      console.log('📊 Transactions response status:', txRes.status, txRes.statusText);
      
      if (!txRes.ok) {
        console.error('❌ Transactions fetch failed:', txRes.status, txRes.statusText);
        throw new Error(`Failed to fetch transactions: ${txRes.status}`);
      }
      
      const txData = await txRes.json();
      console.log('✅ Transactions loaded:', txData.transactions?.length || 0, 'transactions');
      setTransactions(txData.transactions || []);
    } catch (err: any) {
      const errorMsg = err.message || 'Failed to load profile';
      console.error('❌ Profile load error:', errorMsg);
      console.error('❌ Full error:', err);
      setError(`${errorMsg}\n\nBackend URL: ${API_BASE}\nMake sure backend is running on port 8000`);
    } finally {
      setLoading(false);
      console.log('✅ Profile loading complete');
    }
  };

  const generateRandomUser = () => {
    const randomNum = Math.floor(Math.random() * 200) + 1000;
    setUserId(`U${randomNum}`);
    setTimeout(() => loadProfile(), 100);
  };

  const getCreditScoreStatus = (score: number) => {
    if (score >= 750) return { label: 'Excellent', color: 'success' };
    if (score >= 700) return { label: 'Good', color: 'success' };
    if (score >= 650) return { label: 'Fair', color: 'warning' };
    return { label: 'Needs Improvement', color: 'danger' };
  };

  const getDTIStatus = (ratio: number) => {
    if (ratio <= 0.36) return { label: 'Excellent', color: '#00c853' };
    if (ratio <= 0.43) return { label: 'Good', color: '#64dd17' };
    if (ratio <= 0.50) return { label: 'Fair', color: '#ffa726' };
    return { label: 'High', color: '#ff5252' };
  };

  const formatCurrency = (amount: number | undefined) => {
    if (amount === undefined || amount === null || isNaN(amount)) {
      return '₹0';
    }
    return `₹${Math.round(amount).toLocaleString('en-IN')}`;
  };

  if (loading && !profile) {
    console.log('📊 Showing loading screen...');
    return (
      <div className="profile-loading">
        <div className="loader"></div>
        <p>Loading your financial profile...</p>
      </div>
    );
  }

  if (error && !profile) {
    console.log('❌ Showing error screen:', error);
    return (
      <div className="profile-error" style={{ padding: '40px', textAlign: 'center' }}>
        <h2 style={{ color: '#ef4444' }}>⚠️ Error Loading Profile</h2>
        <p style={{ marginTop: '16px', fontSize: '16px' }}>{error}</p>
        <Button variant="primary" onClick={loadProfile} style={{ marginTop: '24px' }}>
          Retry
        </Button>
      </div>
    );
  }

  if (!profile) {
    console.log('⚠️ No profile data, showing empty state');
    return (
      <div className="profile-empty" style={{ padding: '40px', textAlign: 'center' }}>
        <h2>No Profile Data</h2>
        <p style={{ marginTop: '16px' }}>Click "Load Profile" to fetch your data.</p>
        <Button variant="primary" onClick={loadProfile} style={{ marginTop: '24px' }}>
          Load Profile
        </Button>
      </div>
    );
  }

  console.log('✅ Rendering profile for:', profile.user_id);
  return (
    <div className="know-your-profile">
      {/* Bank-style Header */}
      <div className="bank-header">
        <div className="bank-logo-section">
          <div className="bank-logo">🏦</div>
          <div>
            <h1 className="bank-title">TrustBank</h1>
            <p className="bank-subtitle">Know Your Financial Profile</p>
          </div>
        </div>
        <div className="profile-actions">
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter User ID"
            className="user-id-input"
          />
          <Button variant="primary" onClick={loadProfile}>
            Load Profile
          </Button>
          <Button variant="secondary" onClick={generateRandomUser}>
            Random User
          </Button>
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError('')}>✕</button>
        </div>
      )}

      {profile && (
        <>
          {/* KYC Summary Bar */}
          <div className="kyc-summary-bar">
            <div className="kyc-item">
              <span className="kyc-label">Customer ID</span>
              <span className="kyc-value">{profile.user_id}</span>
            </div>
            <div className="kyc-item">
              <span className="kyc-label">AI Trust Score</span>
              <span className="kyc-value trust-score">
                {Math.round(profile.ai_trust_score * 100)}%
              </span>
            </div>
            <div className="kyc-item">
              <span className="kyc-label">Loan Approval Probability</span>
              <span className="kyc-value">
                {Math.round(profile.loan_probability * 100)}%
              </span>
            </div>
            <div className="kyc-item">
              <span className="kyc-label">Total Assets</span>
              <span className="kyc-value">{formatCurrency(profile.total_assets_inr || 0)}</span>
            </div>
          </div>

          {/* Main Dashboard Grid */}
          <div className="profile-grid">
            {/* Credit Score Card */}
            <Card className="bank-card">
              <CardHeader title="Credit Health" />
              <div className="credit-score-container">
                <div className="score-circle">
                  <svg viewBox="0 0 200 200" className="score-svg">
                    <circle
                      cx="100"
                      cy="100"
                      r="80"
                      fill="none"
                      stroke="#e0e0e0"
                      strokeWidth="20"
                    />
                    <circle
                      cx="100"
                      cy="100"
                      r="80"
                      fill="none"
                      stroke={getCreditScoreStatus(profile.credit_score).color === 'success' ? '#4caf50' : '#ff9800'}
                      strokeWidth="20"
                      strokeDasharray={`${(profile.credit_score / 850) * 502.65} 502.65`}
                      transform="rotate(-90 100 100)"
                      className="score-progress"
                    />
                  </svg>
                  <div className="score-text">
                    <div className="score-number">{profile.credit_score}</div>
                    <div className="score-max">out of 850</div>
                  </div>
                </div>
                <Badge variant={getCreditScoreStatus(profile.credit_score).color as any}>
                  {getCreditScoreStatus(profile.credit_score).label}
                </Badge>
                <div className="score-details">
                  <div className="detail-row">
                    <span>Payment History</span>
                    <span className="detail-bar">
                      <div className="bar-fill" style={{ width: '85%', background: '#4caf50' }}></div>
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>Credit Utilization</span>
                    <span className="detail-bar">
                      <div className="bar-fill" style={{ width: '65%', background: '#ffa726' }}></div>
                    </span>
                  </div>
                  <div className="detail-row">
                    <span>Credit History Length</span>
                    <span className="detail-bar">
                      <div className="bar-fill" style={{ width: '75%', background: '#4caf50' }}></div>
                    </span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Debt-to-Income Card */}
            <Card className="bank-card">
              <CardHeader title="Debt-to-Income Ratio" />
              <div className="dti-container">
                <div className="dti-gauge">
                  <div className="gauge-arc">
                    <div 
                      className="gauge-needle" 
                      style={{ transform: `rotate(${(profile.dti_ratio * 180)}deg)` }}
                    ></div>
                  </div>
                  <div className="dti-value">
                    {Math.round(profile.dti_ratio * 100)}%
                  </div>
                  <Badge variant={getDTIStatus(profile.dti_ratio).color === '#00c853' ? 'success' : 'warning'}>
                    {getDTIStatus(profile.dti_ratio).label}
                  </Badge>
                </div>
                <div className="dti-info">
                  <p className="info-text">
                    Your debt payments are {Math.round(profile.dti_ratio * 100)}% of your monthly income.
                  </p>
                  <div className="dti-benchmark">
                    <div className="benchmark-item">
                      <span className="benchmark-dot excellent"></span>
                      <span>Excellent: ≤36%</span>
                    </div>
                    <div className="benchmark-item">
                      <span className="benchmark-dot good"></span>
                      <span>Good: 37-43%</span>
                    </div>
                    <div className="benchmark-item">
                      <span className="benchmark-dot fair"></span>
                      <span>Fair: 44-50%</span>
                    </div>
                    <div className="benchmark-item">
                      <span className="benchmark-dot high"></span>
                      <span>High: &gt;50%</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Assets Breakdown */}
            <Card className="bank-card full-width">
              <CardHeader 
                title="Investment & Assets Portfolio"
                badge={<span className="total-badge">{formatCurrency(profile.total_assets_inr || 0)}</span>}
              />
              <div className="assets-grid">
                <div className="asset-item">
                  <div className="asset-icon" style={{ background: '#e3f2fd' }}>💰</div>
                  <div className="asset-details">
                    <span className="asset-label">Savings Account</span>
                    <span className="asset-value">{formatCurrency(profile.savings_balance_inr || 0)}</span>
                    <ProgressBar 
                      value={profile.savings_balance_inr || 0} 
                      max={profile.total_assets_inr || 1} 
                      variant="default"
                    />
                  </div>
                </div>

                <div className="asset-item">
                  <div className="asset-icon" style={{ background: '#f3e5f5' }}>📈</div>
                  <div className="asset-details">
                    <span className="asset-label">Fixed Deposits</span>
                    <span className="asset-value">{formatCurrency(profile.fd_balance_inr || 0)}</span>
                    <ProgressBar 
                      value={profile.fd_balance_inr || 0} 
                      max={profile.total_assets_inr || 1} 
                      variant="success"
                    />
                  </div>
                </div>

                <div className="asset-item">
                  <div className="asset-icon" style={{ background: '#fff3e0' }}>🔄</div>
                  <div className="asset-details">
                    <span className="asset-label">Recurring Deposits</span>
                    <span className="asset-value">{formatCurrency(profile.rd_balance_inr || 0)}</span>
                    <ProgressBar 
                      value={profile.rd_balance_inr || 0} 
                      max={profile.total_assets_inr || 1} 
                      variant="warning"
                    />
                  </div>
                </div>

                <div className="asset-item">
                  <div className="asset-icon" style={{ background: '#e8f5e9' }}>📊</div>
                  <div className="asset-details">
                    <span className="asset-label">Mutual Funds</span>
                    <span className="asset-value">{formatCurrency(profile.mf_value_inr || 0)}</span>
                    <ProgressBar 
                      value={profile.mf_value_inr || 0} 
                      max={profile.total_assets_inr || 1} 
                      variant="success"
                    />
                  </div>
                </div>

                <div className="asset-item">
                  <div className="asset-icon" style={{ background: '#fce4ec' }}>💹</div>
                  <div className="asset-details">
                    <span className="asset-label">Demat Holdings</span>
                    <span className="asset-value">{formatCurrency(profile.demat_value_inr || 0)}</span>
                    <ProgressBar 
                      value={profile.demat_value_inr || 0} 
                      max={profile.total_assets_inr || 1} 
                      variant="danger"
                    />
                  </div>
                </div>

                <div className="asset-item highlight">
                  <div className="asset-icon" style={{ background: '#e1f5fe' }}>💎</div>
                  <div className="asset-details">
                    <span className="asset-label">Expected Annual Returns</span>
                    <span className="asset-value earning">{formatCurrency(profile.annual_interest_inr || 0)}</span>
                    <span className="monthly-returns">≈ {formatCurrency(Math.round((profile.annual_interest_inr || 0) / 12))}/month</span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Spending Analysis */}
            <Card className="bank-card">
              <CardHeader title="Top Spending Categories" />
              <div className="spending-chart">
                {profile.top_spend && Object.keys(profile.top_spend).length > 0 ? (
                  Object.entries(profile.top_spend)
                    .filter(([, amount]) => amount !== undefined && amount !== null)
                    .sort(([, a], [, b]) => (b || 0) - (a || 0))
                    .map(([category, amount]) => {
                      const maxSpend = Math.max(...Object.values(profile.top_spend).filter(v => v !== undefined && v !== null));
                      return (
                        <div key={category} className="spending-item">
                          <div className="spending-header">
                            <span className="category-name">{category}</span>
                            <span className="category-amount">{formatCurrency(amount)}</span>
                          </div>
                          <div className="spending-bar">
                            <div
                              className="spending-fill"
                              style={{
                                width: `${maxSpend > 0 ? ((amount || 0) / maxSpend) * 100 : 0}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                      );
                    })
                ) : (
                  <p className="no-data">No spending data available</p>
                )}
              </div>
            </Card>

            {/* AI Insights */}
            <Card className="bank-card">
              <CardHeader title="💡 AI-Powered Insights" />
              <div className="insights-container">
                <div className="loan-probability-meter">
                  <div className="meter-label">Loan Approval Probability</div>
                  <div className="probability-circle">
                    <svg viewBox="0 0 120 120">
                      <circle
                        cx="60"
                        cy="60"
                        r="50"
                        fill="none"
                        stroke="#e0e0e0"
                        strokeWidth="10"
                      />
                      <circle
                        cx="60"
                        cy="60"
                        r="50"
                        fill="none"
                        stroke="#4caf50"
                        strokeWidth="10"
                        strokeDasharray={`${profile.loan_probability * 314.16} 314.16`}
                        transform="rotate(-90 60 60)"
                      />
                    </svg>
                    <div className="probability-text">
                      {Math.round(profile.loan_probability * 100)}%
                    </div>
                  </div>
                </div>

                <div className="shap-reasons">
                  <h4>Key Factors Affecting Your Profile:</h4>
                  {profile.shap_reasons && profile.shap_reasons.length > 0 ? (
                    <ul>
                      {profile.shap_reasons.map((reason, idx) => (
                        <li key={idx}>{reason}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="no-reasons">Analysis based on credit score, income, and financial behavior</p>
                  )}
                </div>
              </div>
            </Card>

            {/* Personalized Tips */}
            <Card className="bank-card full-width tips-card">
              <CardHeader title="🎯 Personalized Financial Tips" />
              <div className="tips-grid">
                {profile.tips && profile.tips.length > 0 ? (
                  profile.tips.map((tip, idx) => (
                    <div key={idx} className="tip-item">
                      <div className="tip-number">{idx + 1}</div>
                      <div className="tip-content">{tip}</div>
                    </div>
                  ))
                ) : (
                  <div className="no-tips">
                    <p>✨ Great job! You're managing your finances well. Keep up the good work!</p>
                  </div>
                )}
              </div>
            </Card>

            {/* Recent Transactions */}
            <Card className="bank-card full-width">
              <div className="transaction-header">
                <h3>💳 Recent Transactions</h3>
                <div className="transaction-tabs">
                  <button
                    className={`tab-btn ${transactionTab === 'visual' ? 'active' : ''}`}
                    onClick={() => setTransactionTab('visual')}
                  >
                    📊 Visual Summary
                  </button>
                  <button
                    className={`tab-btn ${transactionTab === 'table' ? 'active' : ''}`}
                    onClick={() => setTransactionTab('table')}
                  >
                    📋 Table View
                  </button>
                </div>
              </div>

              {transactionTab === 'table' ? (
                <div className="transactions-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Category</th>
                        <th>Merchant</th>
                        <th>Payment Mode</th>
                        <th className="amount-col">Amount</th>
                      </tr>
                    </thead>
                    <tbody>
                      {transactions.slice(0, 15).map((tx, idx) => (
                        <tr key={idx}>
                          <td>{new Date(tx.date).toLocaleDateString()}</td>
                          <td>
                            <span className="category-badge">{tx.category}</span>
                          </td>
                          <td>{tx.merchant}</td>
                          <td>
                            <span className="payment-badge">{tx.method}</span>
                          </td>
                          <td className="amount-col debit">-{formatCurrency(tx.amount_inr)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="transaction-visualizations">
                  {transactions && transactions.length > 0 ? (
                    <>
                      <div className="chart-row">
                        {/* Category Spending Bar Chart */}
                        <div className="chart-container">
                          <h4>Spending by Category</h4>
                          <ResponsiveContainer width="100%" height={300}>
                            <BarChart
                              data={Object.entries(
                                transactions.reduce((acc, tx) => {
                                  const category = tx?.category || 'Other';
                                  const amount = tx?.amount_inr || 0;
                                  acc[category] = (acc[category] || 0) + amount;
                                  return acc;
                                }, {} as Record<string, number>)
                              )
                                .map(([category, amount]) => ({
                                  category,
                                  amount
                                }))
                                .sort((a, b) => b.amount - a.amount)}
                              layout="vertical"
                              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                              <XAxis type="number" />
                              <YAxis dataKey="category" type="category" />
                              <Tooltip
                                formatter={(value: any) => `₹${Number(value).toLocaleString('en-IN')}`}
                                contentStyle={{ borderRadius: '8px', border: '1px solid #ddd' }}
                              />
                              <Bar dataKey="amount" fill="#6366f1" radius={[0, 8, 8, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>

                        {/* Payment Method Pie Chart */}
                        <div className="chart-container">
                          <h4>Payment Methods</h4>
                          <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                              <Pie
                                data={Object.entries(
                                  transactions.reduce((acc, tx) => {
                                    const method = tx?.method || 'Unknown';
                                    acc[method] = (acc[method] || 0) + 1;
                                    return acc;
                                  }, {} as Record<string, number>)
                                ).map(([method, count]) => ({
                                  name: method,
                                  value: count
                                }))}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={100}
                                paddingAngle={5}
                                dataKey="value"
                                label={(entry) => `${entry.name}: ${entry.value}`}
                              >
                                {['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'].map((color, index) => (
                                  <Cell key={`cell-${index}`} fill={color} />
                                ))}
                              </Pie>
                              <Tooltip
                                formatter={(value: any) => `${value} transactions`}
                                contentStyle={{ borderRadius: '8px', border: '1px solid #ddd' }}
                              />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      {/* Spending Timeline */}
                      <div className="chart-container full-width-chart">
                        <h4>📈 Spending Over Time</h4>
                        <ResponsiveContainer width="100%" height={250}>
                          <LineChart
                            data={transactions
                              .filter(tx => tx && tx.date && tx.amount_inr !== undefined)
                              .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
                              .reduce((acc, tx) => {
                                const date = new Date(tx.date).toISOString().split('T')[0];
                                const existing = acc.find((item) => item.date === date);
                                const amount = tx.amount_inr || 0;
                                if (existing) {
                                  existing.amount += amount;
                                } else {
                                  acc.push({ date, amount });
                                }
                                return acc;
                              }, [] as { date: string; amount: number }[])}
                            margin={{ top: 5, right: 30, left: 50, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                            <XAxis
                              dataKey="date"
                              tickFormatter={(date) => new Date(date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}
                            />
                            <YAxis />
                            <Tooltip
                              formatter={(value: any) => `₹${Number(value).toLocaleString('en-IN')}`}
                              labelFormatter={(label) => new Date(label).toLocaleDateString('en-IN')}
                              contentStyle={{ borderRadius: '8px', border: '1px solid #ddd' }}
                            />
                            <Line
                              type="monotone"
                              dataKey="amount"
                              stroke="#6366f1"
                              strokeWidth={3}
                              dot={{ fill: '#8b5cf6', r: 5 }}
                              activeDot={{ r: 7 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Transaction Summary Stats */}
                      <div className="transaction-summary">
                        <div className="summary-stat">
                          <div className="summary-icon">🔢</div>
                          <div className="summary-content">
                            <div className="summary-label">Total Transactions</div>
                            <div className="summary-value">{transactions.length}</div>
                          </div>
                        </div>
                        <div className="summary-stat">
                          <div className="summary-icon">💸</div>
                          <div className="summary-content">
                            <div className="summary-label">Total Spent</div>
                            <div className="summary-value">
                              {formatCurrency(
                                transactions.reduce((sum, tx) => sum + (tx?.amount_inr || 0), 0)
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="summary-stat">
                          <div className="summary-icon">📊</div>
                          <div className="summary-content">
                            <div className="summary-label">Avg Transaction</div>
                            <div className="summary-value">
                              {formatCurrency(
                                transactions.length > 0
                                  ? transactions.reduce((sum, tx) => sum + (tx?.amount_inr || 0), 0) / transactions.length
                                  : 0
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="summary-stat">
                          <div className="summary-icon">🏆</div>
                          <div className="summary-content">
                            <div className="summary-label">Largest Transaction</div>
                            <div className="summary-value">
                              {formatCurrency(
                                Math.max(...transactions.map((tx) => tx?.amount_inr || 0), 0)
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="no-transactions">
                      <div className="no-data-icon">📊</div>
                      <h3>No Transaction Data Available</h3>
                      <p>Transaction visualizations will appear here once data is loaded.</p>
                      <p className="help-text">Try clicking "Load Profile" or "Random User" to fetch data.</p>
                    </div>
                  )}
                </div>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
};

