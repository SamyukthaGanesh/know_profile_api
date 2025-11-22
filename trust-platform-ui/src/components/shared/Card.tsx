import React from 'react';
import './Card.css';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export const Card: React.FC<CardProps> = ({ children, className = '', style }) => {
  return (
    <div className={`card ${className}`} style={style}>
      {children}
    </div>
  );
};

interface CardHeaderProps {
  title: string;
  badge?: React.ReactNode;
  action?: React.ReactNode;
}

export const CardHeader: React.FC<CardHeaderProps> = ({ title, badge, action }) => {
  return (
    <div className="card-header">
      <div className="card-header-left">
        <h2 className="card-title">{title}</h2>
        {badge && <span className="card-badge">{badge}</span>}
      </div>
      {action && <div className="card-header-action">{action}</div>}
    </div>
  );
};

