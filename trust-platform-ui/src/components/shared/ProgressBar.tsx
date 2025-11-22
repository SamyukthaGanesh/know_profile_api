import React from 'react';
import './ProgressBar.css';

interface ProgressBarProps {
  value: number;
  max?: number;
  variant?: 'success' | 'warning' | 'danger' | 'default';
  showLabel?: boolean;
  label?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ 
  value, 
  max = 100,
  variant = 'default',
  showLabel = false,
  label
}) => {
  const percentage = Math.min((value / max) * 100, 100);
  
  return (
    <div className="progress-container">
      <div className="progress-bar-wrapper">
        <div 
          className={`progress-bar progress-${variant}`}
          style={{ width: `${percentage}%` }}
        >
          {showLabel && (
            <span className="progress-label">
              {label || `${Math.round(percentage)}%`}
            </span>
          )}
        </div>
      </div>
      {!showLabel && label && (
        <span className="progress-text">{label}</span>
      )}
    </div>
  );
};

