import React from 'react';
import './ToggleSwitch.css';

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ 
  checked, 
  onChange, 
  label,
  disabled = false 
}) => {
  return (
    <div className="toggle-switch-container">
      {label && <span className="toggle-label">{label}</span>}
      <div 
        className={`toggle-switch ${checked ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={() => !disabled && onChange(!checked)}
        role="switch"
        aria-checked={checked}
        aria-label={label || 'Toggle switch'}
        tabIndex={disabled ? -1 : 0}
        onKeyPress={(e) => {
          if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
            onChange(!checked);
          }
        }}
      >
        <div className="toggle-slider"></div>
      </div>
    </div>
  );
};

