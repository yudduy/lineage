import React from 'react';
import { clsx } from 'clsx';

export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  description?: string;
  className?: string;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  checked,
  onChange,
  disabled = false,
  size = 'md',
  label,
  description,
  className,
}) => {
  const sizeClasses = {
    sm: {
      switch: 'w-8 h-4',
      thumb: 'w-3 h-3',
      translate: 'translate-x-4',
    },
    md: {
      switch: 'w-11 h-6',
      thumb: 'w-5 h-5',
      translate: 'translate-x-5',
    },
    lg: {
      switch: 'w-14 h-8',
      thumb: 'w-7 h-7',
      translate: 'translate-x-6',
    },
  };

  const handleToggle = () => {
    if (!disabled) {
      onChange(!checked);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleToggle();
    }
  };

  return (
    <div className={clsx('flex items-start', className)}>
      <div className="flex items-center">
        {/* Switch */}
        <button
          type="button"
          className={clsx(
            'relative inline-flex flex-shrink-0 border-2 border-transparent rounded-full',
            'cursor-pointer transition-colors ease-in-out duration-200',
            'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500',
            sizeClasses[size].switch,
            {
              'bg-primary-600': checked && !disabled,
              'bg-surface border-border': !checked && !disabled,
              'bg-gray-300 cursor-not-allowed': disabled,
            }
          )}
          onClick={handleToggle}
          onKeyDown={handleKeyDown}
          role="switch"
          aria-checked={checked}
          aria-disabled={disabled}
          disabled={disabled}
        >
          <span className="sr-only">
            {label || (checked ? 'Disable' : 'Enable')}
          </span>
          
          {/* Thumb */}
          <span
            className={clsx(
              'pointer-events-none inline-block rounded-full bg-white shadow-lg',
              'transform ring-0 transition ease-in-out duration-200',
              sizeClasses[size].thumb,
              {
                [sizeClasses[size].translate]: checked,
                'translate-x-0': !checked,
              }
            )}
          />
        </button>

        {/* Label and description */}
        {(label || description) && (
          <div className="ml-3">
            {label && (
              <div
                className={clsx('text-sm font-medium text-text', {
                  'text-gray-400': disabled,
                })}
              >
                {label}
              </div>
            )}
            {description && (
              <div
                className={clsx('text-xs text-text-secondary', {
                  'text-gray-400': disabled,
                })}
              >
                {description}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ToggleSwitch;