import React, { useState, useRef, useCallback } from 'react';
import { clsx } from 'clsx';

export interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  showValue?: boolean;
  formatValue?: (value: number) => string;
}

const Slider: React.FC<SliderProps> = ({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className,
  showValue = false,
  formatValue = (val) => val.toString(),
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);

  const percentage = ((value - min) / (max - min)) * 100;

  const getValueFromPosition = useCallback(
    (clientX: number): number => {
      if (!sliderRef.current) return value;

      const rect = sliderRef.current.getBoundingClientRect();
      const percentage = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const rawValue = min + percentage * (max - min);
      
      // Round to nearest step
      return Math.round(rawValue / step) * step;
    },
    [min, max, step, value]
  );

  const handleMouseDown = useCallback(
    (event: React.MouseEvent) => {
      if (disabled) return;

      event.preventDefault();
      setIsDragging(true);

      const newValue = getValueFromPosition(event.clientX);
      onChange(Math.max(min, Math.min(max, newValue)));
    },
    [disabled, getValueFromPosition, onChange, min, max]
  );

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!isDragging || disabled) return;

      event.preventDefault();
      const newValue = getValueFromPosition(event.clientX);
      onChange(Math.max(min, Math.min(max, newValue)));
    },
    [isDragging, disabled, getValueFromPosition, onChange, min, max]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Handle mouse events
  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (disabled) return;

    let newValue = value;

    switch (event.key) {
      case 'ArrowLeft':
      case 'ArrowDown':
        event.preventDefault();
        newValue = Math.max(min, value - step);
        break;
      case 'ArrowRight':
      case 'ArrowUp':
        event.preventDefault();
        newValue = Math.min(max, value + step);
        break;
      case 'Home':
        event.preventDefault();
        newValue = min;
        break;
      case 'End':
        event.preventDefault();
        newValue = max;
        break;
      case 'PageDown':
        event.preventDefault();
        newValue = Math.max(min, value - (max - min) / 10);
        break;
      case 'PageUp':
        event.preventDefault();
        newValue = Math.min(max, value + (max - min) / 10);
        break;
    }

    if (newValue !== value) {
      onChange(newValue);
    }
  };

  return (
    <div className={clsx('relative', className)}>
      <div
        ref={sliderRef}
        className={clsx(
          'relative h-6 flex items-center cursor-pointer group',
          { 'cursor-not-allowed': disabled }
        )}
        onMouseDown={handleMouseDown}
        onKeyDown={handleKeyDown}
        tabIndex={disabled ? -1 : 0}
        role="slider"
        aria-valuenow={value}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-disabled={disabled}
      >
        {/* Track */}
        <div
          className={clsx(
            'w-full h-2 bg-surface rounded-full',
            'border border-border',
            { 'opacity-50': disabled }
          )}
        >
          {/* Progress */}
          <div
            className={clsx(
              'h-full bg-primary-600 rounded-full transition-all',
              { 'bg-primary-400': disabled }
            )}
            style={{ width: `${percentage}%` }}
          />
        </div>

        {/* Thumb */}
        <div
          className={clsx(
            'absolute w-5 h-5 bg-primary-600 border-2 border-white',
            'rounded-full shadow-md transition-all',
            'group-hover:scale-110 focus:scale-110',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            {
              'bg-primary-400 border-gray-300 cursor-not-allowed': disabled,
              'scale-110': isDragging,
            }
          )}
          style={{ left: `${percentage}%`, transform: 'translateX(-50%)' }}
        />
      </div>

      {/* Value display */}
      {showValue && (
        <div className="mt-1 text-center">
          <span className="text-sm text-text-secondary">
            {formatValue(value)}
          </span>
        </div>
      )}
    </div>
  );
};

export default Slider;