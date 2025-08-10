import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wifi, WifiOff, AlertCircle, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import useWebSocket from '@hooks/useWebSocket';

interface RealtimeStatusIndicatorProps {
  className?: string;
  showLabel?: boolean;
}

const RealtimeStatusIndicator: React.FC<RealtimeStatusIndicatorProps> = ({
  className,
  showLabel = false,
}) => {
  const { isConnected, isConnecting, connectionState } = useWebSocket({
    enabled: true,
    reconnect: true,
  });

  const getStatusConfig = () => {
    switch (connectionState) {
      case 'connected':
        return {
          icon: Wifi,
          color: 'text-green-500',
          bgColor: 'bg-green-100 dark:bg-green-900/20',
          label: 'Connected',
          description: 'Real-time updates active',
        };
      case 'connecting':
        return {
          icon: Loader2,
          color: 'text-yellow-500',
          bgColor: 'bg-yellow-100 dark:bg-yellow-900/20',
          label: 'Connecting',
          description: 'Establishing connection...',
          animate: true,
        };
      case 'disconnected':
        return {
          icon: WifiOff,
          color: 'text-gray-400',
          bgColor: 'bg-gray-100 dark:bg-gray-800',
          label: 'Disconnected',
          description: 'Real-time updates unavailable',
        };
      case 'error':
        return {
          icon: AlertCircle,
          color: 'text-red-500',
          bgColor: 'bg-red-100 dark:bg-red-900/20',
          label: 'Error',
          description: 'Connection failed',
        };
      default:
        return {
          icon: WifiOff,
          color: 'text-gray-400',
          bgColor: 'bg-gray-100 dark:bg-gray-800',
          label: 'Unknown',
          description: 'Status unknown',
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  if (showLabel) {
    return (
      <div className={clsx('flex items-center space-x-2', className)}>
        <div className={clsx('p-2 rounded-full', config.bgColor)}>
          <Icon
            className={clsx('h-4 w-4', config.color, {
              'animate-spin': config.animate,
            })}
          />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {config.label}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {config.description}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('relative', className)} title={`${config.label}: ${config.description}`}>
      {/* Status Indicator */}
      <motion.div
        initial={false}
        animate={{
          scale: isConnected ? [1, 1.1, 1] : 1,
        }}
        transition={{
          duration: 2,
          repeat: isConnected ? Infinity : 0,
          repeatType: 'loop',
        }}
        className={clsx('p-2 rounded-full', config.bgColor)}
      >
        <Icon
          className={clsx('h-4 w-4', config.color, {
            'animate-spin': config.animate,
          })}
        />
      </motion.div>

      {/* Connection Pulse Animation */}
      <AnimatePresence>
        {isConnected && (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 2, opacity: 0 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{
              duration: 2,
              repeat: Infinity,
              repeatType: 'loop',
              ease: 'easeOut',
            }}
            className="absolute inset-0 rounded-full border border-green-500 pointer-events-none"
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default RealtimeStatusIndicator;