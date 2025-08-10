import React from 'react';
import { usePaperStore, useUiStore } from '@store/index';
import Button from '@components/ui/Button';
import Select from '@components/ui/Select';
import Slider from '@components/ui/Slider';
import ToggleSwitch from '@components/ui/ToggleSwitch';

const NetworkControls: React.FC = () => {
  const {
    graphConfig,
    updateGraphConfig,
    seedPapers,
  } = usePaperStore();
  
  const {
    isNetworkView3D,
    toggle3DView,
    openModal,
  } = useUiStore();

  const modeOptions = [
    { value: 'references', label: 'References' },
    { value: 'citations', label: 'Citations' },
  ];

  const sizeMetricOptions = [
    { value: 'seedsCitedBy', label: 'Cited by Seeds' },
    { value: 'seedsCited', label: 'Cites Seeds' },
    { value: 'localCitedBy', label: 'Local Citations' },
    { value: 'localReferences', label: 'Local References' },
  ];

  const colorSchemeOptions = [
    { value: 'default', label: 'Default' },
    { value: 'year', label: 'Publication Year' },
    { value: 'citations', label: 'Citation Count' },
    { value: 'community', label: 'Communities' },
    { value: 'impact', label: 'Impact Score' },
  ];

  return (
    <div className="absolute top-4 right-4 z-10 bg-background/90 backdrop-blur-sm rounded-lg border border-border p-4 shadow-lg min-w-[280px]">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-text">Network Controls</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={toggle3DView}
            className="text-xs"
          >
            {isNetworkView3D ? '2D' : '3D'}
          </Button>
        </div>

        {/* Mode Selection */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Mode
          </label>
          <Select
            value={graphConfig.mode}
            onChange={(value) => updateGraphConfig({ mode: value as 'references' | 'citations' })}
            options={modeOptions}
            size="sm"
          />
          <p className="text-xs text-text-secondary mt-1">
            {graphConfig.mode === 'references' 
              ? 'Show papers referenced by seed papers'
              : 'Show papers citing seed papers'
            }
          </p>
        </div>

        {/* Threshold Slider */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Connection Threshold
          </label>
          <Slider
            value={graphConfig.threshold}
            onChange={(value) => updateGraphConfig({ threshold: value })}
            min={0}
            max={10}
            step={1}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-text-secondary mt-1">
            <span>0</span>
            <span>Current: {graphConfig.threshold}</span>
            <span>10+</span>
          </div>
        </div>

        {/* Size Metric */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Node Size
          </label>
          <Select
            value={graphConfig.sizeMetric}
            onChange={(value) => updateGraphConfig({ sizeMetric: value as any })}
            options={sizeMetricOptions}
            size="sm"
          />
        </div>

        {/* Color Scheme */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Color Scheme
          </label>
          <Select
            value={graphConfig.colorScheme}
            onChange={(value) => updateGraphConfig({ colorScheme: value as any })}
            options={colorSchemeOptions}
            size="sm"
          />
        </div>

        {/* Show Labels Toggle */}
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-text-secondary">
            Show Labels
          </label>
          <ToggleSwitch
            checked={graphConfig.showLabels}
            onChange={(checked) => updateGraphConfig({ showLabels: checked })}
          />
        </div>

        {/* Advanced Features */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-text-secondary">
              Community Detection
            </label>
            <ToggleSwitch
              checked={graphConfig.communityDetection}
              onChange={(checked) => updateGraphConfig({ communityDetection: checked })}
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-text-secondary">
              Citation Flow
            </label>
            <ToggleSwitch
              checked={graphConfig.showCitationFlow}
              onChange={(checked) => updateGraphConfig({ showCitationFlow: checked })}
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-text-secondary">
              Path Highlighting
            </label>
            <ToggleSwitch
              checked={graphConfig.pathHighlight}
              onChange={(checked) => updateGraphConfig({ pathHighlight: checked })}
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-text-secondary">
              Animation
            </label>
            <ToggleSwitch
              checked={graphConfig.animation}
              onChange={(checked) => updateGraphConfig({ animation: checked })}
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-text-secondary">
              Node Clustering
            </label>
            <ToggleSwitch
              checked={graphConfig.clustering}
              onChange={(checked) => updateGraphConfig({ clustering: checked })}
            />
          </div>
        </div>

        {/* Statistics */}
        {seedPapers.length > 0 && (
          <div className="pt-2 border-t border-border">
            <div className="text-xs text-text-secondary space-y-1">
              <div className="flex justify-between">
                <span>Seed Papers:</span>
                <span className="font-medium">{seedPapers.length}</span>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="pt-2 border-t border-border space-y-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => openModal('addPaper')}
            className="w-full text-xs"
          >
            Add Papers
          </Button>
          
          {seedPapers.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => openModal('export')}
              className="w-full text-xs"
            >
              Export Network
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

export default NetworkControls;