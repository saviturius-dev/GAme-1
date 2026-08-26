import React, { useState, useRef } from 'react';
import {
  Brain,
  Flame,
  Shield,
  Layers,
  Sliders,
  RotateCcw,
  Zap,
  Activity,
  Heart,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Info,
  Dna,
  Compass,
  Cpu,
  Gauge,
  Play,
  Pause,
  RefreshCw,
  TrendingUp,
  Download,
  Upload,
  CheckCircle2,
  Save,
  Database,
  X,
  Palette,
  Timer,
  Eye,
} from 'lucide-react';
import { SimulationConfig, SimulationStats, SACRewardVariant, EmergentBehaviorProfile, SwarmColorMode } from '../types';

interface SimulationControlsProps {
  config: SimulationConfig;
  stats: SimulationStats;
  onUpdateConfig: (updates: Partial<SimulationConfig>) => void;
  onResetQTable: () => void;
  onClearTerrain: () => void;
  onSpawnSwarmCluster: (count: number) => void;
  onExportModel?: (type: 'sac' | 'qtable' | 'all') => void;
  onImportModel?: (file: File) => Promise<{ success: boolean; message: string }>;
  onResetModel?: (type: 'sac' | 'qtable' | 'all') => void;
  onSetBehaviorProfile?: (profile: EmergentBehaviorProfile) => void;
  onLoadOptimizedEmergentPolicy?: (profile?: EmergentBehaviorProfile) => void;
  onClose?: () => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  config,
  stats,
  onUpdateConfig,
  onResetQTable,
  onClearTerrain,
  onSpawnSwarmCluster,
  onExportModel,
  onImportModel,
  onResetModel,
  onSetBehaviorProfile,
  onLoadOptimizedEmergentPolicy,
  onClose,
}) => {
  const [openSections, setOpenSections] = useState<{ [key: string]: boolean }>({
    sacBrain: true,
    rlBrain: false,
    snakeSpecs: true,
    swarmDynamics: true,
    terrain: false,
  });

  const [importStatus, setImportStatus] = useState<{ message: string; isError: boolean } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onImportModel) return;

    try {
      const res = await onImportModel(file);
      setImportStatus({ message: res.message, isError: !res.success });
      setTimeout(() => setImportStatus(null), 4000);
    } catch (err: any) {
      setImportStatus({ message: err?.message || 'Failed to import model', isError: true });
      setTimeout(() => setImportStatus(null), 4000);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const toggleSection = (section: string) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const sacMetrics = stats.sacMetrics;
  const rlStats = stats.rlAgent;

  return (
    <div
      id="simulation-sidebar"
      className="w-80 sm:w-96 h-full bg-slate-950/95 backdrop-blur-md border-l border-slate-800 flex flex-col z-20 select-none shadow-2xl overflow-y-auto text-slate-200 text-xs shrink-0"
    >
      {/* Sidebar Header */}
      <div className="p-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950 sticky top-0 z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-red-600/20 border border-red-500/40 flex items-center justify-center text-red-400 shadow-sm">
            <Sliders className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-bold text-slate-100 text-sm">Simulation Parameters</h2>
            <span className="text-[10px] text-slate-400 font-mono">SAC Continuous • Physics • Swarm</span>
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            title="Hide Parameters Panel"
            className="p-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="p-4 space-y-4">
        {/* 1. Soft Actor-Critic (SAC) Continuous 3D RL Brain Section */}
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleSection('sacBrain')}
            className="w-full px-3.5 py-3 flex items-center justify-between bg-slate-900/90 hover:bg-slate-800/80 transition-colors font-bold text-slate-200"
          >
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-emerald-400" />
              <span>Soft Actor-Critic (SAC 3D)</span>
            </div>
            {openSections.sacBrain ? (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {openSections.sacBrain && (
            <div className="p-3.5 space-y-3.5 border-t border-slate-800/60">
              {/* Algorithm Enable Toggle */}
              <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
                <div className="flex flex-col">
                  <span className="text-[11px] font-bold text-slate-200">SAC Continuous Controller</span>
                  <span className="text-[10px] text-slate-400">Direct 3D continuous velocity policy $a_t \in [-1, 1]^3$</span>
                </div>
                <button
                  onClick={() => onUpdateConfig({ sacEnabled: !config.sacEnabled })}
                  className={`px-3 py-1 rounded-lg text-[11px] font-bold transition-all ${
                    config.sacEnabled
                      ? 'bg-emerald-600 text-white shadow'
                      : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {config.sacEnabled ? 'ACTIVE' : 'OFF'}
                </button>
              </div>

              {/* Mode: Training vs Evaluation */}
              <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
                <div className="flex flex-col">
                  <span className="text-[11px] font-bold text-slate-200">Policy Execution Mode</span>
                  <span className="text-[10px] text-slate-400">
                    {config.sacIsEvaluation ? 'Deterministic (Mean μ)' : 'Stochastic Exploration'}
                  </span>
                </div>
                <div className="flex gap-1">
                  <button
                    onClick={() => onUpdateConfig({ sacIsEvaluation: false })}
                    className={`px-2 py-1 rounded-lg text-[10px] font-bold transition-all ${
                      !config.sacIsEvaluation
                        ? 'bg-amber-600 text-white shadow'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    TRAIN
                  </button>
                  <button
                    onClick={() => onUpdateConfig({ sacIsEvaluation: true })}
                    className={`px-2 py-1 rounded-lg text-[10px] font-bold transition-all ${
                      config.sacIsEvaluation
                        ? 'bg-cyan-600 text-white shadow'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    EVAL
                  </button>
                </div>
              </div>

              {/* Emergent Behavior Tactics & Profile Selector */}
              <div className="bg-slate-900/90 p-3 rounded-xl border border-indigo-500/40 space-y-2.5 shadow-md">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-indigo-300 font-bold text-[11px]">
                    <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Emergent RL Tactics & Priors</span>
                  </div>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-950/80 border border-indigo-800/80 font-mono text-indigo-300 font-bold uppercase">
                    {sacMetrics?.behaviorProfile?.replace(/_/g, ' ') ?? 'Adaptive'}
                  </span>
                </div>

                <p className="text-[10px] text-slate-400 leading-tight">
                  Bootstraps neural actor-critic feature banks for real-time tactical emergence (spear dodging, baiting, vertical soaring).
                </p>

                <div className="grid grid-cols-2 gap-1.5">
                  {(
                    [
                      { id: 'adaptive_predator', label: 'Adaptive Master', desc: 'Hybrid dynamic tactics' },
                      { id: 'aerial_spear_hunter', label: 'Spear Dodger', desc: 'High-alt soaring evasion' },
                      { id: 'bait_blast_specialist', label: 'Bait & Blast', desc: 'Cluster lure & repel' },
                      { id: 'trench_barrier_architect', label: 'Trench Architect', desc: 'Chokepoint carving' },
                      { id: 'spiral_coil_tank', label: 'Spiral Coil', desc: 'Tight vertical defense' },
                      { id: 'max_growth_runner', label: 'Cruising Hunter', desc: 'Long-range perimeter sweep' },
                    ] as const
                  ).map((p) => {
                    const isSelected = (sacMetrics?.behaviorProfile ?? 'adaptive_predator') === p.id;
                    return (
                      <button
                        key={p.id}
                        onClick={() => onSetBehaviorProfile?.(p.id)}
                        className={`p-1.5 rounded-lg border text-left transition-all ${
                          isSelected
                            ? 'bg-indigo-600/30 border-indigo-400/70 text-indigo-200 shadow-sm font-semibold'
                            : 'bg-slate-950/80 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                        }`}
                      >
                        <div className="text-[10px] flex items-center justify-between">
                          <span>{p.label}</span>
                          {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />}
                        </div>
                        <div className="text-[8px] text-slate-500 truncate mt-0.5">{p.desc}</div>
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={() => onLoadOptimizedEmergentPolicy?.(sacMetrics?.behaviorProfile ?? 'adaptive_predator')}
                  className="w-full py-1.5 px-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-[10px] flex items-center justify-center gap-1.5 transition-all shadow-sm active:scale-95"
                >
                  <Sparkles className="w-3 h-3 text-indigo-200" />
                  <span>Bootstrap Tactical Feature Banks</span>
                </button>
              </div>

              {/* Reward Variant Formulation */}
              <div>
                <div className="flex justify-between text-[11px] mb-1.5">
                  <span className="text-slate-300 font-medium">Reward Formulation</span>
                  <span className="font-mono text-emerald-300 font-bold uppercase text-[10px]">
                    {config.sacRewardVariant}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-1">
                  {(
                    [
                      { id: 'variant_c_combined', label: 'Variant C', desc: 'α·L + β·ΔL - λ·Δa' },
                      { id: 'variant_a_net_growth', label: 'Variant A', desc: '2·ΔL/Ls (Growth)' },
                      { id: 'variant_b_max_size', label: 'Variant B', desc: 'L/Ls (Max Size)' },
                    ] as const
                  ).map((v) => (
                    <button
                      key={v.id}
                      onClick={() => onUpdateConfig({ sacRewardVariant: v.id })}
                      className={`p-1.5 rounded-lg border text-center transition-all ${
                        config.sacRewardVariant === v.id
                          ? 'bg-emerald-600/20 border-emerald-500/50 text-emerald-300 font-bold'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <div className="text-[10px]">{v.label}</div>
                      <div className="text-[8px] opacity-70 truncate">{v.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Action Persistence (k ticks) */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Action Persistence (Hold Ticks)</span>
                  <span className="font-mono text-cyan-300 font-bold">
                    {config.sacActionPersistence ?? 4} ticks ({(
                      ((config.sacActionPersistence ?? 4) * 0.016) * 1000
                    ).toFixed(0)}ms)
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={config.sacActionPersistence ?? 4}
                  onChange={(e) =>
                    onUpdateConfig({ sacActionPersistence: parseInt(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              {/* Learning Rate */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Learning Rate (α_lr)</span>
                  <span className="font-mono text-emerald-300 font-bold">
                    {(config.sacLearningRate ?? 3e-4).toExponential(1)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.00005"
                  max="0.001"
                  step="0.00005"
                  value={config.sacLearningRate ?? 3e-4}
                  onChange={(e) =>
                    onUpdateConfig({ sacLearningRate: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              {/* Discount Factor (Gamma) */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Discount Factor (γ)</span>
                  <span className="font-mono text-emerald-300 font-bold">
                    {(config.sacDiscountFactor ?? 0.99).toFixed(3)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.90"
                  max="0.999"
                  step="0.005"
                  value={config.sacDiscountFactor ?? 0.99}
                  onChange={(e) =>
                    onUpdateConfig({ sacDiscountFactor: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              {/* Soft Target Update Rate (Tau) */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Target Soft Polyak Rate (τ)</span>
                  <span className="font-mono text-cyan-300 font-bold">
                    {(config.sacTau ?? 0.005).toFixed(3)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.05"
                  step="0.001"
                  value={config.sacTau ?? 0.005}
                  onChange={(e) => onUpdateConfig({ sacTau: parseFloat(e.target.value) })}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              {/* Length Scale Constant Ls */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Length Scaling Denominator (L_s)</span>
                  <span className="font-mono text-amber-300 font-bold">
                    {(config.sacLengthScale ?? 50.0).toFixed(0)} units
                  </span>
                </div>
                <input
                  type="range"
                  min="10.0"
                  max="200.0"
                  step="5.0"
                  value={config.sacLengthScale ?? 50.0}
                  onChange={(e) =>
                    onUpdateConfig({ sacLengthScale: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>

              {/* Reward Coefficients Breakdown */}
              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                <div className="text-[11px] font-bold text-slate-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Variant C Weight Coefficients</span>
                </div>

                {/* Alpha Proportional Weight */}
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Length Weight (α)</span>
                    <span className="font-mono text-emerald-300 font-bold">
                      {(config.sacAlphaWeight ?? 1.0).toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="5.0"
                    step="0.1"
                    value={config.sacAlphaWeight ?? 1.0}
                    onChange={(e) =>
                      onUpdateConfig({ sacAlphaWeight: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                {/* Beta Delta Weight */}
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Delta Weight (β)</span>
                    <span className="font-mono text-cyan-300 font-bold">
                      {(config.sacBetaWeight ?? 1.0).toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="5.0"
                    step="0.1"
                    value={config.sacBetaWeight ?? 1.0}
                    onChange={(e) =>
                      onUpdateConfig({ sacBetaWeight: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
                  />
                </div>

                {/* Lambda Smoothness Penalty */}
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Action Smoothness (λ)</span>
                    <span className="font-mono text-rose-300 font-bold">
                      {(config.sacLambdaSmoothness ?? 0.005).toFixed(4)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="0.05"
                    step="0.001"
                    value={config.sacLambdaSmoothness ?? 0.005}
                    onChange={(e) =>
                      onUpdateConfig({ sacLambdaSmoothness: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-rose-500"
                  />
                </div>
              </div>

              {/* PER (Prioritized Experience Replay) & N-Step Multi-Step Returns */}
              <div className="pt-2 border-t border-slate-800/80 space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="text-[11px] font-bold text-slate-200 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-amber-400" />
                    <span>Prioritized Replay (PER) & N-Step</span>
                  </div>
                  <span className="px-1.5 py-0.5 rounded bg-amber-950/80 border border-amber-800/60 text-[9px] font-mono text-amber-300">
                    O(log N) SumTree
                  </span>
                </div>

                {/* PER Enable Toggle */}
                <div className="flex items-center justify-between bg-slate-950/60 p-2 rounded-xl border border-slate-800">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-200">Prioritized Sampling</span>
                    <span className="text-[9px] text-slate-400">Sample high TD-error transitions more frequently</span>
                  </div>
                  <button
                    onClick={() => onUpdateConfig({ sacUsePER: !(config.sacUsePER ?? true) })}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                      (config.sacUsePER ?? true) ? 'bg-amber-500' : 'bg-slate-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                        (config.sacUsePER ?? true) ? 'translate-x-4' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                {/* PER Alpha (Priority Exponent) */}
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Priority Exponent (α_per)</span>
                    <span className="font-mono text-amber-300 font-bold">
                      {(config.sacPerAlpha ?? 0.6).toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={config.sacPerAlpha ?? 0.6}
                    onChange={(e) =>
                      onUpdateConfig({ sacPerAlpha: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-500"
                  />
                </div>

                {/* PER Beta (Importance Sampling Weight) */}
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">IS Initial Beta (β_per)</span>
                    <span className="font-mono text-amber-300 font-bold">
                      {(config.sacPerBeta ?? 0.4).toFixed(2)} → 1.0 (annealed)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={config.sacPerBeta ?? 0.4}
                    onChange={(e) =>
                      onUpdateConfig({ sacPerBeta: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-500"
                  />
                </div>

                {/* N-Step Returns Toggle & Step Size */}
                <div className="flex items-center justify-between bg-slate-950/60 p-2 rounded-xl border border-slate-800">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-200">N-Step Bootstrapping</span>
                    <span className="text-[9px] text-slate-400">Accumulate discounted returns over n steps</span>
                  </div>
                  <button
                    onClick={() => onUpdateConfig({ sacUseNStep: !(config.sacUseNStep ?? true) })}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                      (config.sacUseNStep ?? true) ? 'bg-cyan-500' : 'bg-slate-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                        (config.sacUseNStep ?? true) ? 'translate-x-4' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">N-Step Horizon (n)</span>
                    <span className="font-mono text-cyan-300 font-bold">
                      {config.sacNStep ?? 3} steps
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={config.sacNStep ?? 3}
                    onChange={(e) =>
                      onUpdateConfig({ sacNStep: parseInt(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
                  />
                </div>

                {/* Live TD-Error Telemetry Badge */}
                {sacMetrics && (
                  <div className="grid grid-cols-2 gap-1.5 p-2 bg-slate-950/80 rounded-xl border border-slate-800/80 font-mono text-[10px]">
                    <div>
                      <span className="text-slate-500 block text-[9px]">MEAN TD ERROR</span>
                      <span className="text-amber-300 font-bold">
                        {(sacMetrics.meanTDError ?? 0).toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[9px]">MAX TD ERROR</span>
                      <span className="text-rose-400 font-bold">
                        {(sacMetrics.maxTDError ?? 0).toFixed(4)}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Learned Model Persistence & File Export/Import */}
              <div className="pt-2 border-t border-slate-800/80 space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="text-[11px] font-bold text-slate-200 flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Model Persistence & Files</span>
                  </div>
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-800 text-[10px] font-mono text-emerald-300">
                    <Save className="w-2.5 h-2.5 text-emerald-400 animate-pulse" />
                    Auto-Saving
                  </span>
                </div>

                <p className="text-[10px] text-slate-400 leading-relaxed">
                  RL weights, replay experiences, and policy networks are auto-persisted continuously. Export to a file to share or archive checkpoints.
                </p>

                {importStatus && (
                  <div
                    className={`p-2 rounded-xl border text-[11px] flex items-center gap-2 ${
                      importStatus.isError
                        ? 'bg-rose-950/60 border-rose-800/80 text-rose-200'
                        : 'bg-emerald-950/60 border-emerald-800/80 text-emerald-200'
                    }`}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                    <span>{importStatus.message}</span>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => onExportModel?.('sac')}
                    title="Export SAC Neural Weights to .json file"
                    className="py-1.5 px-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center justify-center gap-1.5 text-[11px] font-semibold transition-all active:scale-95 shadow-sm"
                  >
                    <Download className="w-3 h-3 text-emerald-400" />
                    <span>Export SAC .json</span>
                  </button>

                  <button
                    onClick={() => onExportModel?.('all')}
                    title="Export Complete Multi-Agent RL Bundle"
                    className="py-1.5 px-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center justify-center gap-1.5 text-[11px] font-semibold transition-all active:scale-95 shadow-sm"
                  >
                    <Download className="w-3 h-3 text-cyan-400" />
                    <span>Export All .json</span>
                  </button>
                </div>

                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    onChange={handleFileChange}
                    className="hidden"
                    id="sac-model-file-input"
                  />
                  <label
                    htmlFor="sac-model-file-input"
                    className="w-full py-1.5 px-3 rounded-xl bg-slate-800/90 hover:bg-slate-750 text-slate-300 hover:text-white border border-slate-700/80 flex items-center justify-center gap-1.5 text-[11px] font-semibold cursor-pointer transition-all active:scale-95"
                  >
                    <Upload className="w-3 h-3 text-amber-400" />
                    <span>Import Model File (.json)</span>
                  </label>
                </div>
              </div>

              {/* Reset Replay Buffer */}
              <button
                onClick={() => {
                  if (confirm('Reset SAC model weights and replay buffer? This will clear saved state.')) {
                    onResetModel?.('sac');
                  }
                }}
                className="w-full py-2 px-3 rounded-xl bg-rose-950/30 hover:bg-rose-900/50 text-rose-300 border border-rose-900/50 flex items-center justify-center gap-1.5 text-xs font-semibold transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5 text-rose-400" />
                <span>Reset SAC Policy & Weights</span>
              </button>
            </div>
          )}
        </div>

        {/* 2. Legacy Tabular Q-Learning Brain Section (Collapsible) */}
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleSection('rlBrain')}
            className="w-full px-3.5 py-3 flex items-center justify-between bg-slate-900/90 hover:bg-slate-800/80 transition-colors font-bold text-slate-200"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-rose-400" />
              <span>Tabular Q-Learning (Legacy)</span>
            </div>
            {openSections.rlBrain ? (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {openSections.rlBrain && (
            <div className="p-3.5 space-y-3.5 border-t border-slate-800/60">
              <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
                <span className="text-[11px] text-slate-300">Enable Tabular Q-Brain</span>
                <button
                  onClick={() =>
                    onUpdateConfig({ rlEnabled: !config.rlEnabled, sacEnabled: config.rlEnabled })
                  }
                  className={`px-3 py-1 rounded-lg text-[11px] font-bold ${
                    config.rlEnabled && !config.sacEnabled
                      ? 'bg-rose-600 text-white shadow'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {config.rlEnabled && !config.sacEnabled ? 'ACTIVE' : 'OFF'}
                </button>
              </div>

              {/* Exploration Epsilon */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Exploration Rate (ε)</span>
                  <span className="font-mono text-amber-300 font-bold">
                    {(config.rlEpsilon * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="1.0"
                  step="0.02"
                  value={config.rlEpsilon}
                  onChange={(e) => onUpdateConfig({ rlEpsilon: parseFloat(e.target.value) })}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>

              {/* Reset Q-Table */}
              <button
                onClick={onResetQTable}
                className="w-full py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center justify-center gap-1.5 text-xs font-semibold transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5 text-rose-400" />
                <span>Reset Q-Table</span>
              </button>
            </div>
          )}
        </div>

        {/* 3. Red Snake Mechanics & Kinematics Section */}
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleSection('snakeSpecs')}
            className="w-full px-3.5 py-3 flex items-center justify-between bg-slate-900/90 hover:bg-slate-800/80 transition-colors font-bold text-slate-200"
          >
            <div className="flex items-center gap-2">
              <Dna className="w-4 h-4 text-red-400" />
              <span>Continuous 360° Kinematics</span>
            </div>
            {openSections.snakeSpecs ? (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {openSections.snakeSpecs && (
            <div className="p-3.5 space-y-3.5 border-t border-slate-800/60">
              {/* Max Continuous Velocity Speed */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Forward Velocity (v_max)</span>
                  <span className="font-mono text-red-400 font-bold">
                    {(config.sacMaxSpeed ?? config.playerSpeed ?? 7.5).toFixed(1)} m/s
                  </span>
                </div>
                <input
                  type="range"
                  min="3.0"
                  max="18.0"
                  step="0.5"
                  value={config.sacMaxSpeed ?? config.playerSpeed ?? 7.5}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    onUpdateConfig({ sacMaxSpeed: v, playerSpeed: v, forwardVelocity: v });
                  }}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-red-500"
                />
              </div>

              {/* 3D Vertical Climb / Dive Speed */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">3D Climb / Dive Rate (v_vert)</span>
                  <span className="font-mono text-emerald-400 font-bold">
                    {(config.playerVerticalSpeed ?? 6.0).toFixed(1)} m/s
                  </span>
                </div>
                <input
                  type="range"
                  min="2.0"
                  max="15.0"
                  step="0.5"
                  value={config.playerVerticalSpeed ?? 6.0}
                  onChange={(e) =>
                    onUpdateConfig({ playerVerticalSpeed: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              {/* 3D Flight Ceiling */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">3D Flight Ceiling (y_max)</span>
                  <span className="font-mono text-cyan-300 font-bold">
                    {(config.snakeMaxAltitude ?? 36.0).toFixed(0)}m
                  </span>
                </div>
                <input
                  type="range"
                  min="10.0"
                  max="60.0"
                  step="2.0"
                  value={config.snakeMaxAltitude ?? 36.0}
                  onChange={(e) =>
                    onUpdateConfig({ snakeMaxAltitude: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              {/* Max Angular Turn Rate */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Turning Agility (ω_max)</span>
                  <span className="font-mono text-purple-400 font-bold">
                    {((config.maxTurnRate ?? Math.PI / 2) * (180 / Math.PI)).toFixed(0)}°/s
                  </span>
                </div>
                <input
                  type="range"
                  min={Math.PI / 4}
                  max={Math.PI}
                  step={0.05}
                  value={config.maxTurnRate ?? Math.PI / 2}
                  onChange={(e) =>
                    onUpdateConfig({ maxTurnRate: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>

              {/* Repel Radius */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Blast Repel Radius</span>
                  <span className="font-mono text-rose-300 font-bold">
                    {config.playerRepelRadius.toFixed(1)}m
                  </span>
                </div>
                <input
                  type="range"
                  min="3.0"
                  max="16.0"
                  step="0.5"
                  value={config.playerRepelRadius}
                  onChange={(e) =>
                    onUpdateConfig({ playerRepelRadius: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                />
              </div>

              {/* Repel Cooldown */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Repel Recharge Cooldown</span>
                  <span className="font-mono text-rose-400 font-bold">
                    {config.playerRepelCooldown.toFixed(1)}s
                  </span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="5.0"
                  step="0.1"
                  value={config.playerRepelCooldown}
                  onChange={(e) =>
                    onUpdateConfig({ playerRepelCooldown: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                />
              </div>

              {/* Tactical Dash Speed & Cooldown */}
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Dash Speed</span>
                    <span className="font-mono text-amber-300 font-bold">
                      {config.playerDashSpeed.toFixed(0)} m/s
                    </span>
                  </div>
                  <input
                    type="range"
                    min="10.0"
                    max="35.0"
                    step="1.0"
                    value={config.playerDashSpeed}
                    onChange={(e) =>
                      onUpdateConfig({ playerDashSpeed: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Dash Cooldown</span>
                    <span className="font-mono text-amber-300 font-bold">
                      {config.playerDashCooldown.toFixed(1)}s
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.3"
                    max="4.0"
                    step="0.1"
                    value={config.playerDashCooldown}
                    onChange={(e) =>
                      onUpdateConfig({ playerDashCooldown: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
              </div>

              {/* Continuous Growth Rate */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Continuous Growth Rate (g)</span>
                  <span className="font-mono text-emerald-300 font-bold">
                    +{(config.growthRate ?? 0.5).toFixed(2)} units/s
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="3.0"
                  step="0.05"
                  value={config.growthRate ?? 0.5}
                  onChange={(e) => onUpdateConfig({ growthRate: parseFloat(e.target.value) })}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              {/* Milestone Elongation Interval & Amount */}
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Elongate Interval</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      {(config.snakeElongateInterval ?? 3.0).toFixed(1)}s
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="10.0"
                    step="0.5"
                    value={config.snakeElongateInterval ?? 3.0}
                    onChange={(e) =>
                      onUpdateConfig({ snakeElongateInterval: parseFloat(e.target.value) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Growth Batch</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      +{(config.snakeGrowthAmount ?? 5)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="1"
                    value={config.snakeGrowthAmount ?? 5}
                    onChange={(e) =>
                      onUpdateConfig({ snakeGrowthAmount: parseInt(e.target.value, 10) })
                    }
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>
              </div>

              {/* Continuous Hit Penalty */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Hit Length Penalty (c)</span>
                  <span className="font-mono text-amber-300 font-bold">
                    -{(config.hitPenalty ?? 5.0).toFixed(1)} / bite
                  </span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="15.0"
                  step="0.5"
                  value={config.hitPenalty ?? 5.0}
                  onChange={(e) => onUpdateConfig({ hitPenalty: parseFloat(e.target.value) })}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>

              {/* Minimum Length Floor */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Minimum Length Floor (l_min)</span>
                  <span className="font-mono text-red-400 font-bold">
                    {config.snakeMinLength ?? 10} units
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="60"
                  step="1"
                  value={config.snakeMinLength ?? 10}
                  onChange={(e) =>
                    onUpdateConfig({ snakeMinLength: parseInt(e.target.value, 10) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-red-500"
                />
              </div>

              {/* Segment Spacing */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Vertebrae Segment Spacing</span>
                  <span className="font-mono text-slate-300 font-bold">
                    {(config.snakeSegmentSpacing ?? 0.42).toFixed(2)}m
                  </span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="0.8"
                  step="0.02"
                  value={config.snakeSegmentSpacing ?? 0.42}
                  onChange={(e) =>
                    onUpdateConfig({ snakeSegmentSpacing: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-slate-500"
                />
              </div>

              {/* Dynamic Tail Trailing Particle System Toggle */}
              <div className="flex items-center justify-between bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
                <div className="flex flex-col">
                  <span className="text-[10px] font-bold text-slate-200 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                    Tail Motion Particle Trail
                  </span>
                  <span className="text-[9px] text-slate-400">
                    Luminous living sparks and ethereal wake vortices tracing the tail path
                  </span>
                </div>
                <button
                  onClick={() =>
                    onUpdateConfig({
                      showSnakeTailTrail: !(config.showSnakeTailTrail ?? true),
                    })
                  }
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    (config.showSnakeTailTrail ?? true) ? 'bg-cyan-500' : 'bg-slate-700'
                  }`}
                >
                  <span
                    className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                      (config.showSnakeTailTrail ?? true) ? 'translate-x-4' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* SNAKE HEAD ANATOMY & KINEMATICS */}
              <div className="pt-3 border-t border-slate-800/80 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold text-cyan-300 flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                    Head Anatomy & Mechanics
                  </span>
                  {stats.headTelemetry && (
                    <span className="text-[9px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300 uppercase font-semibold">
                      {stats.headTelemetry.state.replace('_', ' ')}
                    </span>
                  )}
                </div>

                {/* Head Telemetry Quick Stats */}
                {stats.headTelemetry && (
                  <div className="grid grid-cols-2 gap-1.5 p-2 bg-slate-950/80 rounded-xl border border-slate-800/80 font-mono text-[10px]">
                    <div>
                      <span className="text-slate-500 block text-[8px] uppercase">Jaw Aperture</span>
                      <span className="text-cyan-300 font-bold">
                        {stats.headTelemetry.jawAngleDeg.toFixed(1)}°
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[8px] uppercase">Target Distance</span>
                      <span className="text-emerald-400 font-bold">
                        {stats.headTelemetry.eyeTargetDistance !== null
                          ? `${stats.headTelemetry.eyeTargetDistance.toFixed(1)}m`
                          : 'Idle'}
                      </span>
                    </div>
                  </div>
                )}

                {/* 3D Ocular Saccades Prey Tracking Toggle */}
                <div className="flex items-center justify-between bg-slate-950/60 p-2 rounded-xl border border-slate-800">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-200">3D Ocular Eye Tracking</span>
                    <span className="text-[9px] text-slate-400">Eyes & pupils saccade toward nearest swarm boids</span>
                  </div>
                  <button
                    onClick={() =>
                      onUpdateConfig({
                        snakeHeadEyeTracking: !(config.snakeHeadEyeTracking ?? true),
                      })
                    }
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                      (config.snakeHeadEyeTracking ?? true) ? 'bg-cyan-500' : 'bg-slate-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                        (config.snakeHeadEyeTracking ?? true) ? 'translate-x-4' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                {/* Articulated Jaw Style Selector */}
                <div>
                  <label className="text-slate-400 block text-[10px] mb-1">Mandible & Fang Anatomy</label>
                  <div className="grid grid-cols-3 gap-1">
                    {[
                      { id: 'predator_fang', label: 'Viper Needle Fangs' },
                      { id: 'cyber_plasma', label: 'Hinged Maxilla' },
                      { id: 'abyssal_viper', label: 'Recurved Teeth' },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        onClick={() =>
                          onUpdateConfig({
                            snakeHeadJawStyle: opt.id as any,
                          })
                        }
                        className={`py-1 px-1.5 rounded-lg text-[9px] font-medium border text-center transition-all ${
                          (config.snakeHeadJawStyle ?? 'predator_fang') === opt.id
                            ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300 font-bold'
                            : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Head Scutes & Brow Style Selector */}
                <div>
                  <label className="text-slate-400 block text-[10px] mb-1">Cranial Scutes & Brow Plating</label>
                  <div className="grid grid-cols-2 gap-1">
                    {[
                      { id: 'swept_horns', label: 'Supraocular Brows' },
                      { id: 'tactile_whiskers', label: 'Loreal Pits & Nares' },
                      { id: 'crown_crest', label: 'Parietal Armor Scutes' },
                      { id: 'minimal', label: 'Smooth Python Crown' },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        onClick={() =>
                          onUpdateConfig({
                            snakeHeadHornStyle: opt.id as any,
                          })
                        }
                        className={`py-1 px-1.5 rounded-lg text-[9px] font-medium border text-center transition-all ${
                          (config.snakeHeadHornStyle ?? 'swept_horns') === opt.id
                            ? 'bg-purple-500/20 border-purple-500 text-purple-300 font-bold'
                            : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 4. Autonomous Swarm Dynamics & Flow Section */}
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleSection('swarmDynamics')}
            className="w-full px-3.5 py-3 flex items-center justify-between bg-slate-900/90 hover:bg-slate-800/80 transition-colors font-bold text-slate-200"
          >
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-orange-400" />
              <span>Swarm Flow & Attraction</span>
            </div>
            {openSections.swarmDynamics ? (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {openSections.swarmDynamics && (
            <div className="p-3.5 space-y-3.5 border-t border-slate-800/60">
              {/* User-Manipulated Player Attraction Weight */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-300 font-medium flex items-center gap-1">
                    <Compass className="w-3 h-3 text-red-400" />
                    3D Snake Targeting & Pursuit
                  </span>
                  <span className="font-mono text-red-400 font-bold">
                    {(config.playerAttractionWeight ?? 2.2).toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="6.0"
                  step="0.1"
                  value={config.playerAttractionWeight ?? 2.2}
                  onChange={(e) =>
                    onUpdateConfig({ playerAttractionWeight: parseFloat(e.target.value) })
                  }
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-red-500"
                />
              </div>

              {/* 3D Flow Field Force */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">3D Perlin Flow Force</span>
                  <span className="font-mono text-amber-300 font-bold">
                    {config.flowFieldWeight.toFixed(1)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="5.0"
                  step="0.2"
                  value={config.flowFieldWeight}
                  onChange={(e) => onUpdateConfig({ flowFieldWeight: parseFloat(e.target.value) })}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
              </div>

              {/* Max Swarm Points Cap */}
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Max Swarm Capacity</span>
                  <span className="font-mono text-orange-300 font-bold">
                    {config.maxPointsCap} boids
                  </span>
                </div>
                <input
                  type="range"
                  min="500"
                  max="10000"
                  step="500"
                  value={config.maxPointsCap}
                  onChange={(e) => onUpdateConfig({ maxPointsCap: parseInt(e.target.value) })}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-500"
                />
              </div>

              {/* Quick Spawn Buttons */}
              <div className="grid grid-cols-2 gap-2 pt-1">
                <button
                  onClick={() => onSpawnSwarmCluster(25)}
                  className="py-1.5 px-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-[11px] font-semibold"
                >
                  +25 Boids
                </button>
                <button
                  onClick={() => onSpawnSwarmCluster(100)}
                  className="py-1.5 px-2 rounded-xl bg-orange-600/20 hover:bg-orange-600/30 text-orange-300 border border-orange-500/40 text-[11px] font-bold"
                >
                  +100 Cluster
                </button>
              </div>

              {/* Dynamic Swarm Visual Optics & Color Mode Selector */}
              <div className="pt-2 border-t border-slate-800/80">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200">
                    <Palette className="w-3.5 h-3.5 text-pink-400" />
                    <span>Swarm Color & Visual Feedback</span>
                  </div>
                  <span className="text-[9px] font-mono text-pink-400 font-semibold uppercase">
                    {(config.swarmColorMode ?? 'survival_age').replace(/_/g, ' ')}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-1.5 mb-2.5">
                  {[
                    {
                      id: 'survival_age',
                      label: 'Survival Age Spectrum',
                      desc: 'Cyan newborn → Lime → Gold → Crimson → Plasma Violet → Diamond White',
                      color: 'from-cyan-400 via-amber-400 to-fuchsia-400',
                    },
                    {
                      id: 'active_state',
                      label: 'Active Attack State',
                      desc: 'Flocking (Gold), Spear Form (Cyan/Violet), Thrust (White/Pink), Disperse (Orange)',
                      color: 'from-amber-400 via-cyan-400 to-pink-500',
                    },
                    {
                      id: 'hybrid_age_state',
                      label: 'Hybrid State & Age',
                      desc: 'Attack formation modulated by veteran diamond plasma glow',
                      color: 'from-violet-400 via-pink-400 to-cyan-300',
                    },
                    {
                      id: 'state_dynamic',
                      label: 'Tactical Murmuration',
                      desc: 'Velocity interpolation & lance spear color transition',
                      color: 'from-amber-400 to-red-500',
                    },
                    {
                      id: 'generation_lineage',
                      label: 'Replication Generation',
                      desc: 'Lineage tier: Gen 0 Gold → Gen 1 Lime → Gen 2 Cyan → Gen 3+ Magenta',
                      color: 'from-emerald-400 via-cyan-400 to-fuchsia-500',
                    },
                    {
                      id: 'kinetic_energy',
                      label: 'Kinetic Speed Heatmap',
                      desc: 'Thermal velocity map: Azure Blue → Solar Gold → Hypersonic White',
                      color: 'from-blue-500 via-amber-400 to-white',
                    },
                  ].map((mode) => {
                    const isSelected = (config.swarmColorMode ?? 'survival_age') === mode.id;
                    return (
                      <button
                        key={mode.id}
                        onClick={() =>
                          onUpdateConfig({ swarmColorMode: mode.id as SwarmColorMode })
                        }
                        className={`p-2 rounded-xl border text-left transition-all relative overflow-hidden flex flex-col justify-between ${
                          isSelected
                            ? 'bg-slate-800/90 border-pink-500/80 shadow-[0_0_12px_rgba(236,72,153,0.15)] ring-1 ring-pink-500/50'
                            : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1 mb-1">
                          <span className={`text-[10px] font-bold ${isSelected ? 'text-pink-300' : 'text-slate-300'}`}>
                            {mode.label}
                          </span>
                          <div className={`w-3.5 h-1.5 rounded-full bg-gradient-to-r ${mode.color} shrink-0`} />
                        </div>
                        <span className="text-[8.5px] text-slate-400 line-clamp-2 leading-tight">
                          {mode.desc}
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Survival Age Threshold Slider (shown when survival_age or hybrid_age_state is active) */}
                {((config.swarmColorMode ?? 'survival_age') === 'survival_age' ||
                  config.swarmColorMode === 'hybrid_age_state') && (
                  <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 mb-2">
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-slate-300 font-medium flex items-center gap-1">
                        <Timer className="w-3 h-3 text-cyan-400" />
                        Ancient Boid Lifespan Target
                      </span>
                      <span className="font-mono text-cyan-400 font-bold">
                        {(config.swarmAgeMaxThreshold ?? 20.0).toFixed(0)}s max
                      </span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="60"
                      step="1"
                      value={config.swarmAgeMaxThreshold ?? 20.0}
                      onChange={(e) =>
                        onUpdateConfig({ swarmAgeMaxThreshold: parseFloat(e.target.value) })
                      }
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                    />
                    <div className="flex justify-between text-[8px] text-slate-500 mt-1 font-mono">
                      <span>5s Fast Cycle</span>
                      <span>20s Standard</span>
                      <span>60s Ancient</span>
                    </div>
                  </div>
                )}

                {/* Veteran Dynamic Particle Size Toggle */}
                <div className="flex items-center justify-between bg-slate-950/60 p-2 rounded-xl border border-slate-800 mb-2">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-200">Longevity Particle Scale</span>
                    <span className="text-[9px] text-slate-400">Elder & veteran boids scale up slightly in size</span>
                  </div>
                  <button
                    onClick={() =>
                      onUpdateConfig({
                        swarmDynamicSizeByAge: !(config.swarmDynamicSizeByAge ?? true),
                      })
                    }
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                      (config.swarmDynamicSizeByAge ?? true) ? 'bg-pink-500' : 'bg-slate-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                        (config.swarmDynamicSizeByAge ?? true) ? 'translate-x-4' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>

                {/* Real-Time Longevity & Demographics Telemetry */}
                <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                  <div className="flex items-center justify-between border-b border-slate-800/60 pb-1.5">
                    <span className="text-[10px] font-bold text-slate-300 flex items-center gap-1.5">
                      <Activity className="w-3 h-3 text-emerald-400" />
                      Swarm Demographics & Attack State
                    </span>
                    <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
                      stats.swarmAttackState === 2
                        ? 'bg-pink-500/20 text-pink-300 border border-pink-500/40'
                        : stats.swarmAttackState === 1
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : stats.swarmAttackState === 3
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}>
                      {stats.swarmAttackStateName ?? 'Flocking'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                      <span className="text-slate-500 block text-[8px] uppercase font-mono">Avg Survival Time</span>
                      <span className="text-cyan-300 font-mono font-bold text-xs">
                        {(stats.avgBoidSurvivalTime ?? 0).toFixed(1)}s
                      </span>
                    </div>
                    <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800">
                      <span className="text-slate-500 block text-[8px] uppercase font-mono">Oldest Boid (Veteran)</span>
                      <span className="text-amber-300 font-mono font-bold text-xs">
                        {(stats.maxBoidSurvivalTime ?? 0).toFixed(1)}s
                      </span>
                    </div>
                  </div>

                  {/* Demographics Spectrum Distribution Bar */}
                  <div>
                    <div className="flex justify-between text-[8.5px] text-slate-400 mb-1 font-mono">
                      <span>Age Spectrum Breakdown</span>
                      <span>{stats.activeBoids} Total</span>
                    </div>
                    {(() => {
                      const total = Math.max(1, stats.activeBoids);
                      const d = stats.swarmDemographics ?? { newborn: 0, youth: 0, mature: 0, veteran: 0, ancient: 0 };
                      const pNew = (d.newborn / total) * 100;
                      const pYouth = (d.youth / total) * 100;
                      const pMat = (d.mature / total) * 100;
                      const pVet = (d.veteran / total) * 100;
                      const pAnc = (d.ancient / total) * 100;

                      return (
                        <>
                          <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden flex border border-slate-800">
                            <div style={{ width: `${pNew}%` }} className="bg-cyan-400 transition-all duration-300" title={`Newborn (<2s): ${d.newborn}`} />
                            <div style={{ width: `${pYouth}%` }} className="bg-amber-400 transition-all duration-300" title={`Youth (2-6s): ${d.youth}`} />
                            <div style={{ width: `${pMat}%` }} className="bg-rose-500 transition-all duration-300" title={`Mature (6-15s): ${d.mature}`} />
                            <div style={{ width: `${pVet}%` }} className="bg-purple-500 transition-all duration-300" title={`Veteran (15-30s): ${d.veteran}`} />
                            <div style={{ width: `${pAnc}%` }} className="bg-white transition-all duration-300" title={`Ancient (>30s): ${d.ancient}`} />
                          </div>
                          <div className="grid grid-cols-5 gap-1 text-[7.5px] font-mono text-slate-400 mt-1 text-center">
                            <div><span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400 mr-0.5" />New: {d.newborn}</div>
                            <div><span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 mr-0.5" />Yth: {d.youth}</div>
                            <div><span className="inline-block w-1.5 h-1.5 rounded-full bg-rose-500 mr-0.5" />Mat: {d.mature}</div>
                            <div><span className="inline-block w-1.5 h-1.5 rounded-full bg-purple-500 mr-0.5" />Vet: {d.veteran}</div>
                            <div><span className="inline-block w-1.5 h-1.5 rounded-full bg-white mr-0.5" />Anc: {d.ancient}</div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 5. Tactical Deformable Terrain Trenches */}
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleSection('terrain')}
            className="w-full px-3.5 py-3 flex items-center justify-between bg-slate-900/90 hover:bg-slate-800/80 transition-colors font-bold text-slate-200"
          >
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Deformable Terrain Trenches</span>
            </div>
            {openSections.terrain ? (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {openSections.terrain && (
            <div className="p-3.5 space-y-3.5 border-t border-slate-800/60">
              <div className="text-[11px] text-slate-400">
                Carve deep ground trenches and subterranean channels to fracture swarm murmuration flow.
              </div>

              {/* Clear Trenches */}
              <button
                onClick={onClearTerrain}
                className="w-full py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center justify-center gap-1.5 text-xs font-semibold transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
                <span>Clear All Terrain Trenches</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
