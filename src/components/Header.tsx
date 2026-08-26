import React from 'react';
import {
  Shield,
  Flame,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Code2,
  Sliders,
  Brain,
  User,
  Heart,
  Timer,
  Trophy,
  Eye,
  EyeOff,
} from 'lucide-react';
import { simulationSound } from '../audio/soundEffects';

interface HeaderProps {
  isRunning: boolean;
  timeScale: number;
  playerHealth: number;
  playerMaxHealth: number;
  survivalTime: number;
  bestSurvivalTime: number;
  activeBoids: number;
  repelCount: number;
  totalEliminated: number;
  controlMode: 'rl_agent' | 'manual_player';
  isSidebarOpen: boolean;
  isZenMode?: boolean;
  onToggleSidebar: () => void;
  onToggleZenMode?: () => void;
  onToggleRunning: () => void;
  onReset: () => void;
  onChangeTimeScale: (scale: number) => void;
  onOpenCodeModal: () => void;
  onToggleControlMode: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  isRunning,
  timeScale,
  playerHealth,
  playerMaxHealth,
  survivalTime,
  bestSurvivalTime,
  activeBoids,
  repelCount,
  totalEliminated,
  controlMode,
  isSidebarOpen,
  isZenMode = false,
  onToggleSidebar,
  onToggleZenMode,
  onToggleRunning,
  onReset,
  onChangeTimeScale,
  onOpenCodeModal,
  onToggleControlMode,
}) => {
  const [soundMuted, setSoundMuted] = React.useState(simulationSound.getIsMuted());

  const handleToggleSound = () => {
    const muted = simulationSound.toggleMuted();
    setSoundMuted(muted);
  };

  const hpPercent = Math.max(0, Math.min(100, (playerHealth / playerMaxHealth) * 100));

  return (
    <header
      id="app-header"
      className="flex items-center justify-between px-4 py-2.5 bg-slate-950/95 backdrop-blur-md border-b border-slate-800 z-30 select-none shadow-xl"
    >
      {/* Title & Agent Mode Switch */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-red-600 to-rose-500 flex items-center justify-center shadow-lg shadow-red-500/20 border border-red-400/40">
          <Flame className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-bold text-slate-100 tracking-tight">
              Red vs Swarm
            </h1>
            <button
              onClick={onToggleControlMode}
              title="Toggle between Autonomous RL Survival Agent and Manual Player Control"
              className={`flex items-center gap-1 text-[11px] font-mono font-semibold px-2 py-0.5 rounded-full border transition-all ${
                controlMode === 'rl_agent'
                  ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 hover:bg-rose-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/30'
              }`}
            >
              {controlMode === 'rl_agent' ? (
                <>
                  <Brain className="w-3 h-3 text-rose-400" />
                  <span>RL Agent Mode</span>
                </>
              ) : (
                <>
                  <User className="w-3 h-3 text-emerald-400" />
                  <span>Manual Control</span>
                </>
              )}
            </button>
          </div>
          <p className="text-[11px] text-slate-400 hidden sm:block">
            One Player RL Survival Engine
          </p>
        </div>
      </div>

      {/* Center Live Telemetry */}
      <div className="hidden md:flex items-center gap-3 font-mono text-xs">
        {/* Red Player Health Bar */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800">
          <Heart className="w-3.5 h-3.5 text-rose-400 animate-pulse" />
          <div className="w-20 bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700">
            <div
              className={`h-full transition-all duration-150 ${
                hpPercent < 30 ? 'bg-red-500' : hpPercent < 60 ? 'bg-amber-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${hpPercent}%` }}
            />
          </div>
          <span className="font-bold text-slate-200">{Math.round(playerHealth)} HP</span>
        </div>

        {/* Survival Time */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800 text-slate-300">
          <Timer className="w-3.5 h-3.5 text-cyan-400" />
          <span>Survived:</span>
          <strong className="text-cyan-300 font-bold">{survivalTime.toFixed(1)}s</strong>
        </div>

        {/* Best Record */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-950/40 border border-amber-900/60 text-amber-300">
          <Trophy className="w-3.5 h-3.5 text-amber-400" />
          <span>Best:</span>
          <strong className="text-amber-200 font-bold">{bestSurvivalTime.toFixed(1)}s</strong>
        </div>

        {/* Swarm Count */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-red-950/40 border border-red-900/60 text-red-300">
          <Flame className="w-3.5 h-3.5 text-red-400" />
          <span>Swarm:</span>
          <strong className="text-red-200 font-bold">{activeBoids}</strong>
        </div>
      </div>

      {/* Right Controls & Quick Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onToggleRunning}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all shadow-md ${
            isRunning
              ? 'bg-amber-600 hover:bg-amber-500 text-white'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white'
          }`}
        >
          {isRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          <span className="hidden sm:inline">{isRunning ? 'Pause' : 'Play'}</span>
        </button>

        {/* Speed multiplier */}
        <div className="hidden sm:flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
          {[1, 2, 5].map((scale) => (
            <button
              key={scale}
              onClick={() => onChangeTimeScale(scale)}
              className={`px-2 py-0.5 rounded font-mono transition-colors ${
                timeScale === scale
                  ? 'bg-red-600 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {scale}x
            </button>
          ))}
        </div>

        {/* Audio Toggle */}
        <button
          onClick={handleToggleSound}
          title={soundMuted ? 'Unmute Audio' : 'Mute Audio'}
          className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-colors"
        >
          {soundMuted ? <VolumeX className="w-4 h-4 text-slate-500" /> : <Volume2 className="w-4 h-4 text-rose-400" />}
        </button>

        {/* Code Modal Trigger */}
        <button
          onClick={onOpenCodeModal}
          title="View Python Ursina / Panda3D RL Simulation Engine Code"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-200 hover:text-white border border-slate-800 text-xs font-medium transition-colors"
        >
          <Code2 className="w-4 h-4 text-emerald-400" />
          <span className="hidden md:inline">Engine Code</span>
        </button>

        {/* Reset Episode */}
        <button
          onClick={onReset}
          title="Reset Episode"
          className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Zen Mode / Clean View Toggle */}
        {onToggleZenMode && (
          <button
            onClick={onToggleZenMode}
            title={isZenMode ? 'Show HUD & Panels (Press H)' : 'Clean View / Hide All Banners & Panels (Press H)'}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all ${
              isZenMode
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-slate-900 text-slate-300 border-slate-800 hover:bg-slate-800'
            }`}
          >
            {isZenMode ? <Eye className="w-4 h-4 text-amber-400" /> : <EyeOff className="w-4 h-4 text-slate-400" />}
            <span className="hidden sm:inline">{isZenMode ? 'Show UI [H]' : 'Clean View [H]'}</span>
          </button>
        )}

        {/* Sidebar Toggle */}
        <button
          onClick={onToggleSidebar}
          title={isSidebarOpen ? 'Hide Parameters Panel' : 'Show Parameters Panel'}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
            isSidebarOpen
              ? 'bg-red-600 text-white border-red-500 shadow-sm shadow-red-600/30 hover:bg-red-500'
              : 'bg-slate-900 text-slate-200 border-slate-700 hover:bg-slate-800 hover:text-white'
          }`}
        >
          <Sliders className="w-4 h-4 text-red-400" />
          <span className="inline">{isSidebarOpen ? 'Hide Panel' : 'Show Panel'}</span>
        </button>
      </div>
    </header>
  );
};
