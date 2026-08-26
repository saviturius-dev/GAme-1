import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { SimulationViewport } from './components/SimulationViewport';
import { SimulationControls } from './components/SimulationControls';
import { UrsinaCodeModal } from './components/UrsinaCodeModal';
import { UrsinaRepelSimulation } from './simulation/simulationEngine';
import { SimulationConfig } from './types';

export function App() {
  const simRef = useRef<UrsinaRepelSimulation | null>(null);

  if (!simRef.current) {
    simRef.current = new UrsinaRepelSimulation();
  }

  const simulation = simRef.current;

  const [isRunning, setIsRunning] = useState(true);
  const [timeScale, setTimeScale] = useState(1);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isZenMode, setIsZenMode] = useState(false);
  const [isCodeModalOpen, setIsCodeModalOpen] = useState(false);
  const [config, setConfig] = useState<SimulationConfig>(simulation.config);
  const [, setTick] = useState(0);

  // Key tracking for manual WASD control
  const keysPressed = useRef<{ [key: string]: boolean }>({});

  const handleToggleZenMode = useCallback(() => {
    setIsZenMode((prev) => !prev);
  }, []);

  const handleToggleSidebar = useCallback(() => {
    if (isZenMode) {
      setIsZenMode(false);
      setIsSidebarOpen(true);
    } else {
      setIsSidebarOpen((prev) => !prev);
    }
  }, [isZenMode]);

  // Frame tick for React state synchronization
  useEffect(() => {
    let animId: number;
    const updateLoop = () => {
      // Handle continuous 360° omnidirectional steering & 3D altitudinal flight for manual player
      if (simulation.player.controlMode === 'manual_player' && !simulation.player.isDead) {
        const keys = keysPressed.current;
        const style = simulation.config.manualControlStyle ?? 'omnidirectional_vector';

        if (style === 'tank_steering') {
          // Direct steering angular velocity input
          let steer = 0;
          if (keys['KeyA'] || keys['ArrowLeft']) steer -= 1.0;
          if (keys['KeyD'] || keys['ArrowRight']) steer += 1.0;
          simulation.player.steeringInput = steer;
          simulation.player.targetHeading = undefined;
        } else {
          // Continuous 360° omnidirectional vector steering
          let dx = 0;
          let dz = 0;
          if (keys['KeyW'] || keys['ArrowUp']) dz -= 1;
          if (keys['KeyS'] || keys['ArrowDown']) dz += 1;
          if (keys['KeyA'] || keys['ArrowLeft']) dx -= 1;
          if (keys['KeyD'] || keys['ArrowRight']) dx += 1;

          if (dx !== 0 || dz !== 0) {
            // Target heading in continuous 360-degree radian space
            const targetHeading = Math.atan2(dx, -dz);
            simulation.player.targetHeading = targetHeading;
          } else {
            simulation.player.targetHeading = undefined;
            simulation.player.steeringInput = 0;
          }
        }

        // 3D Altitude Flight Control (Climb: R / E / PageUp; Dive: F / Q / PageDown)
        if (keys['KeyR'] || keys['PageUp']) {
          simulation.manualAscend(0.05);
        } else if (keys['KeyF'] || keys['PageDown']) {
          simulation.manualDescend(0.05);
        } else {
          simulation.resetVerticalControl();
        }
      }

      setTick((t) => (t + 1) % 1000000);
      animId = requestAnimationFrame(updateLoop);
    };

    animId = requestAnimationFrame(updateLoop);
    return () => cancelAnimationFrame(animId);
  }, [simulation]);

  // Global Keyboard event listeners
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept when typing in text inputs or modal is open
      if (e.target instanceof HTMLInputElement || isCodeModalOpen) return;

      keysPressed.current[e.code] = true;

      // Space -> Repel Blast
      if (e.code === 'Space') {
        e.preventDefault();
        simulation.manualRepel();
      }

      // Shift -> Dash
      if (e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        simulation.manualDash();
      }

      // T or KeyC -> Carve Trench
      if (e.code === 'KeyT' || e.code === 'KeyC') {
        simulation.carveTrench();
      }

      // M -> Toggle RL Agent / Manual Control
      if (e.code === 'KeyM') {
        const nextMode =
          simulation.player.controlMode === 'rl_agent' ? 'manual_player' : 'rl_agent';
        simulation.setControlMode(nextMode);
      }

      // H -> Toggle Zen / Minimal View (hide all banners & panel)
      if (e.code === 'KeyH' || e.code === 'KeyU') {
        handleToggleZenMode();
      }

      // S -> Spawn Swarm (when not pressing W or in RL mode)
      if (e.code === 'KeyS' && !keysPressed.current['KeyW']) {
        if (simulation.player.controlMode !== 'manual_player') {
          simulation.manualSpawnSwarm(20);
        }
      }

      // KeyX or Shift+R -> Reset Arena
      if (e.code === 'KeyX' || (e.code === 'KeyR' && e.shiftKey)) {
        simulation.reset(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      keysPressed.current[e.code] = false;
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [simulation, isCodeModalOpen, handleToggleZenMode]);

  const handleUpdateConfig = useCallback(
    (updates: Partial<SimulationConfig>) => {
      setConfig((prev) => {
        const next = { ...prev, ...updates };
        // Instantly propagate all ability & physics parameters into simulation engine
        simulation.updateConfig(next);
        return next;
      });
    },
    [simulation]
  );

  const handleToggleControlMode = () => {
    const nextMode =
      simulation.player.controlMode === 'rl_agent' ? 'manual_player' : 'rl_agent';
    simulation.setControlMode(nextMode);
  };

  const handleResetQTable = () => {
    simulation.resetRLModel('all');
  };

  const handleExportModel = (type: 'sac' | 'qtable' | 'all') => {
    simulation.exportModelToFile(type);
  };

  const handleImportModel = async (file: File) => {
    return await simulation.importModelFromFile(file);
  };

  const handleResetModel = (type: 'sac' | 'qtable' | 'all') => {
    simulation.resetRLModel(type);
  };

  const handleClearTerrain = () => {
    simulation.resetTerrain();
  };

  const handleSpawnSwarm = (count: number = 20) => {
    simulation.manualSpawnSwarm(count);
  };

  const handleReset = () => {
    simulation.reset(true);
  };

  return (
    <div className="flex flex-col w-screen h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans select-none">
      {/* Top Application Header (Hidden in Zen Mode) */}
      {!isZenMode && (
        <Header
          isRunning={isRunning}
          timeScale={timeScale}
          playerHealth={simulation.player.health}
          playerMaxHealth={simulation.player.maxHealth}
          survivalTime={simulation.player.survivalTime}
          bestSurvivalTime={simulation.stats.bestSurvivalTime}
          activeBoids={simulation.swarmField.count}
          repelCount={simulation.stats.repelCount}
          totalEliminated={simulation.stats.totalRepelled}
          controlMode={simulation.player.controlMode}
          isSidebarOpen={isSidebarOpen}
          isZenMode={isZenMode}
          onToggleSidebar={handleToggleSidebar}
          onToggleZenMode={handleToggleZenMode}
          onToggleRunning={() => setIsRunning(!isRunning)}
          onReset={handleReset}
          onChangeTimeScale={(scale) => setTimeScale(scale)}
          onOpenCodeModal={() => setIsCodeModalOpen(true)}
          onToggleControlMode={handleToggleControlMode}
        />
      )}

      {/* Main Workspace Layout */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* 3D WebGL Viewport */}
        <div className="flex-1 relative h-full">
          <SimulationViewport
            simulation={simulation}
            config={config}
            isRunning={isRunning}
            isSidebarOpen={isSidebarOpen && !isZenMode}
            isZenMode={isZenMode}
            onToggleSidebar={handleToggleSidebar}
            onToggleZenMode={handleToggleZenMode}
            onUpdateConfig={handleUpdateConfig}
            onManualRepel={() => simulation.manualRepel()}
            onManualDash={() => simulation.manualDash()}
            onManualAscend={() => simulation.manualAscend(0.05)}
            onManualDescend={() => simulation.manualDescend(0.05)}
            onCarveTrench={() => simulation.carveTrench()}
            onManualSpawn={() => simulation.manualSpawnSwarm(20)}
            onReset={handleReset}
            onToggleControlMode={handleToggleControlMode}
          />
        </div>

        {/* Collapsible Control Sidebar (Hidden in Zen Mode) */}
        {isSidebarOpen && !isZenMode && (
          <SimulationControls
            config={config}
            stats={simulation.stats}
            onUpdateConfig={handleUpdateConfig}
            onResetQTable={handleResetQTable}
            onClearTerrain={handleClearTerrain}
            onSpawnSwarmCluster={handleSpawnSwarm}
            onExportModel={handleExportModel}
            onImportModel={handleImportModel}
            onResetModel={handleResetModel}
            onSetBehaviorProfile={(profile) => simulation.setEmergentBehaviorProfile(profile)}
            onLoadOptimizedEmergentPolicy={(profile) => simulation.loadOptimizedEmergentPolicy(profile)}
            onClose={() => setIsSidebarOpen(false)}
          />
        )}
      </div>

      {/* Python Ursina Code Modal */}
      <UrsinaCodeModal isOpen={isCodeModalOpen} onClose={() => setIsCodeModalOpen(false)} />
    </div>
  );
}

export default App;
