<div align="center">
<img width="1200" height="475" alt="Banner" src="https://raw.githubusercontent.com/omshukla/remix--4d-entity-simulator/main/.github/banner.png" />
</div>

# Ursina Swarm Combat & Simulation Engine

A high-performance 3D swarm simulation and combat engine built with React, Three.js, and TypeScript. It combines Craig Reynolds boids, reinforcement learning, real-time terrain deformation, and kaleidoscopic spatial geometry into a single interactive web application.

## Features

- **Swarm Combat Simulation** — Party A (Repel Guardian) vs Party B (Summoner) with minion AI using Craig Reynolds boids and aerial flight dynamics.
- **4-State Swarm Spear Attacks** — Coordinated offensive maneuvers with parabolic projectile arcs, lava pool hazards, and Thermal Panic flee behavior.
- **Q-Learning Reinforcement Learning** — Real-time Bellman-update tactical brain optimizing chase, warp, control, and repel actions with discrete state quantization.
- **Trait Survival Evolution** — Minions accumulate survival age and unlock adaptive traits: Agile Speed, Invulnerability Shields, Apex Titan Colossus, and Hellfire Artillery.
- **Doctor Strange Mirror Dimension** — Polar symmetry folding, angular domain repetition, and dynamic spatial accordion shearing for kaleidoscopic terrain.
- **Polar Mandala Spatial Matrix** — Decoupled polar spatial engine for headless RL training (thousands of epochs/sec) and real-time 3D rendering.
- **Real-Time Deformable Cubic Terrain** — Discrete 3D cube terrain with Small Cubes (3.5m), Big Cubes (7.0m), and Custom Cuboids that stack into towers.
- **Interactive Controls** — Full parameter panel with terrain deformation, swarm tuning, RL brain controls, and keyboard shortcuts.

## Tech Stack

- **Frontend:** React 19, TypeScript, Tailwind CSS, Motion (Framer Motion)
- **3D Rendering:** Three.js
- **Build Tool:** Vite
- **Python Integration:** Ursina / Panda3D scripts for headless RL and PBR rendering

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The app runs at `http://localhost:3000` by default.

### Build

```bash
npm run build
```

### Lint

```bash
npm run lint
```

## Project Structure

```
src/
  simulation/
    simulationEngine.ts       # Core simulation loop and entity state
    swarmFieldEngine.ts       # O(N) 3-equation flow field & spatial grid
    spatialMatrix.ts          # Polar mandala spatial matrix
    mirrorDimension.ts        # Kaleidoscope mirror geometry
    huntingRLAgent.ts         # Tabular Q-learning tactical brain
  rendering/
    snakeFactory.ts           # Snake entity mesh generation
    mannequinFactory.ts       # Humanoid mannequin mesh generation
  components/
    SimulationViewport.tsx    # Three.js scene and renderer
    SimulationControls.tsx    # Interactive parameter panel
    Header.tsx                # Top bar with stats and shortcuts
    UrsinaCodeModal.tsx       # Python script viewer for Ursina/Panda3D
  audio/
    soundEffects.ts           # Audio synthesis and effects
  types.ts                    # TypeScript interfaces and config
  App.tsx                     # Root component and state management
  main.tsx                    # React entry point
```

## Controls

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `R` | Reset Simulation |
| `E` | Manual Repel |
| `S` | Manual Spawn Minion |
| `F` | Trigger Fusion |
| `T` / `X` | Trigger Spear Attack |
| `C` | Toggle Code Modal |
| `H` | Toggle Banners |
| `B` / `Tab` | Toggle Sidebar |

## Architecture

The engine is split into three major layers:

1. **Simulation Layer** — Pure TypeScript simulation loop with deterministic boids, collision detection, and spatial hashing. Runs independently of the renderer for headless RL training.
2. **Rendering Layer** — Three.js scene graph managed by React components. Entity meshes are generated procedurally (snakes, mannequins, lava pools, crystalline prisms).
3. **Control Layer** — React state management with `useSyncExternalStore`-style polling for HUD updates and a comprehensive sidebar for live parameter tuning.

The decoupled design allows the same simulation engine to drive both the web UI and headless Python training scripts included in the UrsinaCodeModal.

## Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server on port 3000 |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |
| `npm run lint` | TypeScript type check |
| `npm run clean` | Remove build artifacts |

## License

MIT
