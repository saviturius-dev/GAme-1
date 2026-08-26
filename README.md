# Ursina Swarm Combat & Simulation Engine

A professional real-time 3D swarm combat sandbox built with React, Three.js, TypeScript, and browser-native reinforcement learning. The project models an autonomous luminous point-cloud swarm, a continuous-control red serpent survivor, deformable cubic terrain, and both modern actor-critic and classic tabular RL control loops in a single interactive web application.

This README intentionally avoids generic generator badges, stock banners, or AI-watermark imagery. It is written as engineering documentation: what the system does, how it is organized, which algorithms it uses, and how to operate or extend it.

## Table of Contents

- [Core Capabilities](#core-capabilities)
- [System Architecture](#system-architecture)
- [Runtime Data Flow](#runtime-data-flow)
- [Swarm Rendering Architecture](#swarm-rendering-architecture)
- [Swarm Physics and Algorithms](#swarm-physics-and-algorithms)
- [Attack State Machine](#attack-state-machine)
- [Snake Survivor Simulation](#snake-survivor-simulation)
- [Reinforcement Learning Architecture](#reinforcement-learning-architecture)
- [Soft Actor-Critic Continuous Controller](#soft-actor-critic-continuous-controller)
- [Tabular Q-Learning Controller](#tabular-q-learning-controller)
- [Terrain Deformation and Tactical Geometry](#terrain-deformation-and-tactical-geometry)
- [Rendering Pipeline](#rendering-pipeline)
- [Audio and Feedback Systems](#audio-and-feedback-systems)
- [Project Structure](#project-structure)
- [Configuration Surface](#configuration-surface)
- [Controls](#controls)
- [Getting Started](#getting-started)
- [Scripts](#scripts)
- [Development Notes](#development-notes)
- [Extension Guide](#extension-guide)
- [License](#license)

## Core Capabilities

- **Autonomous luminous swarm simulation** using a zero-GC `Float32Array` / `Uint8Array` structure-of-arrays layout for particle positions, velocities, colors, health, generation, seed, and rendering properties.
- **High-throughput Three.js point-cloud rendering** through dynamic `THREE.BufferGeometry` attributes and additive `THREE.PointsMaterial` glow sprites.
- **3D Perlin flow-field locomotion** with deterministic gradient noise, sinusoidal per-boid micro-wobble, damped velocity integration, and terminal velocity capping.
- **Swarm self-replication** with high-frequency `1 -> 4` multiplication up to a configurable capacity limit.
- **4-state spear attack machine** covering flocking, spear formation, thrusting, and radial dispersal / impact burst behavior.
- **Continuous 3D snake survivor** with altitude-aware kinematics, dash, repel, barrier, trench, body-length growth, damage, recovery, and survival telemetry.
- **Soft Actor-Critic-style continuous RL controller** implemented in TypeScript with lightweight neural network utilities, actor / twin-critic updates, entropy temperature, replay buffer, prioritized experience replay, n-step returns, layer normalization, potential-based reward shaping, and curriculum-aware reward components.
- **Classic tabular Q-learning fallback** with discrete tactical states and an epsilon-greedy action policy for comparison against continuous control.
- **Tactical deformable cubic terrain** supporting raised cubes, carved trenches, stacked barriers, cuboids, shield detection, collision response, and swarm terrain adherence.
- **Interactive React control panel** for tuning swarm, terrain, camera, reinforcement learning, rendering, and audio parameters at runtime.
- **Embedded reference code modal** for related Gymnasium, PyTorch PPO, and Ursina / Panda3D implementations.

## System Architecture

The application is split into clear simulation, rendering, control, and support layers.

```text
React App
├── App-level lifecycle and global UI state
├── SimulationControls: live parameter editing and RL import/export controls
├── SimulationViewport: Three.js scene, geometry sync, HUD overlays, input handling
├── Header: shortcuts, app status, modal launchers
└── UrsinaCodeModal: reference code viewer for Python/PPO/Ursina variants

Simulation Core
├── UrsinaRepelSimulation: authoritative world state and per-frame step orchestration
├── SwarmFieldEngine: autonomous point swarm physics, replication, attack states, buffers
├── ContinuousSACAgent: continuous actor-critic RL policy and training telemetry
└── RedSurvivalRLAgent: tabular Q-learning baseline

Rendering and Effects
├── snakeFactory: procedural snake mesh construction
├── mannequinFactory: procedural humanoid/guardian mesh construction
├── snakeTailTrail: temporal trail geometry for motion readability
├── skyShader: environment shader utilities
└── soundEffects: procedural audio feedback
```

### Design Principles

1. **Simulation-first state ownership**: game state lives in `UrsinaRepelSimulation`, not in Three.js meshes. Rendering mirrors typed data each frame.
2. **Renderer decoupling**: the swarm and RL loops can run without depending on React component state.
3. **Typed-array hot paths**: swarm updates use preallocated buffers to reduce garbage collection pressure during dense simulations.
4. **Algorithmic visibility**: RL metrics, Q-values, replay status, rewards, action vectors, and emergent tactic labels are exposed through UI telemetry rather than hidden in black boxes.
5. **Autonomous swarm behavior**: the swarm is self-sustaining and does not require a summoner object to exist, spawn, replicate, navigate, or attack.

## Runtime Data Flow

1. `App.tsx` creates a single `UrsinaRepelSimulation` instance and stores it in a React ref.
2. UI controls update the simulation configuration through partial config patches.
3. `SimulationViewport` advances the simulation while mounted and synchronizes Three.js objects from simulation state.
4. `UrsinaRepelSimulation.update()` coordinates player motion, terrain, swarm replication, swarm physics, collisions, damage, RL decisions, reward accounting, and telemetry.
5. `SwarmFieldEngine.update()` mutates preallocated swarm buffers in place.
6. Rendering attributes read the same buffers for positions and colors, minimizing per-frame object allocation.
7. HUD and panels poll stats for visual feedback, training diagnostics, and user-facing controls.

## Swarm Rendering Architecture

The swarm is represented as a luminous 3D particle point cloud rather than thousands of individual mesh objects.

### Geometry Layout

`SwarmFieldEngine` allocates all particle storage up front:

- `boidX`, `boidY`, `boidZ`: scalar position channels.
- `boidVx`, `boidVy`, `boidVz`: scalar velocity channels.
- `boidAge`, `boidGen`, `boidSeed`: lifecycle and variation metadata.
- `boidHealth`, `boidMaxHealth`: combat durability.
- `positions`: flat `Float32Array` with three floats per point for direct geometry upload.
- `colors`: flat RGB `Float32Array` with three floats per point.
- `velocities`: flat velocity triplets for rendering or diagnostics.
- `properties`: four floats per particle for generation, age, scale, and random seed.
- `activeFlags`: compact liveness marker array.

This structure-of-arrays design favors tight loops, predictable memory access, and efficient partial rendering updates.

### Glow Material

The intended particle material is a `THREE.PointsMaterial` configured for luminous additive blending:

- `size: 0.75`
- `sizeAttenuation: true`
- `vertexColors: true`
- `transparent: true`
- `blending: THREE.AdditiveBlending`
- `depthWrite: false`

A procedural 64x64 radial glow texture can be generated with five optical stops: white core, golden ember, amber flame, red outer halo, and transparent feathered edge. This keeps the visual identity in the repository logic instead of depending on branded or watermarked image assets.

## Swarm Physics and Algorithms

### 1. Deterministic 3D Perlin Flow Field

The swarm uses an in-repo implementation of improved gradient Perlin noise. A deterministic permutation table is initialized once with a fixed seed, and each boid samples a continuous field over position and time.

Conceptually:

```text
theta = noise(x * flowScale, z * flowScale, t * flowSpeed) * 2π
verticalTheta = noise((x + offset) * flowScale, (z + offset) * flowScale, t * flowSpeed * 0.7) * 2π
```

The horizontal angle controls X/Z heading, while a second noise sample adds vertical motion. Per-boid sine and cosine wobble offsets keep the flock organic instead of perfectly laminar.

### 2. Damped Velocity Integration

Each boid updates velocity by blending previous velocity with acceleration from the flow field, pursuit, attack-state modifiers, and local separation. The integration uses damping to prevent unbounded oscillation:

```text
v = v * damping + acceleration * dt
```

A squared-magnitude speed cap avoids expensive square roots unless a velocity exceeds the maximum speed threshold.

### 3. Spatial Density Grid

The engine maintains a fixed-size 2D grid over the X/Z arena for density estimation. Boids contribute to `cellCounts`, and density pressure is used to reduce overcrowding. This approach is not a full all-neighbor boids implementation; it is an efficient O(N) approximation designed for large point counts where pairwise O(N²) neighborhood checks would be too expensive.

### 4. Target Pursuit and Segment Distribution

In flocking mode, a subset of boids can target distributed snake body segments to create surrounding pressure rather than collapsing every particle onto the head. During spear states, targeting concentrates toward the head to produce a readable lance formation and thrust.

### 5. Terrain Adherence

Boid height is constrained by terrain features. Raised terrain and carved areas influence cruising altitude so the swarm appears to skim, climb, or float around cubic obstacles instead of ignoring the playfield.

### 6. Self-Replication

Replication is configurable through:

- `minionReplicationInterval`
- `minionReplicationMultiplier`
- `maxPointsCap`

The default behavior supports rapid `1 -> 4` expansion: each replication cycle attempts to spawn three child boids per parent until the cap is reached. Children inherit part of the parent velocity and receive jittered positions, generation increments, and randomized spread impulses.

## Attack State Machine

The swarm uses a four-state tactical color and motion machine.

| State | Name | Behavior | Visual Intent |
|---|---|---|---|
| `0` | `FLOCKING` / Murmuration | Organic flow-field movement, distributed pressure, body-segment targeting | Velocity-mapped gold-to-crimson flock glow |
| `1` | `FORMING_SPEAR` | Particles organize into a tapered lance aligned toward the target | Cyan lead tip with magenta/violet body |
| `2` | `THRUSTING_SPEAR` | High-speed committed attack toward the snake | Plasma-white tip and hot-pink kinetic trail |
| `3` | `DISPERSING` / Impact Burst | Short radial burst after hit or timeout | Incandescent orange/amber explosion |

Transitions are timer- and distance-driven. Flocking escalates when the swarm has spent enough time near the player, spear formation advances into thrust, thrust resolves into dispersal, and dispersal returns to flocking after a short burst window.

## Snake Survivor Simulation

The red snake is the controlled survival agent. It has:

- Head position, velocity, heading, pitch, altitude, and vertical velocity.
- Body segment history for smooth trailing geometry.
- Continuous body length with passive growth and damage-driven shortening.
- Dash, repel, terrain barrier, and trench abilities.
- Health, cooldowns, invulnerability windows, and combat telemetry.
- Manual and RL-controlled movement modes.

The snake objective is survival and growth under swarm pressure. It must manage altitude, avoid spear attacks, time defensive abilities, exploit terrain, and preserve body length.

## Reinforcement Learning Architecture

The repository contains two RL systems serving different purposes.

1. **Continuous SAC-style controller**: the default modern controller for high-dimensional 3D movement and ability activation.
2. **Tabular Q-learning controller**: a compact, interpretable baseline for discrete tactical decisions.

Both controllers are integrated into `UrsinaRepelSimulation`, which computes observations, dispatches actions, applies rewards, and emits metrics.

## Soft Actor-Critic Continuous Controller

`ContinuousSACAgent` is the primary adaptive controller for the snake.

### Observation Space

The SAC controller uses a 126-dimensional normalized observation vector. It includes signals such as:

- Snake kinematics: position, velocity, heading, altitude, angular state, and speed.
- Local swarm perception: nearest enemies, predicted movement, local density, and threat geometry.
- Radar-style sector densities for spatial awareness.
- Ability state: repel, dash, barrier, trench cooldowns and availability.
- Defensive state: health, length, shielding, proximity to terrain, and recent hit information.
- Tactical context: spear state, winding/enclosure estimates, curriculum stage, and shaping features.

The exact vector is generated in the simulation layer so the agent receives world-state observations without coupling to React or Three.js.

### Action Space

The SAC action vector has seven dimensions:

```text
[ax, ay, az, a_repel, a_dash, a_barrier, a_trench]
```

- `ax`, `ay`, `az`: continuous acceleration / movement intent across 3D space.
- `a_repel`: continuous trigger intent for radial blast defense.
- `a_dash`: continuous trigger intent for escape or repositioning burst.
- `a_barrier`: continuous trigger intent for raising terrain defense.
- `a_trench`: continuous trigger intent for carving terrain.

The simulator smooths action vectors with exponential moving average behavior and action persistence so the policy does not jitter every frame.

### Neural Components

The in-browser SAC implementation includes:

- Dense MLP layers backed by typed arrays.
- LeakyReLU-style nonlinear processing.
- Optional layer normalization for stability.
- Actor network for continuous action production.
- Twin critic networks to reduce overestimation bias.
- Target critic networks updated with Polyak averaging.
- Adam-style optimizer state for weights, biases, and layer-norm parameters.
- Automatic entropy temperature tracking through `alpha` and target entropy.

### Replay and Bootstrapping

The replay system supports:

- Fixed-capacity transition storage.
- Prioritized Experience Replay (PER), controlled by `sacUsePER`, `sacPerAlpha`, and `sacPerBeta`.
- Importance sampling weights for biased replay correction.
- N-step returns, controlled by `sacUseNStep` and `sacNStep`.
- TD-error telemetry including mean and maximum error.

### Reward Design

Rewards combine survival, growth, tactical defense, and smooth control. Three named reward variants are exposed:

| Variant | Purpose |
|---|---|
| `variant_a_net_growth` | Emphasizes net length gain after accounting for damage and shrinkage. |
| `variant_b_max_size` | Rewards reaching and preserving large body size. |
| `variant_c_combined` | Balances growth, survival, tactical ability use, altitude, and smoothness. |

Additional terms may include:

- Length-scaled survival reward.
- Hit and death penalties.
- Reward for well-timed repel usage against dense clusters.
- Reward for dash escapes and altitude changes during spear pressure.
- Barrier and trench tactical bonuses.
- Smoothness penalty to discourage erratic control.
- Potential-Based Reward Shaping (PBRS): `F(s, s') = gamma * Phi(s') - Phi(s)`.
- Curriculum-stage adjustments for survival, tactical harvesting, and adversarial mastery.

### Emergent Tactic Telemetry

The SAC agent labels high-level tactical tendencies for UI diagnostics, including:

- `soar_spear_evade`
- `orbital_flank_peel`
- `bait_kinetic_blast`
- `trench_chokepoint_trap`
- `barrier_shield_deflect`
- `surge_dash_escape`
- `spiral_coil_defense`
- `volumetric_dive_bomb`
- `free_adaptive_cruise`

These labels are telemetry aids, not hardcoded state-machine actions. They help interpret what the continuous policy appears to be doing.

## Tabular Q-Learning Controller

`RedSurvivalRLAgent` provides a discrete comparison policy.

### Discrete State Encoding

State keys are built from six tactical dimensions:

| Dimension | Values |
|---|---|
| Proximity | `close`, `mid`, `far` |
| Local swarm density | `low`, `high`, `critical` |
| Repel cooldown | `ready`, `cooling` |
| Barrier coverage | `shielded`, `exposed` |
| Health | `critical`, `healthy` |
| Altitude | `ground`, `mid`, `aerial` |

This creates `3 * 3 * 2 * 2 * 2 * 3 = 216` prebootstrapped states.

### Action Set

The Q-agent chooses among:

- `evade_kite`
- `aerial_soar`
- `ground_dive`
- `altitudinal_corkscrew`
- `blast_repel`
- `carve_trench`
- `tactical_dash`
- `flank_reposition`

### Learning Rule

The update rule is standard Bellman Q-learning:

```text
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
```

Exploration uses epsilon-greedy action selection with exponential epsilon decay down to a configurable floor.

### Why Keep Q-Learning?

The tabular policy is useful because it is:

- Easy to inspect and debug.
- Fast to reset and compare.
- Suitable for teaching discrete RL concepts.
- A baseline for evaluating whether the continuous SAC controller is learning genuinely better 3D tactics.

## Terrain Deformation and Tactical Geometry

The terrain system models discrete cubic deformations on the X/Z arena.

### Terrain Records

Each deformation stores:

- Center point.
- Cube size, width, height, and depth.
- Base elevation and stack level.
- Type: small cube, big cube, or cuboid.
- Signed height change: positive for raised barriers, negative for carved trenches.
- Carved-state flag and timestamp.

### Tactical Uses

- **Raised barriers** can block line-of-attack and create shielded states for RL.
- **Trenches** can create chokepoints or escape lanes.
- **Stacked cubes** introduce vertical cover and force altitude-aware movement.
- **Terrain collision** can redirect or vault the snake.
- **Swarm adherence** makes boids respect elevated arena features visually.

## Rendering Pipeline

The renderer is hosted by `SimulationViewport` and uses Three.js scene objects synchronized from simulation state.

Important responsibilities include:

- Camera preset handling and follow modes.
- Swarm point-cloud geometry updates.
- Procedural snake mesh construction.
- Guardian/mannequin mesh rendering.
- Terrain cube and cuboid rendering.
- Lava pools, particles, shockwaves, floating damage numbers, health bars, trails, and HUD overlays.
- RL diagnostic cards for SAC or Q-learning mode.

The rendering layer should not become the source of truth for physics. Add state to simulation classes first, then expose it to the renderer.

## Audio and Feedback Systems

`src/audio/soundEffects.ts` provides procedural sound feedback for simulation events. Audio is controlled through the simulation configuration and can be disabled through the UI.

Feedback channels include:

- Combat hits.
- Repel blasts.
- Damage / decay events.
- Ability triggers.
- Swarm and simulation state cues.

## Project Structure

```text
.
├── README.md
├── AGENTS.md
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── src
    ├── App.tsx
    ├── main.tsx
    ├── index.css
    ├── types.ts
    ├── audio
    │   └── soundEffects.ts
    ├── components
    │   ├── Header.tsx
    │   ├── SimulationControls.tsx
    │   ├── SimulationViewport.tsx
    │   └── UrsinaCodeModal.tsx
    ├── rendering
    │   ├── mannequinFactory.ts
    │   ├── skyShader.ts
    │   ├── snakeFactory.ts
    │   └── snakeTailTrail.ts
    └── simulation
        ├── continuousSACAgent.ts
        ├── huntingRLAgent.ts
        ├── simulationEngine.ts
        └── swarmFieldEngine.ts
```

## Configuration Surface

Most gameplay, rendering, and learning behavior is controlled through `SimulationConfig`.

### Swarm Parameters

- `boidsEnabled`
- `playerAttractionWeight`
- `boidsAttackWeight`
- `boidsSeparationWeight`
- `boidsAlignmentWeight`
- `boidsCohesionWeight`
- `boidsSeparationRadius`
- `boidsNeighborRadius`
- `boidsMaxSpeed`
- `boidsMaxForce`
- `minionReplicationInterval`
- `minionReplicationMultiplier`
- `maxPointsCap`
- `flowFieldEnabled`
- `flowFieldWeight`
- `flowFieldScale`
- `flowFieldSpeed`
- `anchorClusterWeight`
- `individualWiggleWeight`
- `swarmColorMode`

### Snake Parameters

- `playerSpeed`
- `playerDashSpeed`
- `playerMaxHealth`
- `playerRepelRadius`
- `playerRepelCooldown`
- `playerDashCooldown`
- `playerAutoHealRate`
- `snakeInitialLength`
- `snakeMaxCap`
- `snakeElongateInterval`
- `snakeGrowthAmount`
- `snakeMinLength`
- `snakeShortenPerBite`
- `snakeSegmentSpacing`
- `enable3DFlight`
- `snakeMinAltitude`
- `snakeMaxAltitude`

### SAC Parameters

- `sacEnabled`
- `sacLearningRate`
- `sacDiscountFactor`
- `sacTau`
- `sacBatchSize`
- `sacReplayCapacity`
- `sacActionPersistence`
- `sacRewardVariant`
- `sacLengthScale`
- `sacAlphaWeight`
- `sacBetaWeight`
- `sacLambdaSmoothness`
- `sacNearestEnemiesCount`
- `sacPredictionHorizon`
- `sacDensityRadius`
- `sacIsEvaluation`
- `sacMaxSpeed`
- `sacRolloutSteps`
- `sacUsePER`
- `sacPerAlpha`
- `sacPerBeta`
- `sacUseNStep`
- `sacNStep`
- `sacUsePBRS`
- `sacUseLayerNorm`

### Q-Learning Parameters

- `rlEnabled`
- `rlLearningRate`
- `rlDiscountFactor`
- `rlEpsilon`
- `rlEpsilonDecay`
- `rlMinEpsilon`
- `rlDecisionInterval`
- `rlRewardSurvivalPerSec`
- `rlRewardGrowth`
- `rlRewardRepelKill`
- `rlRewardBarrierDeflect`
- `rlPenaltyShortened`
- `rlPenaltyDamage`
- `rlPenaltyWastedRepel`
- `rlPenaltyDeath`

### Visual and Environment Parameters

- `terrainDeformationEnabled`
- `terrainBrushMode`
- `terrainCubeSize`
- `terrainCubeHeight`
- `terrainCubeDepth`
- `terrainWireframe`
- `terrainGridResolution`
- `showSnakeTailTrail`
- `showRepelRadius`
- `showPanicZone`
- `showMinionTrails`
- `showHealthBars`
- `showDamageNumbers`
- `showQDecisionHUD`
- `arenaGridSize`
- `cameraPreset`
- `soundEnabled`
- `bloomIntensity`

## Controls

| Key | Action |
|---|---|
| `Space` | Play / pause simulation |
| `R` | Reset simulation |
| `E` | Manual repel blast |
| `S` | Manual spawn swarm cluster / minion |
| `F` | Trigger fusion where available |
| `T` / `X` | Trigger spear attack / respawn episode depending on context |
| `C` | Toggle code modal |
| `H` | Toggle banners / minimal view |
| `B` / `Tab` | Toggle sidebar |

Mouse and pointer controls are handled in the viewport and terrain tools. The sidebar exposes sliders, toggles, and buttons for live tuning.

## Getting Started

### Prerequisites

- Node.js 18 or newer.
- npm. Yarn or pnpm may work, but npm is the documented path because the repository includes npm scripts.

### Install

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

The Vite server listens on port `3000` and binds to `0.0.0.0` according to the project script.

### Build Production Bundle

```bash
npm run build
```

### Type Check

```bash
npm run lint
```

The `lint` script currently runs TypeScript in no-emit mode.

## Scripts

| Script | Command | Description |
|---|---|---|
| `dev` | `vite --port=3000 --host=0.0.0.0` | Start the local development server. |
| `build` | `vite build` | Create the production bundle. |
| `preview` | `vite preview` | Preview the production bundle locally. |
| `clean` | `rm -rf dist server.js` | Remove generated build artifacts. |
| `lint` | `tsc --noEmit` | Run the TypeScript checker without writing output files. |

## Development Notes

- Keep simulation logic in `src/simulation` and rendering synchronization in `src/components/SimulationViewport.tsx`.
- Prefer typed arrays and in-place updates in swarm hot paths.
- Avoid introducing per-boid Three.js meshes for large swarm particles; use point-cloud buffers instead.
- Do not wrap imports in `try/catch` blocks.
- When adding RL metrics, extend the relevant TypeScript interfaces first so the UI and simulation remain type-safe.
- When changing visual behavior, update the README if the public rendering model or color/state semantics change.
- The swarm should remain autonomous and independent from any summoner entity.

## Extension Guide

### Add a New Swarm Color Mode

1. Extend the `SwarmColorMode` type in `src/types.ts`.
2. Add UI selection support in `SimulationControls.tsx`.
3. Update color assignment logic in `SwarmFieldEngine.update()`.
4. Verify point colors update through the existing buffer geometry path.

### Add a New SAC Reward Term

1. Add configuration fields to `SimulationConfig` if the term must be tunable.
2. Compute the environment signal in `UrsinaRepelSimulation.stepSACAgent()` or the observation builder.
3. Add the reward component in `ContinuousSACAgent.calculateReward()`.
4. Expose telemetry in `SACMetrics` if it should be displayed or exported.
5. Test with both training and evaluation modes.

### Add a New Q-Learning Action

1. Extend `RLAction` in `src/types.ts`.
2. Add the action to `RL_ACTIONS`.
3. Initialize Q-values in `bootstrapAllStates()` and `ensureStateExists()`.
4. Implement action execution in the simulation engine.
5. Add HUD/control labels if the action should be visible to users.

### Add a New Terrain Tool

1. Extend `TerrainBrushMode` and `TerrainDeformationRecord` if needed.
2. Add UI controls in `SimulationControls.tsx`.
3. Implement deformation creation in the simulation engine.
4. Add rendering support in `SimulationViewport.tsx`.
5. Update collision, shielding, and swarm adherence if the tool changes navigation.

## License

MIT
