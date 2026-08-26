# Project System Directives & Swarm Architecture

## Swarm Appearance Mechanics & Visual Specification
The particle swarm system is governed by the following authoritative visual, rendering, and mathematical mechanics:

1. **Luminous 3D Particle Point Cloud Architecture (Zero-GC WebGL BufferGeometry)**:
   - **Render Primitive**: Rendered via `THREE.Points` utilizing dynamic `THREE.BufferGeometry` with interleaved/flat `Float32Array` buffers for coordinates (`position`: 3 floats per point) and RGB vertex colors (`color`: 3 floats per point), supporting up to 20,000+ real-time points with zero per-frame Garbage Collection (GC) overhead.
   - **Procedural Particle Glow Texture**: Generated dynamically using a 64x64 Canvas 2D radial gradient with five optical stops:
     - Center (`0.0`): Pure white luminous core (`rgba(255, 255, 255, 1.0)`)
     - Inner Ember (`0.2`): Bright warm golden yellow (`rgba(254, 240, 138, 0.95)`)
     - Mid Flame (`0.5`): Deep radiant amber (`rgba(245, 158, 11, 0.70)`)
     - Outer Halo (`0.8`): Soft incandescent fiery red (`rgba(239, 68, 68, 0.30)`)
     - Edge (`1.0`): Transparent feathering (`rgba(239, 68, 68, 0.0)`)
   - **Material Configuration**: `THREE.PointsMaterial` with `size: 0.75`, `sizeAttenuation: true`, `vertexColors: true`, `transparent: true`, `blending: THREE.AdditiveBlending`, and `depthWrite: false`.

2. **4-State Attack & Dynamic Color Transition Machine**:
   - **State 0 (FLOCKING / Murmuration)**:
     - Dynamic velocity-mapped color interpolation: High-velocity particles glow brilliant golden yellow (`RGB: [1.0, 1.0, 0.10]`), transitioning to fiery crimson (`RGB: [1.0, 0.25, 0.45]`) during low-velocity hovering and deceleration.
     - Natural sinusoidal micro-wobble offsets simulating organic avian murmuration and insectoid flocking behavior.
   - **State 1 (FORMING_SPEAR)**:
     - Lead lance tip (first 10–20 boids): Luminous electric cyan (`RGB: [0.20, 0.90, 1.00]`).
     - Spear shaft & tapered wedge body: Vibrant neon magenta / electric violet (`RGB: [0.95, 0.25, 0.85]`).
   - **State 2 (THRUSTING_SPEAR)**:
     - High-velocity piercing tip: Hyper-luminous plasma diamond white (`RGB: [1.00, 0.95, 1.00]`).
     - Trailing kinetic lance body: Blazing neon hot-pink (`RGB: [1.00, 0.15, 0.55]`).
   - **State 3 (DISPERSING / Impact Burst)**:
     - Incandescent explosion orange/amber (`RGB: [1.00, 0.45, 0.10]`) with radial kinetic impulse.

3. **Physics, 3D Perlin Flow Fields & Reynolds Boids Dynamics**:
   - **Flocking Rules**: Craig Reynolds Separation, Alignment, and Cohesion forces combined with an $O(1)$ spatial density grid.
   - **3D Perlin Flow Field**: Continuous volumetric vector field guiding organic trajectories:
     $\theta = \text{Noise}(x \cdot 0.002, z \cdot 0.002, t \cdot 0.10) \cdot 2\pi$
   - **Continuous High-Frequency Self-Replication**: Dynamic $1 \to 4$ splitting ($+3$ clones per cycle) occurring at user-controlled intervals ($0.01\text{s}$) up to the swarm capacity buffer.
   - **Terrain Adherence**: Point heights smoothly follow discrete deformable cubic terrain elevations with cruising altitude buoyancy.

4. **Summoner Independence**:
   - The swarm is completely autonomous and self-sustaining across the arena.
   - It does not depend on, spawn from, or require any summoner entity to exist, replicate, navigate, or attack.
