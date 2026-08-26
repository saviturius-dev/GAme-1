import { Vector3D, SwarmColorMode } from '../types';

/**
 * Fast 3D Gradient Hash Noise (Exact Ken Perlin improved noise)
 * Zero memory allocation, zero GC overhead.
 */
const PERM = new Uint8Array(512);
const GRAD3 = [
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
  1, 1, 0, 0, -1, 1, -1, 1, 0, 0, -1, -1,
];

// Initialize deterministic permutation table
(function initPerm() {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  let seed = 42;
  for (let i = 255; i > 0; i--) {
    seed = (seed * 16807) % 2147483647;
    const j = Math.floor((seed / 2147483647) * (i + 1));
    const tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }
  for (let i = 0; i < 512; i++) {
    PERM[i] = p[i & 255];
  }
})();

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

function grad3D(hash: number, x: number, y: number, z: number): number {
  const h = (hash & 15) * 3;
  return GRAD3[h] * x + GRAD3[h + 1] * y + GRAD3[h + 2] * z;
}

/**
 * Evaluates continuous 3D Perlin noise at (x, y, z) in O(1) time
 */
export function perlin3D(x: number, y: number, z: number): number {
  const X = Math.floor(x) & 255;
  const Y = Math.floor(y) & 255;
  const Z = Math.floor(z) & 255;

  const fx = x - Math.floor(x);
  const fy = y - Math.floor(y);
  const fz = z - Math.floor(z);

  const u = fade(fx);
  const v = fade(fy);
  const w = fade(fz);

  const A = PERM[X] + Y;
  const AA = PERM[A] + Z;
  const AB = PERM[A + 1] + Z;
  const B = PERM[X + 1] + Y;
  const BA = PERM[B] + Z;
  const BB = PERM[B + 1] + Z;

  return lerp(
    lerp(
      lerp(grad3D(PERM[AA], fx, fy, fz), grad3D(PERM[BA], fx - 1, fy, fz), u),
      lerp(grad3D(PERM[AB], fx, fy - 1, fz), grad3D(PERM[BB], fx - 1, fy - 1, fz), u),
      v
    ),
    lerp(
      lerp(grad3D(PERM[AA + 1], fx, fy, fz - 1), grad3D(PERM[BA + 1], fx - 1, fy, fz - 1), u),
      lerp(grad3D(PERM[AB + 1], fx, fy - 1, fz - 1), grad3D(PERM[BB + 1], fx - 1, fy - 1, fz - 1), u),
      v
    ),
    w
  );
}

/**
 * High-Performance Autonomous Swarm Field Engine (No Summoner, No Spear)
 * 
 * Rebuilt from scratch based on the official Boid Swarm Architecture:
 * - Struct-of-Arrays (SoA) TypedArray Data Layout (boidX, boidY, boidZ, boidVx, boidVy, boidVz, boidAge, boidGen, boidSeed).
 * - 3D Perlin Flow Field Generator (Ken Perlin Improved Noise).
 * - The 4 Core Equations:
 *     Eq 1: Continuous Flow Heading + Sinusoidal Individual Wiggle.
 *     Eq 2: Damped Velocity Integration (DAMPING = 0.92, ACCEL = 15.0).
 *     Eq 3: Spatial Grid Density Separation in O(1) Time (CELL_SIZE = 2.0, DENSITY_LIMIT = 6).
 *     Eq 4: Squared Magnitude Terminal Velocity Cap (MAX_SPEED = 6.5 m/s).
 * - Continuous High-Frequency Replication (1 -> 4 every 0.01s up to capacity cap).
 * - Zero-GC In-Place Compaction.
 * - Dynamic Luminous 3D Particle Point Cloud Buffers (Positions, Colors, Velocities, Properties).
 */
export class SwarmFieldEngine {
  public maxCapacity: number;
  public count: number = 0;

  // === STRUCT-OF-ARRAYS (SoA) DATA LAYOUT ===
  public boidX: Float32Array;
  public boidY: Float32Array;
  public boidZ: Float32Array;
  public boidVx: Float32Array;
  public boidVy: Float32Array;
  public boidVz: Float32Array;
  public boidAge: Float32Array;
  public boidGen: Float32Array;
  public boidSeed: Float32Array;
  public boidHealth: Float32Array;
  public boidMaxHealth: Float32Array;

  // Interleaved buffers for direct WebGL / Three.js BufferGeometry rendering
  public positions: Float32Array;
  public colors: Float32Array; // 3 floats per point (RGB)
  public velocities: Float32Array;
  public properties: Float32Array; // [generation, age, scale, seed] per boid
  public activeFlags: Uint8Array;

  // Spatial Grid Map for O(1) density checks
  public worldSize: number = 60.0;
  public cellSize: number = 2.0;
  public gridCols: number;
  public gridRows: number;
  public cellCounts: Uint16Array;

  // Swarm Telemetry & 4-State Attack Machine
  public swarmCentroid: Vector3D = { x: 0, y: 1.5, z: 0 };
  public swarmHeading: Vector3D = { x: 1, y: 0, z: 0 };
  public attackState: number = 0; // 0: FLOCKING, 1: FORMING_SPEAR, 2: THRUSTING_SPEAR, 3: DISPERSING
  public attackStateTimer: number = 0;

  // Survival Time & Demographics Feedback
  public avgSurvivalTime: number = 0;
  public maxSurvivalTime: number = 0;
  public oldestBoidAge: number = 0;
  public demographics = {
    newborn: 0,
    youth: 0,
    mature: 0,
    veteran: 0,
    ancient: 0,
  };

  private replicationTimer: number = 0;

  constructor(maxCapacity: number = 20000, worldSize: number = 60.0, cellSize: number = 2.0) {
    this.maxCapacity = maxCapacity;
    this.worldSize = worldSize;
    this.cellSize = cellSize;

    // Allocate SoA buffers
    this.boidX = new Float32Array(maxCapacity);
    this.boidY = new Float32Array(maxCapacity);
    this.boidZ = new Float32Array(maxCapacity);
    this.boidVx = new Float32Array(maxCapacity);
    this.boidVy = new Float32Array(maxCapacity);
    this.boidVz = new Float32Array(maxCapacity);
    this.boidAge = new Float32Array(maxCapacity);
    this.boidGen = new Float32Array(maxCapacity);
    this.boidSeed = new Float32Array(maxCapacity);
    this.boidHealth = new Float32Array(maxCapacity);
    this.boidMaxHealth = new Float32Array(maxCapacity);

    // Allocate WebGL rendering buffers
    this.positions = new Float32Array(maxCapacity * 3);
    this.colors = new Float32Array(maxCapacity * 3);
    this.velocities = new Float32Array(maxCapacity * 3);
    this.properties = new Float32Array(maxCapacity * 4);
    this.activeFlags = new Uint8Array(maxCapacity);

    // Initialize Spatial Grid
    this.gridCols = Math.ceil(this.worldSize / this.cellSize);
    this.gridRows = Math.ceil(this.worldSize / this.cellSize);
    this.cellCounts = new Uint16Array(this.gridCols * this.gridRows);
  }

  public reset() {
    this.count = 0;
    this.boidX.fill(0);
    this.boidY.fill(0);
    this.boidZ.fill(0);
    this.boidVx.fill(0);
    this.boidVy.fill(0);
    this.boidVz.fill(0);
    this.boidAge.fill(0);
    this.boidGen.fill(0);
    this.boidSeed.fill(0);
    this.boidHealth.fill(100);
    this.boidMaxHealth.fill(100);
    this.activeFlags.fill(0);
    this.positions.fill(0);
    this.colors.fill(0);
    this.velocities.fill(0);
    this.properties.fill(0);
    this.cellCounts.fill(0);
    this.replicationTimer = 0;
    this.swarmCentroid = { x: 0, y: 1.5, z: 0 };
    this.swarmHeading = { x: 1, y: 0, z: 0 };
  }

  /**
   * Spawns a point boid directly into the SoA pool in O(1)
   */
  public spawnPoint(
    x: number,
    y: number,
    z: number,
    vx: number = (Math.random() - 0.5) * 2.0,
    vy: number = (Math.random() - 0.5) * 1.0,
    vz: number = (Math.random() - 0.5) * 2.0,
    generation: number = 0,
    maxHealth: number = 100.0
  ): number {
    if (this.count >= this.maxCapacity) {
      return -1;
    }

    const idx = this.count;
    const p3 = idx * 3;
    const p4 = idx * 4;

    const clampedY = Math.max(0.35, y);
    const seed = Math.random() * 1000;

    this.boidX[idx] = x;
    this.boidY[idx] = clampedY;
    this.boidZ[idx] = z;

    this.boidVx[idx] = vx;
    this.boidVy[idx] = vy;
    this.boidVz[idx] = vz;

    this.boidAge[idx] = 0.0;
    this.boidGen[idx] = generation;
    this.boidSeed[idx] = seed;
    this.boidHealth[idx] = maxHealth;
    this.boidMaxHealth[idx] = maxHealth;

    this.positions[p3 + 0] = x;
    this.positions[p3 + 1] = clampedY;
    this.positions[p3 + 2] = z;

    // Golden yellow core [1.0, 0.95, 0.2]
    this.colors[p3 + 0] = 1.0;
    this.colors[p3 + 1] = 0.95;
    this.colors[p3 + 2] = 0.2;

    this.velocities[p3 + 0] = vx;
    this.velocities[p3 + 1] = vy;
    this.velocities[p3 + 2] = vz;

    this.properties[p4 + 0] = generation;
    this.properties[p4 + 1] = 0.0; // Age
    this.properties[p4 + 2] = 0.45 + Math.min(generation * 0.04, 0.35); // Scale
    this.properties[p4 + 3] = seed;

    this.activeFlags[idx] = 1;
    this.count++;

    return idx;
  }

  /**
   * Spawn multiple boids in a cluster
   */
  public spawnCluster(
    count: number,
    centerX: number,
    centerY: number,
    centerZ: number,
    spread: number = 3.5
  ): number {
    const toSpawn = Math.min(count, this.maxCapacity - this.count);
    for (let i = 0; i < toSpawn; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * spread;
      const x = centerX + Math.cos(angle) * dist;
      const y = Math.max(0.5, centerY + (Math.random() - 0.5) * 1.5);
      const z = centerZ + Math.sin(angle) * dist;

      this.spawnPoint(
        x,
        y,
        z,
        (Math.random() - 0.5) * 3.0,
        (Math.random() - 0.5) * 1.0,
        (Math.random() - 0.5) * 3.0,
        0
      );
    }
    return toSpawn;
  }

  /**
   * Continuous High-Frequency Replication (Step 6 from Tutorial)
   * Multiplies boids every interval up to capacity
   */
  public stepReplication(
    dt: number,
    interval: number = 0.01,
    multiplier: number = 4,
    cap: number = 4000
  ): number {
    this.replicationTimer += dt;
    if (this.replicationTimer < interval || this.count >= cap || this.count === 0) {
      return 0;
    }

    this.replicationTimer = 0;
    const initialCount = this.count;
    const childrenPerParent = Math.max(1, multiplier - 1); // e.g. 3 children for 1 -> 4
    let spawnedTotal = 0;

    for (let i = 0; i < initialCount; i++) {
      if (this.count >= cap) break;

      const px = this.boidX[i];
      const py = this.boidY[i];
      const pz = this.boidZ[i];
      const pvx = this.boidVx[i];
      const pvy = this.boidVy[i];
      const pvz = this.boidVz[i];
      const gen = this.boidGen[i] + 1;

      for (let k = 0; k < childrenPerParent; k++) {
        if (this.count >= cap) break;

        const jitter = 0.45;
        const nx = px + (Math.random() - 0.5) * jitter;
        const ny = Math.max(0.35, py + (Math.random() - 0.5) * jitter);
        const nz = pz + (Math.random() - 0.5) * jitter;

        const angle = Math.random() * Math.PI * 2;
        const nvx = pvx * 0.5 + Math.cos(angle) * 2.0;
        const nvy = pvy * 0.5 + (Math.random() - 0.5) * 0.8;
        const nvz = pvz * 0.5 + Math.sin(angle) * 2.0;

        if (this.spawnPoint(nx, ny, nz, nvx, nvy, nvz, gen) !== -1) {
          spawnedTotal++;
        }
      }
    }

    return spawnedTotal;
  }

  /**
   * Main Simulation Update Loop — The 4 Core Equations (Step 5 from Tutorial)
   * Enhanced with Full 3D Snake Targeting & Unhindered Full-Altitude Traversal
   */
  public update(
    dt: number,
    simTime: number,
    targetX: number,
    targetY: number,
    targetZ: number,
    barriers: { point: Vector3D; cubeSize: number; heightChange: number }[] = [],
    config: {
      flowFieldEnabled?: boolean;
      flowFieldWeight?: number;
      flowFieldScale?: number;
      flowFieldSpeed?: number;
      playerAttractionWeight?: number;
      boidsAttackWeight?: number;
      boidsSeparationWeight?: number;
      boidsAlignmentWeight?: number;
      boidsCohesionWeight?: number;
      boidsMaxSpeed?: number;
      boidsMaxForce?: number;
      swarmColorMode?: SwarmColorMode;
      swarmAgeMaxThreshold?: number;
      swarmDynamicSizeByAge?: boolean;
    } = {},
    targetSegments?: { x: number; y: number; z: number }[]
  ) {
    const activeCount = this.count;
    if (activeCount === 0) return;

    // Prevent spiral of death on lag spikes
    const dtClamped = Math.min(dt, 0.05);

    // Flow field noise frequency parameters
    const spatialFreq = config.flowFieldScale ?? 0.002;
    const timeFreq = config.flowFieldSpeed ?? 0.10;

    // Physics parameters from tutorial
    const DAMPING = 0.92;
    const ACCEL = (config.flowFieldWeight ?? 2.6) * 5.8; // default ~15.0
    const MAX_SPEED = config.boidsMaxSpeed ?? 8.5;
    const MAX_SPEED_SQ = MAX_SPEED * MAX_SPEED;
    const CELL_SIZE = this.cellSize;
    const DENSITY_LIMIT = 6;
    const ATTACK_WEIGHT = config.playerAttractionWeight ?? config.boidsAttackWeight ?? 2.2;

    const colorMode: SwarmColorMode = config.swarmColorMode || 'survival_age';
    const ageMaxThreshold = Math.max(5.0, config.swarmAgeMaxThreshold || 20.0);
    const dynamicSizeByAge = config.swarmDynamicSizeByAge !== false;

    const HALF_WORLD = this.worldSize * 0.5 - 0.5;

    // Reset spatial grid density buffer
    this.cellCounts.fill(0);

    // Centroid and demographic accumulators
    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    let sumVx = 0;
    let sumVy = 0;
    let sumVz = 0;

    let totalAge = 0;
    let maxAgeFound = 0;
    let dNewborn = 0;
    let dYouth = 0;
    let dMature = 0;
    let dVeteran = 0;
    let dAncient = 0;

    let writeIdx = 0;

    // Update 4-State Attack Machine & state transitions
    this.attackStateTimer += dtClamped;
    const targetDist = Math.hypot(
      targetX - this.swarmCentroid.x,
      (targetY ?? 0.55) - this.swarmCentroid.y,
      targetZ - this.swarmCentroid.z
    );

    if (this.attackState === 3) {
      // State 3: DISPERSING / Impact Burst (lasts ~0.8s then resets to flocking)
      if (this.attackStateTimer >= 0.8) {
        this.attackState = 0;
        this.attackStateTimer = 0;
      }
    } else if (this.attackState === 2) {
      // State 2: THRUSTING_SPEAR
      if (targetDist < 2.5 || this.attackStateTimer >= 2.0) {
        this.attackState = 3; // Impact Burst / Disperse
        this.attackStateTimer = 0;
      }
    } else if (this.attackState === 1) {
      // State 1: FORMING_SPEAR
      if (this.attackStateTimer >= 1.5 || targetDist < 8.0) {
        this.attackState = 2; // Thrust Spear
        this.attackStateTimer = 0;
      }
    } else {
      // State 0: FLOCKING / Murmuration
      if (this.attackStateTimer >= 4.0 && targetDist < 22.0) {
        this.attackState = 1; // Form Spear
        this.attackStateTimer = 0;
      }
    }

    for (let i = 0; i < activeCount; i++) {
      let x = this.boidX[i];
      let y = this.boidY[i];
      let z = this.boidZ[i];
      let vx = this.boidVx[i];
      let vy = this.boidVy[i];
      let vz = this.boidVz[i];
      let age = this.boidAge[i] + dtClamped;
      const gen = this.boidGen[i];
      const seed = this.boidSeed[i];

      // === EQUATION 1: Continuous 3D Flow field angle from Ken Perlin noise ===
      let angle = 0;
      let vAngle = 0;
      if (config.flowFieldEnabled !== false) {
        angle = perlin3D(
          x * spatialFreq,
          z * spatialFreq,
          simTime * timeFreq
        ) * Math.PI * 4.0;

        vAngle = perlin3D(
          (x + 100) * spatialFreq,
          (z + 100) * spatialFreq,
          simTime * timeFreq * 0.7
        ) * Math.PI * 2.0;
      }

      // Per-boid sinusoidal micro-wobble for natural flock murmuration
      const wiggle = Math.sin(simTime * 3.0 + seed) * 0.35;
      const vWiggle = Math.cos(simTime * 2.2 + seed * 1.3) * 0.4;

      // === EQUATION 2: Damped velocity integration across all 3 spatial axes ===
      vx = (vx * DAMPING) + Math.cos(angle + wiggle) * ACCEL * dtClamped;
      vz = (vz * DAMPING) + Math.sin(angle + wiggle) * ACCEL * dtClamped;
      vy = (vy * DAMPING) + Math.sin(vAngle + vWiggle) * (ACCEL * 0.5) * dtClamped;

      // === FULL 3D TARGET PURSUIT (Snake Head & Dynamic Body Intercept) ===
      if (ATTACK_WEIGHT > 0) {
        // Target selection: in spear formation, focus on head; in flocking, distribute across segments
        let tX = targetX;
        let tY = targetY ?? 0.55;
        let tZ = targetZ;

        if (this.attackState === 0 && targetSegments && targetSegments.length > 0 && (i % 3 === 0)) {
          const segIdx = (i * 7) % targetSegments.length;
          const seg = targetSegments[segIdx];
          if (seg) {
            tX = seg.x;
            tY = seg.y ?? 0.55;
            tZ = seg.z;
          }
        }

        const dxA = tX - x;
        const dyA = tY - y;
        const dzA = tZ - z;
        const distA = Math.sqrt(dxA * dxA + dyA * dyA + dzA * dzA) || 0.001;
        const invDistA = 1.0 / distA;

        // Dynamic 3D pursuit force targeting the snake precisely across all altitudes
        const thrustMultiplier = this.attackState === 2 ? 2.2 : (this.attackState === 1 ? 1.4 : 1.0);
        const pursuitPower = ATTACK_WEIGHT * 4.2 * thrustMultiplier;
        vx += dxA * invDistA * pursuitPower * dtClamped;
        vy += dyA * invDistA * (pursuitPower * 1.6) * dtClamped;
        vz += dzA * invDistA * pursuitPower * dtClamped;
      }

      // === EQUATION 3: Spatial grid density check (O(1) separation) ===
      const cX = Math.min(Math.max(Math.floor((x + HALF_WORLD) / CELL_SIZE), 0), this.gridCols - 1);
      const cZ = Math.min(Math.max(Math.floor((z + HALF_WORLD) / CELL_SIZE), 0), this.gridRows - 1);
      const cellIdx = cX + cZ * this.gridCols;

      this.cellCounts[cellIdx]++;
      if (this.cellCounts[cellIdx] > DENSITY_LIMIT) {
        // Too crowded — random 3D micro-jitter to prevent clumping
        vx += (Math.random() - 0.5) * 1.5;
        vy += (Math.random() - 0.5) * 1.0;
        vz += (Math.random() - 0.5) * 1.5;
      }

      // === Terrain Elevation Adherence (Smoothly float above cubic terrain features) ===
      let groundElevation = 0.25;
      if (y < 8.0 && barriers.length > 0) {
        for (let b = 0; b < barriers.length; b++) {
          const bar = barriers[b];
          const halfSize = (bar.cubeSize || 2.0) * 0.6;
          if (Math.abs(x - bar.point.x) < halfSize && Math.abs(z - bar.point.z) < halfSize) {
            if (bar.heightChange > 0) {
              groundElevation = Math.max(groundElevation, bar.heightChange + 0.35);
            }
          }
        }
      }

      // === EQUATION 4: Terminal velocity cap in 3D ===
      const maxSpdCurrent = this.attackState === 2 ? MAX_SPEED * 1.3 : MAX_SPEED;
      const maxSpdCurrentSq = maxSpdCurrent * maxSpdCurrent;
      const speedSq = vx * vx + vy * vy + vz * vz;
      if (speedSq > maxSpdCurrentSq) {
        const scale = maxSpdCurrent / Math.sqrt(speedSq);
        vx *= scale;
        vy *= scale;
        vz *= scale;
      }

      // === Position integration (Equal agile velocity across X, Y, and Z) ===
      x += vx * dtClamped * 4.0;
      y += vy * dtClamped * 4.0;
      z += vz * dtClamped * 4.0;

      // === Arena wrapping / Soft Boundary Containment ===
      if (x < -HALF_WORLD) x = HALF_WORLD - 0.5;
      if (x > HALF_WORLD) x = -HALF_WORLD + 0.5;
      if (z < -HALF_WORLD) z = HALF_WORLD - 0.5;
      if (z > HALF_WORLD) z = -HALF_WORLD + 0.5;

      // === Full-Altitude 3D Traversal with Terrain Elevation Float ===
      if (y < groundElevation) {
        y = groundElevation;
        vy = Math.max(0.5, vy * 0.5 + 1.2);
      } else if (y > 45.0) {
        // Gentle downward aerodynamic guidance at extreme sky heights
        y = 45.0;
        vy = -Math.abs(vy) * 0.5;
      }

      // Accumulate for swarm centroid
      sumX += x;
      sumY += y;
      sumZ += z;
      sumVx += vx;
      sumVy += vy;
      sumVz += vz;

      // === In-Place Compaction & Buffer Sync (Zero GC) ===
      this.boidX[writeIdx] = x;
      this.boidY[writeIdx] = y;
      this.boidZ[writeIdx] = z;
      this.boidVx[writeIdx] = vx;
      this.boidVy[writeIdx] = vy;
      this.boidVz[writeIdx] = vz;
      this.boidAge[writeIdx] = age;
      this.boidGen[writeIdx] = gen;
      this.boidSeed[writeIdx] = seed;

      // Update WebGL rendering buffers
      const p3 = writeIdx * 3;
      const p4 = writeIdx * 4;

      this.positions[p3 + 0] = x;
      this.positions[p3 + 1] = y;
      this.positions[p3 + 2] = z;

      this.velocities[p3 + 0] = vx;
      this.velocities[p3 + 1] = vy;
      this.velocities[p3 + 2] = vz;

      // Demographic tracking
      totalAge += age;
      if (age > maxAgeFound) maxAgeFound = age;
      if (age < 2.0) dNewborn++;
      else if (age < 6.0) dYouth++;
      else if (age < 15.0) dMature++;
      else if (age < 30.0) dVeteran++;
      else dAncient++;

      // === VISUAL FEEDBACK & COLORING ENGINE ===
      const curSpeed = Math.sqrt(speedSq);
      const speedRatio = Math.min(1.0, Math.max(0.0, curSpeed / (MAX_SPEED * 0.8)));
      const normAge = Math.min(1.0, Math.max(0.0, age / ageMaxThreshold));

      if (colorMode === 'survival_age') {
        // === SURVIVAL TIME / LONGEVITY LIFECYCLE GRADIENT ===
        // Newborn (Cyan/Mint) -> Youth (Gold) -> Mature (Crimson Flame) -> Veteran (Plasma Violet) -> Ancient (Diamond White)
        if (normAge < 0.15) {
          // 0 - 3s: Newborn Electric Mint/Cyan [0.15, 0.95, 1.00] -> Lime Green [0.35, 1.00, 0.60]
          const t = normAge / 0.15;
          this.colors[p3 + 0] = lerp(0.15, 0.35, t);
          this.colors[p3 + 1] = lerp(0.95, 1.00, t);
          this.colors[p3 + 2] = lerp(1.00, 0.60, t);
        } else if (normAge < 0.40) {
          // 3 - 8s: Lime Green -> Radiant Solar Gold [1.00, 0.90, 0.15]
          const t = (normAge - 0.15) / 0.25;
          this.colors[p3 + 0] = lerp(0.35, 1.00, t);
          this.colors[p3 + 1] = lerp(1.00, 0.90, t);
          this.colors[p3 + 2] = lerp(0.60, 0.15, t);
        } else if (normAge < 0.70) {
          // 8 - 14s: Solar Gold -> Incandescent Crimson Flame [1.00, 0.22, 0.30]
          const t = (normAge - 0.40) / 0.30;
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = lerp(0.90, 0.22, t);
          this.colors[p3 + 2] = lerp(0.15, 0.30, t);
        } else if (normAge < 0.90) {
          // 14 - 18s: Crimson Flame -> Deep Luminous Plasma Violet [0.85, 0.15, 0.95]
          const t = (normAge - 0.70) / 0.20;
          this.colors[p3 + 0] = lerp(1.00, 0.85, t);
          this.colors[p3 + 1] = lerp(0.22, 0.15, t);
          this.colors[p3 + 2] = lerp(0.30, 0.95, t);
        } else {
          // 18s+: Ancient Celestial Diamond White [1.00, 0.95, 1.00]
          const t = (normAge - 0.90) / 0.10;
          this.colors[p3 + 0] = lerp(0.85, 1.00, t);
          this.colors[p3 + 1] = lerp(0.15, 0.95, t);
          this.colors[p3 + 2] = lerp(0.95, 1.00, t);
        }
      } else if (colorMode === 'active_state') {
        // === DISCRETE ACTIVE STATE COLOR FEEDBACK ===
        if (this.attackState === 3) {
          // State 3 (DISPERSING): Incandescent Nova Orange
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = 0.42;
          this.colors[p3 + 2] = 0.08;
        } else if (this.attackState === 2) {
          // State 2 (THRUSTING_SPEAR): Diamond White Tip / Blazing Hot-Pink Lance
          if (writeIdx < 20) {
            this.colors[p3 + 0] = 1.00;
            this.colors[p3 + 1] = 0.98;
            this.colors[p3 + 2] = 1.00;
          } else {
            this.colors[p3 + 0] = 1.00;
            this.colors[p3 + 1] = 0.12;
            this.colors[p3 + 2] = 0.52;
          }
        } else if (this.attackState === 1) {
          // State 1 (FORMING_SPEAR): Electric Cyan Tip / Neon Violet Lance
          if (writeIdx < 20) {
            this.colors[p3 + 0] = 0.15;
            this.colors[p3 + 1] = 0.90;
            this.colors[p3 + 2] = 1.00;
          } else {
            this.colors[p3 + 0] = 0.92;
            this.colors[p3 + 1] = 0.22;
            this.colors[p3 + 2] = 0.88;
          }
        } else {
          // State 0 (FLOCKING): Radiant Solar Amber Gold
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = 0.84;
          this.colors[p3 + 2] = 0.12;
        }
      } else if (colorMode === 'hybrid_age_state') {
        // === HYBRID: ACTIVE FORMATION STATE MODULATED BY SURVIVAL TIME ===
        let baseR = 1.0;
        let baseG = 0.84;
        let baseB = 0.12;

        if (this.attackState === 3) {
          baseR = 1.00; baseG = 0.42; baseB = 0.08;
        } else if (this.attackState === 2) {
          if (writeIdx < 20) { baseR = 1.00; baseG = 0.98; baseB = 1.00; }
          else { baseR = 1.00; baseG = 0.12; baseB = 0.52; }
        } else if (this.attackState === 1) {
          if (writeIdx < 20) { baseR = 0.15; baseG = 0.90; baseB = 1.00; }
          else { baseR = 0.92; baseG = 0.22; baseB = 0.88; }
        } else {
          baseR = 1.00;
          baseG = lerp(0.35, 0.95, speedRatio);
          baseB = lerp(0.45, 0.12, speedRatio);
        }

        // Modulate with age: elder boids gain diamond plasma glow, newborn boids gain cyan tinge
        if (normAge > 0.6) {
          const elderWeight = (normAge - 0.6) / 0.4;
          this.colors[p3 + 0] = lerp(baseR, 1.00, elderWeight * 0.7);
          this.colors[p3 + 1] = lerp(baseG, 0.92, elderWeight * 0.7);
          this.colors[p3 + 2] = lerp(baseB, 1.00, elderWeight * 0.7);
        } else if (normAge < 0.12) {
          const newWeight = (0.12 - normAge) / 0.12;
          this.colors[p3 + 0] = lerp(baseR, 0.20, newWeight * 0.6);
          this.colors[p3 + 1] = lerp(baseG, 0.95, newWeight * 0.6);
          this.colors[p3 + 2] = lerp(baseB, 1.00, newWeight * 0.6);
        } else {
          this.colors[p3 + 0] = baseR;
          this.colors[p3 + 1] = baseG;
          this.colors[p3 + 2] = baseB;
        }
      } else if (colorMode === 'generation_lineage') {
        // === GENERATION LINEAGE GRADIENT ===
        if (gen === 0) {
          // Primordial Ancestors: Solar Gold
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = 0.88;
          this.colors[p3 + 2] = 0.15;
        } else if (gen === 1) {
          // 1st Generation: Emerald Lime
          this.colors[p3 + 0] = 0.25;
          this.colors[p3 + 1] = 0.95;
          this.colors[p3 + 2] = 0.45;
        } else if (gen === 2) {
          // 2nd Generation: Electric Cyan
          this.colors[p3 + 0] = 0.18;
          this.colors[p3 + 1] = 0.88;
          this.colors[p3 + 2] = 1.00;
        } else if (gen === 3) {
          // 3rd Generation: Radiant Indigo
          this.colors[p3 + 0] = 0.65;
          this.colors[p3 + 1] = 0.30;
          this.colors[p3 + 2] = 1.00;
        } else {
          // 4th+ Generation: Neon Hyper-Magenta
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = 0.15;
          this.colors[p3 + 2] = 0.75;
        }
      } else if (colorMode === 'kinetic_energy') {
        // === KINETIC SPEED & ENERGY HEATMAP ===
        if (speedRatio < 0.4) {
          const t = speedRatio / 0.4;
          this.colors[p3 + 0] = lerp(0.18, 0.30, t);
          this.colors[p3 + 1] = lerp(0.45, 0.95, t);
          this.colors[p3 + 2] = lerp(0.95, 0.85, t);
        } else if (speedRatio < 0.8) {
          const t = (speedRatio - 0.4) / 0.4;
          this.colors[p3 + 0] = lerp(0.30, 1.00, t);
          this.colors[p3 + 1] = lerp(0.95, 0.85, t);
          this.colors[p3 + 2] = lerp(0.85, 0.15, t);
        } else {
          const t = (speedRatio - 0.8) / 0.2;
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = lerp(0.85, 0.98, t);
          this.colors[p3 + 2] = lerp(0.15, 0.95, t);
        }
      } else {
        // === STATE_DYNAMIC (Default AGENTS.md Velocity & Spear Machine) ===
        if (this.attackState === 3) {
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = 0.45;
          this.colors[p3 + 2] = 0.10;
        } else if (this.attackState === 2) {
          if (writeIdx < 20) {
            this.colors[p3 + 0] = 1.00;
            this.colors[p3 + 1] = 0.95;
            this.colors[p3 + 2] = 1.00;
          } else {
            this.colors[p3 + 0] = 1.00;
            this.colors[p3 + 1] = 0.15;
            this.colors[p3 + 2] = 0.55;
          }
        } else if (this.attackState === 1) {
          if (writeIdx < 20) {
            this.colors[p3 + 0] = 0.20;
            this.colors[p3 + 1] = 0.90;
            this.colors[p3 + 2] = 1.00;
          } else {
            this.colors[p3 + 0] = 0.95;
            this.colors[p3 + 1] = 0.25;
            this.colors[p3 + 2] = 0.85;
          }
        } else {
          this.colors[p3 + 0] = 1.00;
          this.colors[p3 + 1] = lerp(0.25, 1.00, speedRatio);
          this.colors[p3 + 2] = lerp(0.45, 0.10, speedRatio);
        }
      }

      // Dynamic particle scale factoring in generation and survival age
      const ageSizeBonus = dynamicSizeByAge ? Math.min(normAge * 0.35, 0.35) : 0;
      this.properties[p4 + 0] = gen;
      this.properties[p4 + 1] = age;
      this.properties[p4 + 2] = 0.45 + Math.min(gen * 0.03, 0.25) + ageSizeBonus;
      this.properties[p4 + 3] = seed;

      this.activeFlags[writeIdx] = 1;
      writeIdx++;
    }

    this.count = writeIdx;

    if (this.count > 0) {
      const invCount = 1.0 / this.count;
      this.swarmCentroid.x = sumX * invCount;
      this.swarmCentroid.y = sumY * invCount;
      this.swarmCentroid.z = sumZ * invCount;

      const avgSpeed = Math.hypot(sumVx, sumVz) || 1.0;
      this.swarmHeading.x = (sumVx * invCount) / avgSpeed;
      this.swarmHeading.y = (sumVy * invCount) / avgSpeed;
      this.swarmHeading.z = (sumVz * invCount) / avgSpeed;

      this.avgSurvivalTime = totalAge * invCount;
      this.maxSurvivalTime = maxAgeFound;
      this.oldestBoidAge = maxAgeFound;
      this.demographics.newborn = dNewborn;
      this.demographics.youth = dYouth;
      this.demographics.mature = dMature;
      this.demographics.veteran = dVeteran;
      this.demographics.ancient = dAncient;
    }
  }

  /**
   * Apply Kinetic Repel Shockwave Blast to Swarm
   * Deals kinetic force and damage. Boids reaching 0 health trigger onBoidDeath explosion callback.
   */
  public applyRepelBlast(
    centerX: number,
    centerY: number,
    centerZ: number,
    radius: number,
    force: number = 28.0,
    damage: number = 100.0,
    onBoidDeath?: (x: number, y: number, z: number, vx: number, vy: number, vz: number) => void
  ): number {
    let repelledCount = 0;
    const radSq = radius * radius;

    for (let i = this.count - 1; i >= 0; i--) {
      const dx = this.boidX[i] - centerX;
      const dy = this.boidY[i] - centerY;
      const dz = this.boidZ[i] - centerZ;
      const distSq = dx * dx + dz * dz;

      if (distSq < radSq) {
        const dist = Math.sqrt(distSq) || 0.1;
        const falloff = 1.0 - dist / radius;
        const impulse = force * (0.6 + falloff * 0.8);

        this.boidVx[i] = (dx / dist) * impulse;
        this.boidVy[i] = Math.max(4.0, (dy / dist) * impulse * 0.6 + 3.0);
        this.boidVz[i] = (dz / dist) * impulse;

        // Flash luminous white on kinetic impact
        const p3 = i * 3;
        this.colors[p3 + 0] = 1.0;
        this.colors[p3 + 1] = 1.0;
        this.colors[p3 + 2] = 1.0;

        repelledCount++;

        if (damage > 0) {
          this.boidHealth[i] -= damage * (0.5 + falloff * 0.5);
          if (this.boidHealth[i] <= 0) {
            if (onBoidDeath) {
              onBoidDeath(
                this.boidX[i],
                this.boidY[i],
                this.boidZ[i],
                this.boidVx[i],
                this.boidVy[i],
                this.boidVz[i]
              );
            }
            this.removeBoid(i);
          }
        }
      }
    }

    if (repelledCount > 0) {
      this.attackState = 3; // DISPERSING / Impact Burst
      this.attackStateTimer = 0;
    }

    return repelledCount;
  }

  /**
   * Applies damage to a specific boid index. If health drops to <= 0, triggers onBoidDeath callback and removes boid.
   */
  public damageBoid(
    index: number,
    damage: number,
    onBoidDeath?: (x: number, y: number, z: number, vx: number, vy: number, vz: number) => void
  ): boolean {
    if (index < 0 || index >= this.count) return false;
    this.boidHealth[index] -= damage;
    if (this.boidHealth[index] <= 0) {
      if (onBoidDeath) {
        onBoidDeath(
          this.boidX[index],
          this.boidY[index],
          this.boidZ[index],
          this.boidVx[index],
          this.boidVy[index],
          this.boidVz[index]
        );
      }
      this.removeBoid(index);
      return true;
    }
    return false;
  }

  /**
   * Apply direct damage to all boids within 3D sphere.
   * Any boid whose health reaches zero dies and triggers onBoidDeath callback.
   */
  public damageBoidsInRadius(
    centerX: number,
    centerY: number,
    centerZ: number,
    radius: number,
    damage: number,
    onBoidDeath?: (x: number, y: number, z: number, vx: number, vy: number, vz: number) => void
  ): number {
    let killedCount = 0;
    const radSq = radius * radius;

    for (let i = this.count - 1; i >= 0; i--) {
      const dx = this.boidX[i] - centerX;
      const dy = this.boidY[i] - centerY;
      const dz = this.boidZ[i] - centerZ;
      const distSq = dx * dx + dy * dy + dz * dz;

      if (distSq < radSq) {
        this.boidHealth[i] -= damage;
        if (this.boidHealth[i] <= 0) {
          if (onBoidDeath) {
            onBoidDeath(
              this.boidX[i],
              this.boidY[i],
              this.boidZ[i],
              this.boidVx[i],
              this.boidVy[i],
              this.boidVz[i]
            );
          }
          this.removeBoid(i);
          killedCount++;
        }
      }
    }

    return killedCount;
  }

  /**
   * Eliminate boids within radius
   */
  public eliminateBoidsInRadius(
    centerX: number,
    centerZ: number,
    radius: number,
    maxEliminate: number = 20,
    onBoidDeath?: (x: number, y: number, z: number, vx: number, vy: number, vz: number) => void
  ): number {
    let eliminated = 0;
    const radSq = radius * radius;

    for (let i = this.count - 1; i >= 0; i--) {
      const dx = this.boidX[i] - centerX;
      const dz = this.boidZ[i] - centerZ;
      if (dx * dx + dz * dz < radSq) {
        if (onBoidDeath) {
          onBoidDeath(
            this.boidX[i],
            this.boidY[i],
            this.boidZ[i],
            this.boidVx[i],
            this.boidVy[i],
            this.boidVz[i]
          );
        }
        this.removeBoid(i);
        eliminated++;
        if (eliminated >= maxEliminate) break;
      }
    }

    return eliminated;
  }

  /**
   * Fast O(1) swap-and-pop boid removal
   */
  public removeBoid(index: number) {
    if (index < 0 || index >= this.count) return;
    const lastIdx = this.count - 1;

    if (index !== lastIdx) {
      this.boidX[index] = this.boidX[lastIdx];
      this.boidY[index] = this.boidY[lastIdx];
      this.boidZ[index] = this.boidZ[lastIdx];
      this.boidVx[index] = this.boidVx[lastIdx];
      this.boidVy[index] = this.boidVy[lastIdx];
      this.boidVz[index] = this.boidVz[lastIdx];
      this.boidAge[index] = this.boidAge[lastIdx];
      this.boidGen[index] = this.boidGen[lastIdx];
      this.boidSeed[index] = this.boidSeed[lastIdx];
      this.boidHealth[index] = this.boidHealth[lastIdx];
      this.boidMaxHealth[index] = this.boidMaxHealth[lastIdx];

      const p3 = index * 3;
      const lp3 = lastIdx * 3;
      this.positions[p3 + 0] = this.positions[lp3 + 0];
      this.positions[p3 + 1] = this.positions[lp3 + 1];
      this.positions[p3 + 2] = this.positions[lp3 + 2];

      this.colors[p3 + 0] = this.colors[lp3 + 0];
      this.colors[p3 + 1] = this.colors[lp3 + 1];
      this.colors[p3 + 2] = this.colors[lp3 + 2];

      this.velocities[p3 + 0] = this.velocities[lp3 + 0];
      this.velocities[p3 + 1] = this.velocities[lp3 + 1];
      this.velocities[p3 + 2] = this.velocities[lp3 + 2];

      const p4 = index * 4;
      const lp4 = lastIdx * 4;
      this.properties[p4 + 0] = this.properties[lp4 + 0];
      this.properties[p4 + 1] = this.properties[lp4 + 1];
      this.properties[p4 + 2] = this.properties[lp4 + 2];
      this.properties[p4 + 3] = this.properties[lp4 + 3];
    }

    this.activeFlags[lastIdx] = 0;
    this.count--;
  }
}
