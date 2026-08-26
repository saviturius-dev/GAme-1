import {
  FloatingDamageNumber,
  ParticleEffect,
  RedPlayerEntity,
  RLAction,
  SACActionVector,
  SACRewardVariant,
  ShockwaveEffect,
  SimulationConfig,
  SimulationStats,
  TerrainDeformationRecord,
  Vector3D,
  EmergentBehaviorProfile,
  CurriculumStage,
} from '../types';
import { simulationSound } from '../audio/soundEffects';
import { ContinuousSACAgent } from './continuousSACAgent';
import { RedSurvivalRLAgent } from './huntingRLAgent';
import { SwarmFieldEngine } from './swarmFieldEngine';

function getDistance(a: Vector3D, b: Vector3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function getDistance2D(a: Vector3D, b: Vector3D): number {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dz * dz);
}

export class UrsinaRepelSimulation {
  public player: RedPlayerEntity;
  public particles: ParticleEffect[] = [];
  public shockwaves: ShockwaveEffect[] = [];
  public floatingNumbers: FloatingDamageNumber[] = [];
  public stats: SimulationStats;
  public config: SimulationConfig;
  private autoSaveTimer: number = 0;

  // High-Performance O(N) Swarm Field Engine (Zero-GC Flat TypedArrays)
  public swarmField: SwarmFieldEngine = new SwarmFieldEngine(20000);

  // Soft Actor-Critic (SAC) Continuous Control RL Agent for 3D Snake
  public sacAgent: ContinuousSACAgent;
  private sacPreviousState: Float32Array | null = null;
  private sacCurrentAction: Float32Array = new Float32Array([0, 0, 0, 0, 0, 0, 0]);
  private sacSmoothedAction: Float32Array = new Float32Array([0, 0, 0, 0, 0, 0, 0]);
  private sacPreviousAction: Float32Array = new Float32Array([0, 0, 0, 0, 0, 0, 0]);
  private sacActionHoldCounter: number = 0;
  private sacRolloutStepCount: number = 0;
  private sacOldLength: number = 24;
  private sacStepRepelled: number = 0;
  private sacStepDash: boolean = false;
  private sacStepBarrier: boolean = false;
  private sacStepTrench: boolean = false;
  private abilityDebounce = { repel: 0, dash: 0, barrier: 0, trench: 0 };

  // Legacy / Tabular Q-Learning Agent (available for comparison)
  public rlAgent: RedSurvivalRLAgent;
  private rlDecisionTimer: number = 0;
  private rlPreviousStateKey: string = '';
  private rlPreviousAction: RLAction | null = null;
  private rlAccumulatedReward: number = 0;

  // Terrain Barriers & Trenches
  public terrainDeformations: TerrainDeformationRecord[] = [];

  private particleIdCounter: number = 0;
  private shockwaveIdCounter: number = 0;
  private floatingDamageIdCounter: number = 0;
  private terrainDeformIdCounter: number = 0;
  private episodeTimer: number = 0;
  private episodeCount: number = 1;
  private bestSurvivalTime: number = 0;
  private totalElapsedTime: number = 0;

  // Autonomous Anti-Stall & Unstuck Watchdog
  private unstuckTimer: number = 0;
  private lastUnstuckPos: { x: number; y: number; z: number } = { x: 0, y: 0.55, z: 0 };
  private stuckDuration: number = 0;

  constructor(customConfig?: Partial<SimulationConfig>) {
    this.config = {
      // Red Snake Player Settings
      playerSpeed: 5.4,
      playerDashSpeed: 15.0,
      playerMaxHealth: 100,
      playerRepelRadius: 6.5,
      playerRepelCooldown: 1.8,
      playerDashCooldown: 2.8,
      playerAutoHealRate: 1.2,
      snakeInitialLength: 24,
      snakeMaxCap: 1000,
      snakeElongateInterval: 3.0,
      snakeGrowthAmount: 5,
      snakeMinLength: 10,
      snakeShortenPerBite: 1,
      snakeSegmentSpacing: 0.42,

      // Autonomous Swarm & Continuous Self-Replication
      boidsEnabled: true,
      playerAttractionWeight: 2.2, // Active 3D Snake targeting and pursuit by default
      boidsAttackWeight: 2.2,
      boidsSeparationWeight: 2.0,
      boidsAlignmentWeight: 1.4,
      boidsCohesionWeight: 1.1,
      boidsSeparationRadius: 1.5,
      boidsNeighborRadius: 4.0,
      boidsMaxSpeed: 8.5,
      boidsMaxForce: 15.0,
      minionReplicationInterval: 0.01,
      minionReplicationMultiplier: 4,
      maxPointsCap: 4000,

      // O(N) 3D Perlin Flow Field & Organic Murmuration
      flowFieldEnabled: true,
      flowFieldWeight: 2.6,
      flowFieldScale: 0.002,
      flowFieldSpeed: 0.10,
      anchorClusterWeight: 1.4,
      individualWiggleWeight: 1.8,

      // Continuous Omnidirectional Kinematics & 3D Flight Formulation
      continuousSteering: true,
      enable3DFlight: true,
      playerVerticalSpeed: 5.0,
      snakeMinAltitude: 0.45,
      snakeMaxAltitude: 36.0,
      maxTurnRate: Math.PI / 2, // omega_max: 90 deg/s or 1.57 rad/s
      growthRate: 0.5, // g: +0.5 units/s passive growth
      hitPenalty: 5.0, // c: -5.0 units per hit
      forwardVelocity: 5.4, // v: constant forward speed
      rlAlphaLengthReward: 0.1, // alpha: survival reward scaled by length
      rlBetaHitPenalty: 10.0, // beta: penalty for taking a hit
      rlGammaTerminalPenalty: 100.0, // gamma: terminal death penalty
      nearestEnemiesCount: 3, // n: 3 nearest swarm boids
      manualControlStyle: 'omnidirectional_vector',

      // Real-Time 3D Deformable Cubic Terrain Barriers
      terrainDeformationEnabled: true,
      terrainBrushMode: 'none',
      terrainCubeSize: 3.0,
      terrainCubeHeight: 3.0,
      terrainCubeDepth: 2.5,
      terrainWireframe: false,
      terrainGridResolution: 50,

      // Soft Actor-Critic (SAC) Continuous 3D Control Configuration
      sacEnabled: true,
      sacLearningRate: 3e-4,
      sacDiscountFactor: 0.99,
      sacTau: 0.005,
      sacBatchSize: 32,
      sacReplayCapacity: 50000,
      sacActionPersistence: 4,
      sacRewardVariant: 'variant_c_combined',
      sacLengthScale: 50.0,
      sacAlphaWeight: 1.0,
      sacBetaWeight: 1.0,
      sacLambdaSmoothness: 0.005,
      sacNearestEnemiesCount: 8,
      sacPredictionHorizon: 10,
      sacDensityRadius: 8.0,
      sacIsEvaluation: false,
      sacMaxSpeed: 7.5,
      sacRolloutSteps: 5000,

      // Q-Learning RL Configuration for Red Survival
      rlEnabled: false,
      rlLearningRate: 0.12,
      rlDiscountFactor: 0.92,
      rlEpsilon: 0.35,
      rlEpsilonDecay: 0.996,
      rlMinEpsilon: 0.03,
      rlDecisionInterval: 0.25,

      // Reward weights
      rlRewardSurvivalPerSec: 0.25,
      rlRewardGrowth: 2.5,
      rlRewardRepelKill: 0.2,
      rlRewardBarrierDeflect: 1.2,
      rlPenaltyShortened: -2.0,
      rlPenaltyDamage: -1.0,
      rlPenaltyWastedRepel: -1.5,
      rlPenaltyDeath: -25.0,

      // Visuals & Environment
      showSnakeTailTrail: true,
      showRepelRadius: true,
      showPanicZone: true,
      showMinionTrails: true,
      showHealthBars: true,
      showDamageNumbers: true,
      showQDecisionHUD: true,
      arenaGridSize: 32,
      cameraPreset: 'overhead_follow_red',
      soundEnabled: true,
      bloomIntensity: 1.0,
      ...customConfig,
    };

    this.sacAgent = new ContinuousSACAgent(
      126, // Full 126-dim observation vector (includes 8-sector radar, kinematics, abilities, cooldowns, shielding, winding area)
      7, // 7D Continuous action vector: [ax, ay, az, a_repel, a_dash, a_barrier, a_trench]
      this.config.sacLearningRate ?? 3e-4,
      this.config.sacDiscountFactor ?? 0.99,
      this.config.sacTau ?? 0.005,
      this.config.sacReplayCapacity ?? 50000
    );
    if (this.config.sacUsePER !== undefined) this.sacAgent.usePER = this.config.sacUsePER;
    if (this.config.sacPerAlpha !== undefined) this.sacAgent.perAlpha = this.config.sacPerAlpha;
    if (this.config.sacPerBeta !== undefined) this.sacAgent.perBeta = this.config.sacPerBeta;
    if (this.config.sacUseNStep !== undefined) this.sacAgent.useNStep = this.config.sacUseNStep;
    if (this.config.sacNStep !== undefined) this.sacAgent.nStep = this.config.sacNStep;

    this.rlAgent = new RedSurvivalRLAgent(
      this.config.rlEpsilon ?? 0.35,
      this.config.rlLearningRate ?? 0.12,
      this.config.rlDiscountFactor ?? 0.92
    );

    const initLen = this.config.snakeInitialLength ?? 24;
    const initialSegments: { x: number; y: number; z: number; angle: number }[] = [];
    const initialHistory: Vector3D[] = [];
    for (let i = 0; i < initLen; i++) {
      const pos = {
        x: 0,
        y: 0.55,
        z: -i * this.config.snakeSegmentSpacing,
      };
      initialSegments.push({
        ...pos,
        angle: 0,
      });
      initialHistory.push(pos);
    }

    const initialForwardVel = this.config.forwardVelocity ?? this.config.playerSpeed ?? 5.4;
    this.player = {
      position: { x: 0, y: 0.55, z: 0 },
      velocity: { x: 0, y: 0, z: initialForwardVel },
      targetPos: { x: 0, y: 0.55, z: 0 },
      facingAngle: 0,
      heading: 0,
      pitch: 0,
      altitude: 0.55,
      verticalVelocity: 0,
      steeringInput: 0,
      angularVelocity: 0,
      continuousLength: initLen,
      bodyHistory: initialHistory,
      lastStepReward: 0,
      lastStepHits: 0,
      observationVector: [initLen, 0, 0, 0, 0, 0, 0, 0],
      turnSpeed: 4.5,
      walkCycle: 0,
      speed: this.config.playerSpeed,
      scale: 1.25,
      segments: initialSegments,
      snakeLength: initLen,
      snakeMaxCap: this.config.snakeMaxCap ?? 1000,
      elongateTimer: 0,
      elongateInterval: this.config.snakeElongateInterval ?? 3.0,
      growthAmount: this.config.snakeGrowthAmount ?? 5,
      peakLength: initLen,
      shortenCount: 0,
      lastShortenedTimer: 0,
      growthPulseTimer: 0,
      headState: 'idle_cruise',
      jawAngle: 0,
      targetJawAngle: 0,
      kineticCrownCharge: 0,
      eyeTargetPos: null,
      health: this.config.playerMaxHealth,
      maxHealth: this.config.playerMaxHealth,
      isDead: false,
      damageFlashTimer: 0,
      score: 0,
      lastDamageTaken: 0,
      repelCooldown: 0,
      maxRepelCooldown: this.config.playerRepelCooldown,
      repelRadius: this.config.playerRepelRadius,
      flashTimer: 0,
      isDashing: false,
      dashTimer: 0,
      dashCooldown: 0,
      maxDashCooldown: this.config.playerDashCooldown,
      dashDirection: { x: 0, y: 0, z: 0 },
      controlMode: 'rl_agent',
      survivalTime: 0,
      boidsRepelledCount: 0,
      barriersPlacedCount: 0,
      trenchesCarvedCount: 0,
      damageTakenTotal: 0,
    };

    this.stats = {
      snakeLength: initLen,
      continuousLength: initLen,
      heading: 0,
      pitch: 0,
      altitude: 0.55,
      verticalVelocity: 0,
      steeringInput: 0,
      angularVelocity: 0,
      lastStepReward: 0,
      lastStepHits: 0,
      observationVector: [initLen, 0, 0, 0, 0, 0, 0, 0],
      snakeMaxCap: 1000,
      peakLength: initLen,
      nextElongateCountdown: 3.0,
      bitesTakenCount: 0,
      playerHealth: 100,
      playerMaxHealth: 100,
      playerIsDead: false,
      playerSurvivalTime: 0,
      bestSurvivalTime: 0,
      episodeNumber: 1,
      playerScore: 0,
      playerControlMode: 'rl_agent',
      playerRepelCooldownRemaining: 0,
      playerDashCooldownRemaining: 0,
      lastDamageEvent: null,
      activeBoids: 0,
      totalSpawned: 0,
      totalRepelled: 0,
      repelCount: 0,
      boidsNearPlayer: 0,
      boidsInBlastZone: 0,
      totalBarriersPlaced: 0,
      totalTrenchesCarved: 0,
      playerAttractionWeight: this.config.playerAttractionWeight,
      avgBoidSurvivalTime: 0,
      maxBoidSurvivalTime: 0,
      swarmAttackState: 0,
      swarmAttackStateName: 'Flocking Murmuration',
      swarmDemographics: {
        newborn: 0,
        youth: 0,
        mature: 0,
        veteran: 0,
        ancient: 0,
      },
      sacMetrics: this.sacAgent.getMetrics(
        this.config.sacEnabled,
        this.config.sacIsEvaluation,
        this.config.sacRewardVariant
      ),
      rlAgent: this.rlAgent.getStats(this.config.rlEnabled),
      elapsedTime: 0,
      fps: 60,
    };

    this.seedInitialSwarm();
  }

  public seedInitialSwarm() {
    this.swarmField.reset();
    // Seed autonomous boids surrounding the arena
    for (let i = 0; i < 28; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = 9.0 + Math.random() * 12.0;
      this.swarmField.spawnPoint(
        Math.cos(angle) * dist,
        1.0 + Math.random() * 2.5,
        Math.sin(angle) * dist,
        (Math.random() - 0.5) * 3.0,
        (Math.random() - 0.5) * 1.5,
        (Math.random() - 0.5) * 3.0,
        0
      );
    }
  }

  public reset(keepQTable: boolean = true) {
    if (this.player.survivalTime > this.bestSurvivalTime) {
      this.bestSurvivalTime = this.player.survivalTime;
    }
    if (this.player.snakeLength > this.player.peakLength) {
      this.player.peakLength = this.player.snakeLength;
    }

    this.rlAgent.recordEpisodeEnd({
      episode: this.episodeCount,
      survivalTime: this.player.survivalTime,
      peakLength: this.player.peakLength,
      repelsCount: this.player.boidsRepelledCount,
      boidsEliminated: this.stats.totalRepelled,
      totalReward: this.rlAgent.totalReward,
      barriersDeployed: this.player.barriersPlacedCount,
      causeOfDeath: 'Manual Reset',
    });

    // Auto-save learned models so reset never wipes learned progress
    this.sacAgent.saveToLocalStorage();
    this.rlAgent.saveToLocalStorage();

    if (!keepQTable) {
      this.rlAgent.resetQTable();
    }

    this.episodeCount++;
    this.episodeTimer = 0;

    const initLen = this.config.snakeInitialLength ?? 24;
    const initialSegments: { x: number; y: number; z: number; angle: number }[] = [];
    const initialHistory: Vector3D[] = [];
    for (let i = 0; i < initLen; i++) {
      const pos = {
        x: 0,
        y: 0.55,
        z: -i * this.config.snakeSegmentSpacing,
      };
      initialSegments.push({
        ...pos,
        angle: 0,
      });
      initialHistory.push(pos);
    }

    const forwardVel = this.config.forwardVelocity ?? this.config.playerSpeed ?? 5.4;
    this.player.position = { x: 0, y: 0.55, z: 0 };
    this.player.velocity = { x: 0, y: 0, z: forwardVel };
    this.player.targetPos = { x: 0, y: 0.55, z: 0 };
    this.player.facingAngle = 0;
    this.player.heading = 0;
    this.player.steeringInput = 0;
    this.player.angularVelocity = 0;
    this.player.continuousLength = initLen;
    this.player.bodyHistory = initialHistory;
    this.player.lastStepReward = 0;
    this.player.lastStepHits = 0;
    this.player.observationVector = [initLen, 0, 0, 0, 0, 0, 0, 0];
    this.player.walkCycle = 0;
    this.player.segments = initialSegments;
    this.player.snakeLength = initLen;
    this.player.elongateTimer = 0;
    this.player.shortenCount = 0;
    this.player.lastShortenedTimer = 0;
    this.player.growthPulseTimer = 0;
    this.player.health = this.config.playerMaxHealth;
    this.player.isDead = false;
    this.player.damageFlashTimer = 0;
    this.player.headState = 'idle_cruise';
    this.player.jawAngle = 0;
    this.player.targetJawAngle = 0;
    this.player.kineticCrownCharge = 0;
    this.player.eyeTargetPos = null;
    this.player.repelCooldown = 0;
    this.player.dashCooldown = 0;
    this.player.isDashing = false;
    this.player.survivalTime = 0;
    this.player.boidsRepelledCount = 0;
    this.player.barriersPlacedCount = 0;
    this.player.trenchesCarvedCount = 0;

    this.particles = [];
    this.shockwaves = [];
    this.floatingNumbers = [];
    this.rlDecisionTimer = 0;
    this.rlPreviousAction = null;
    this.rlPreviousStateKey = '';
    this.rlAccumulatedReward = 0;

    this.seedInitialSwarm();
    this.syncStats();
  }

  /**
   * Spawns an explosion particle effect triggered when a boid's health reaches zero,
   * fading from ash grey to slate to charcoal soot to simulate a decaying creature.
   */
  public spawnDecayExplosion(
    x: number,
    y: number,
    z: number,
    baseVx: number = 0,
    baseVy: number = 0,
    baseVz: number = 0,
    count: number = 20
  ) {
    const particleCount = Math.max(12, Math.min(36, count));
    for (let i = 0; i < particleCount; i++) {
      // Outward 3D spherical blast vector with organic spread
      const phi = Math.random() * Math.PI * 2;
      const cosTheta = 2 * Math.random() - 1;
      const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
      const speed = 2.2 + Math.random() * 5.2;

      const vx = sinTheta * Math.cos(phi) * speed + baseVx * 0.25;
      const vy = Math.abs(cosTheta) * speed * 0.7 + Math.random() * 2.2 + baseVy * 0.25; // upward thermal ashen draft
      const vz = sinTheta * Math.sin(phi) * speed + baseVz * 0.25;

      const maxLife = 0.65 + Math.random() * 0.65; // 0.65s to 1.30s decay time
      const size = 0.55 + Math.random() * 0.40;

      this.particles.push({
        id: `decay_${this.particleIdCounter++}`,
        position: {
          x: x + (Math.random() - 0.5) * 0.25,
          y: Math.max(0.1, y + (Math.random() - 0.5) * 0.25),
          z: z + (Math.random() - 0.5) * 0.25,
        },
        velocity: { x: vx, y: vy, z: vz },
        color: '#b0b5be', // Starting grey ash tone
        size,
        life: maxLife,
        maxLife,
        type: 'decay_creature_explosion',
        drag: 0.94,
        gravity: 2.4,
      });
    }
  }

  public manualRepel(): number {
    if (this.player.repelCooldown > 0) return 0;

    const onDeath = (x: number, y: number, z: number, vx: number, vy: number, vz: number) => {
      this.spawnDecayExplosion(x, y, z, vx, vy, vz, 20);
    };

    // Apply blast centered on Snake Head
    let repelled = this.swarmField.applyRepelBlast(
      this.player.position.x,
      this.player.position.y,
      this.player.position.z,
      this.player.repelRadius,
      32.0,
      100.0,
      onDeath
    );

    // Secondary blast along snake body to protect tail
    if (this.player.segments.length > 6) {
      const midSeg = this.player.segments[Math.floor(this.player.segments.length * 0.5)];
      if (midSeg) {
        repelled += this.swarmField.applyRepelBlast(
          midSeg.x,
          midSeg.y,
          midSeg.z,
          this.player.repelRadius * 0.8,
          24.0,
          100.0,
          onDeath
        );
      }
    }

    this.player.repelCooldown = this.player.maxRepelCooldown;
    this.player.flashTimer = 0.35;
    this.player.boidsRepelledCount += repelled;
    this.stats.repelCount++;
    this.stats.totalRepelled += repelled;

    // Trigger kinetic shockwave visual effect
    this.shockwaves.push({
      id: `repel_${this.shockwaveIdCounter++}`,
      position: { ...this.player.position },
      radius: 0.5,
      maxRadius: this.player.repelRadius,
      life: 0.35,
      maxLife: 0.35,
      color: '#f43f5e',
    });

    if (this.config.soundEnabled) {
      simulationSound.playRepelBlast();
    }

    return repelled;
  }

  public manualAscend(dt: number = 0.05) {
    const vSpeed = this.config.playerVerticalSpeed ?? 6.0;
    this.player.targetAltitude = undefined;
    this.player.targetPitch = 0.55; // Dynamic pitch up into 3D sky
    this.player.velocity.y = Math.min(vSpeed * 1.5, Math.max(vSpeed * 0.8, this.player.velocity.y + vSpeed * dt * 12.0));
  }

  public manualDescend(dt: number = 0.05) {
    const vSpeed = this.config.playerVerticalSpeed ?? 6.0;
    this.player.targetAltitude = undefined;
    this.player.targetPitch = -0.55; // Dynamic pitch down toward ground
    this.player.velocity.y = Math.max(-vSpeed * 1.5, Math.min(-vSpeed * 0.8, this.player.velocity.y - vSpeed * dt * 12.0));
  }

  public resetVerticalControl() {
    this.player.targetPitch = undefined;
  }

  public setTargetAltitude(alt: number) {
    this.player.targetAltitude = Math.max(
      this.config.snakeMinAltitude ?? 0.45,
      Math.min(this.config.snakeMaxAltitude ?? 36.0, alt)
    );
    this.player.targetPitch = undefined;
  }

  public manualDash() {
    if (this.player.dashCooldown > 0 || this.player.isDashing) return;

    // Dash in direction of movement or facing angle
    let dx = this.player.velocity.x;
    let dz = this.player.velocity.z;
    const len = Math.hypot(dx, dz);

    if (len < 0.1) {
      dx = Math.sin(this.player.facingAngle);
      dz = Math.cos(this.player.facingAngle);
    } else {
      dx /= len;
      dz /= len;
    }

    this.player.isDashing = true;
    this.player.dashTimer = 0.22;
    this.player.dashCooldown = this.player.maxDashCooldown;
    this.player.dashDirection = { x: dx, y: 0, z: dz };

    // Spawn dash particles
    for (let i = 0; i < 8; i++) {
      this.particles.push({
        id: `part_${this.particleIdCounter++}`,
        position: { ...this.player.position },
        velocity: {
          x: -dx * 4.0 + (Math.random() - 0.5) * 2,
          y: Math.random() * 1.5,
          z: -dz * 4.0 + (Math.random() - 0.5) * 2,
        },
        color: '#dc2626',
        size: 0.35,
        life: 0.35,
        maxLife: 0.35,
      });
    }
  }

  public deployBarrier(point?: Vector3D, cubeSize: number = 3.0, height: number = 3.0) {
    if (this.player.isDead) return;

    let targetPoint = point;
    if (!targetPoint) {
      // Place barrier directly between Red and the swarm centroid
      const centroid = this.swarmField.swarmCentroid;
      const dx = centroid.x - this.player.position.x;
      const dz = centroid.z - this.player.position.z;
      const dist = Math.hypot(dx, dz) || 1.0;
      const offset = 2.8;

      targetPoint = {
        x: Math.max(-25, Math.min(25, this.player.position.x + (dx / dist) * offset)),
        y: 0,
        z: Math.max(-25, Math.min(25, this.player.position.z + (dz / dist) * offset)),
      };
    }

    this.deformTerrain(targetPoint, cubeSize, height, 'big');
    this.player.barriersPlacedCount++;
    this.stats.totalBarriersPlaced++;
  }

  public carveTrench(point?: Vector3D, cubeSize: number = 3.5, depth: number = -2.5) {
    if (this.player.isDead) return;

    let targetPoint = point;
    if (!targetPoint) {
      const centroid = this.swarmField.swarmCentroid;
      const dx = centroid.x - this.player.position.x;
      const dz = centroid.z - this.player.position.z;
      const dist = Math.hypot(dx, dz) || 1.0;
      const offset = 3.5;

      targetPoint = {
        x: Math.max(-25, Math.min(25, this.player.position.x + (dx / dist) * offset)),
        y: 0,
        z: Math.max(-25, Math.min(25, this.player.position.z + (dz / dist) * offset)),
      };
    }

    this.deformTerrain(targetPoint, cubeSize, depth, 'big');
    this.player.trenchesCarvedCount++;
    this.stats.totalTrenchesCarved++;
  }

  public deformTerrain(
    point: { x: number; y: number; z: number },
    cubeSize: number = 3.0,
    heightChange: number = 3.0,
    cubeType: 'small' | 'big' | 'cuboid' = 'big',
    customDimensions?: { width?: number; height?: number; depth?: number }
  ) {
    const record: TerrainDeformationRecord = {
      id: `deform_${this.terrainDeformIdCounter++}`,
      point: { x: Math.round(point.x * 2) / 2, y: 0, z: Math.round(point.z * 2) / 2 },
      cubeSize,
      width: customDimensions?.width ?? cubeSize,
      height: customDimensions?.height ?? Math.abs(heightChange),
      depth: customDimensions?.depth ?? cubeSize,
      baseY: 0,
      cubeType,
      heightChange,
      isCarved: heightChange < 0,
      timestamp: Date.now(),
    };

    // Remove overlapping records at similar coordinates
    this.terrainDeformations = this.terrainDeformations.filter(
      (t) => getDistance2D(t.point, record.point) > cubeSize * 0.85
    );

    this.terrainDeformations.push(record);
    if (this.terrainDeformations.length > 25) {
      this.terrainDeformations.shift();
    }
  }

  public resetTerrain() {
    this.terrainDeformations = [];
  }

  public manualSpawnSwarm(count: number = 20) {
    const angle = Math.random() * Math.PI * 2;
    const dist = 10.0 + Math.random() * 8.0;
    this.swarmField.spawnCluster(
      count,
      Math.cos(angle) * dist,
      1.5,
      Math.sin(angle) * dist,
      3.0
    );
  }

  public setControlMode(mode: 'rl_agent' | 'manual_player') {
    this.player.controlMode = mode;
  }

  /**
   * Synchronize and dynamically apply updated configuration parameters to all active systems
   */
  public updateConfig(updates: Partial<SimulationConfig>) {
    this.config = { ...this.config, ...updates };

    // Synchronize Red Snake parameters immediately
    if (this.config.playerRepelRadius !== undefined) {
      this.player.repelRadius = this.config.playerRepelRadius;
    }
    if (this.config.playerRepelCooldown !== undefined) {
      this.player.maxRepelCooldown = this.config.playerRepelCooldown;
    }
    if (this.config.playerDashCooldown !== undefined) {
      this.player.maxDashCooldown = this.config.playerDashCooldown;
    }
    if (this.config.snakeElongateInterval !== undefined) {
      this.player.elongateInterval = this.config.snakeElongateInterval;
    }
    if (this.config.snakeGrowthAmount !== undefined) {
      this.player.growthAmount = this.config.snakeGrowthAmount;
    }
    if (this.config.snakeMaxCap !== undefined) {
      this.player.snakeMaxCap = this.config.snakeMaxCap;
    }
    if (this.config.playerMaxHealth !== undefined) {
      this.player.maxHealth = this.config.playerMaxHealth;
    }
    if (this.config.sacMaxSpeed !== undefined || this.config.playerSpeed !== undefined || this.config.forwardVelocity !== undefined) {
      this.player.speed = this.config.sacMaxSpeed ?? this.config.forwardVelocity ?? this.config.playerSpeed ?? 7.5;
    }

    // Synchronize SAC Brain Hyperparameters
    if (this.sacAgent) {
      if (this.config.sacLearningRate !== undefined) {
        this.sacAgent.lr = this.config.sacLearningRate;
      }
      if (this.config.sacDiscountFactor !== undefined) {
        this.sacAgent.gamma = this.config.sacDiscountFactor;
      }
      if (this.config.sacTau !== undefined) {
        this.sacAgent.tau = this.config.sacTau;
      }
      if (this.config.sacUsePER !== undefined) {
        this.sacAgent.usePER = this.config.sacUsePER;
      }
      if (this.config.sacPerAlpha !== undefined) {
        this.sacAgent.perAlpha = this.config.sacPerAlpha;
      }
      if (this.config.sacPerBeta !== undefined) {
        this.sacAgent.perBeta = this.config.sacPerBeta;
      }
      if (this.config.sacUseNStep !== undefined) {
        this.sacAgent.useNStep = this.config.sacUseNStep;
      }
      if (this.config.sacNStep !== undefined) {
        this.sacAgent.nStep = this.config.sacNStep;
      }
    }

    // Synchronize Tabular RL Agent Hyperparameters
    if (this.rlAgent) {
      if (this.config.rlLearningRate !== undefined) {
        this.rlAgent.alpha = this.config.rlLearningRate;
      }
      if (this.config.rlDiscountFactor !== undefined) {
        this.rlAgent.gamma = this.config.rlDiscountFactor;
      }
      if (this.config.rlEpsilon !== undefined) {
        this.rlAgent.epsilon = this.config.rlEpsilon;
      }
    }

    this.syncStats();
  }

  /**
   * Main Simulation Step Update
   */
  public update(dt: number) {
    dt = Math.min(0.1, dt);
    this.totalElapsedTime += dt;
    this.episodeTimer += dt;
    this.player.survivalTime += dt;

    // 1. Swarm Continuous High-Frequency Replication (1 becomes 4 every interval)
    if (this.config.boidsEnabled) {
      this.swarmField.stepReplication(
        dt,
        this.config.minionReplicationInterval,
        this.config.minionReplicationMultiplier,
        this.config.maxPointsCap
      );
    }

    // 2. Swarm Field Update (Organic Flow Field + Full 3D Multi-Altitude Snake Targeting)
    this.swarmField.update(
      dt,
      this.totalElapsedTime,
      this.player.position.x,
      this.player.position.y ?? 0.55,
      this.player.position.z,
      this.terrainDeformations,
      {
        flowFieldEnabled: this.config.flowFieldEnabled,
        flowFieldWeight: this.config.flowFieldWeight,
        flowFieldScale: this.config.flowFieldScale,
        flowFieldSpeed: this.config.flowFieldSpeed,
        playerAttractionWeight: this.config.playerAttractionWeight,
        boidsSeparationWeight: this.config.boidsSeparationWeight,
        boidsAlignmentWeight: this.config.boidsAlignmentWeight,
        boidsCohesionWeight: this.config.boidsCohesionWeight,
        boidsMaxSpeed: this.config.boidsMaxSpeed,
        boidsMaxForce: this.config.boidsMaxForce,
        swarmColorMode: this.config.swarmColorMode,
        swarmAgeMaxThreshold: this.config.swarmAgeMaxThreshold,
        swarmDynamicSizeByAge: this.config.swarmDynamicSizeByAge,
      },
      this.player.segments
    );

    // 3. Update Red Snake Physics, Serpentine Kinematics & 3-Second Elongation
    this.updatePlayerPhysics(dt);

    // 4. Reinforcement Learning Agent Brain Step (Snake Defense & Length Optimization)
    if (this.player.controlMode === 'rl_agent') {
      if (this.config.sacEnabled) {
        this.stepSACAgent(dt);
      } else if (this.config.rlEnabled) {
        this.stepRLAgent(dt);
      }
    }

    // 5. Check Swarm Collisions (Boids actively shorten snake upon bites)
    this.checkSwarmCollisions(dt);

    // 6. Update Visual Particles, Shockwaves & Damage Numbers
    this.updateEffects(dt);

    // 7. Log SAC Step Telemetry
    if (this.config.sacEnabled) {
      this.sacAgent.logStepMetrics(
        this.player.snakeLength,
        this.player.lastStepHits,
        this.player.growthPulseTimer > 0.7
      );
    }

    // 8. Auto-save learned models to local storage periodically so training is never lost
    this.autoSaveTimer += dt;
    if (this.autoSaveTimer >= 3.0) {
      this.autoSaveTimer = 0;
      this.sacAgent.saveToLocalStorage();
      this.rlAgent.saveToLocalStorage();
    }

    this.syncStats();
  }

  /**
   * Generates the normalized Continuous Observation Vector s_t (dim = 126) for SAC 3D Continuous Control & Ability Triggering:
   * - Snake kinematics: pos [3], vel [3], length [1] = [7]
   * - Boundary distances: [6]
   * - Nearest K=6 Swarm Agents: relative pos [3], relative vel [3], distance [1], future predicted relative pos [3] = 6 * 10 = [60]
   * - 8-Sector Egocentric Radar: Sector Densities [8], Min Distances [8], Approach Velocities [8] = [24]
   * - Snake Body Proprioception & Spine Curvature: Angles [3], Bounding Envelope [3] = [6]
   * - Swarm Centroid Kinematics & Threat Vector: Relative Pos [3], Vel [3], Dist [1] = [7]
   * - Local swarm density: [1]
   * - Previous action vector (7D: [ax, ay, az, a_repel, a_dash, a_barrier, a_trench]): [7]
   * - Recent hits & damage: [2]
   * - Cooldowns & Tactical State: Repel CD [1], Dash CD [1], Swarm Attack State [1], Barrier Shielded [1], Winding Area [1], Curriculum [1] = [6]
   * Total = 7 + 6 + 60 + 24 + 6 + 7 + 1 + 7 + 2 + 6 = 126
   */
  public getSnakeObservation(): Float32Array {
    const obs = new Float32Array(126);
    const W = 28.0; // Arena horizontal half-bound
    const H = 36.0; // Arena vertical height
    const vMax = this.config.sacMaxSpeed ?? 7.5;
    const lScale = this.config.sacLengthScale ?? 50.0;
    const px = this.player.position.x;
    const py = this.player.position.y ?? 0.55;
    const pz = this.player.position.z;
    const pvx = this.player.velocity.x;
    const pvy = this.player.velocity.y;
    const pvz = this.player.velocity.z;

    // 1. Snake State (7 floats: 0..6)
    obs[0] = Math.max(-1.0, Math.min(1.0, px / W));
    obs[1] = Math.max(0.0, Math.min(1.0, py / H));
    obs[2] = Math.max(-1.0, Math.min(1.0, pz / W));
    obs[3] = Math.max(-1.0, Math.min(1.0, pvx / vMax));
    obs[4] = Math.max(-1.0, Math.min(1.0, pvy / vMax));
    obs[5] = Math.max(-1.0, Math.min(1.0, pvz / vMax));
    obs[6] = Math.max(0.0, Math.min(20.0, this.player.continuousLength / lScale));

    // 2. Boundary Distances (6 floats: 7..12)
    obs[7] = Math.max(0.0, Math.min(1.0, (px + W) / (2 * W))); // Dist to -X
    obs[8] = Math.max(0.0, Math.min(1.0, (W - px) / (2 * W))); // Dist to +X
    obs[9] = Math.max(0.0, Math.min(1.0, py / H)); // Dist to floor
    obs[10] = Math.max(0.0, Math.min(1.0, (H - py) / H)); // Dist to ceiling
    obs[11] = Math.max(0.0, Math.min(1.0, (pz + W) / (2 * W))); // Dist to -Z
    obs[12] = Math.max(0.0, Math.min(1.0, (W - pz) / (2 * W))); // Dist to +Z

    // 3. Nearest K=6 Swarm Agents & 8-Sector Egocentric Radar Data Prep
    const activeCount = this.swarmField.count;
    const tauPrediction = (this.config.sacPredictionHorizon ?? 10) * 0.016;
    const densityRadius = this.config.sacDensityRadius ?? 8.0;

    // Zero-GC fixed top-6 nearest boids tracking
    const top6DistSq = [Infinity, Infinity, Infinity, Infinity, Infinity, Infinity];
    const top6Idx = [-1, -1, -1, -1, -1, -1];

    // 8-Sector Radar Accumulators (heading reference in X-Z plane)
    const heading = Math.atan2(pvz, pvx) || 0;
    const sectorCounts = new Float32Array(8);
    const sectorMinDists = new Float32Array(8).fill(2.0);
    const sectorApprVel = new Float32Array(8);

    let localDensityCount = 0;

    for (let i = 0; i < activeCount; i++) {
      const bx = this.swarmField.boidX[i];
      const by = this.swarmField.boidY[i];
      const bz = this.swarmField.boidZ[i];
      const bvx = this.swarmField.boidVx[i];
      const bvy = this.swarmField.boidVy[i];
      const bvz = this.swarmField.boidVz[i];

      const dx = bx - px;
      const dy = by - py;
      const dz = bz - pz;
      const distSq = dx * dx + dy * dy + dz * dz;

      // Track top 6 nearest boids in O(1) with zero heap allocations
      if (distSq < top6DistSq[5]) {
        let ins = 5;
        while (ins > 0 && distSq < top6DistSq[ins - 1]) {
          top6DistSq[ins] = top6DistSq[ins - 1];
          top6Idx[ins] = top6Idx[ins - 1];
          ins--;
        }
        top6DistSq[ins] = distSq;
        top6Idx[ins] = i;
      }

      const dist = Math.sqrt(distSq);

      if (dist < densityRadius) {
        localDensityCount++;
      }

      // Compute egocentric sector (0..7)
      const boidAngle = Math.atan2(dz, dx);
      let relAngle = boidAngle - heading;
      while (relAngle > Math.PI) relAngle -= 2 * Math.PI;
      while (relAngle < -Math.PI) relAngle += 2 * Math.PI;
      const sectorIdx = Math.max(0, Math.min(7, Math.floor(((relAngle + Math.PI) / (2 * Math.PI)) * 8)));

      sectorCounts[sectorIdx]++;
      const normDist = Math.min(2.0, dist / W);
      if (normDist < sectorMinDists[sectorIdx]) {
        sectorMinDists[sectorIdx] = normDist;
      }
      const radialVel = dist > 0.001 ? ((bx - px) * (bvx - pvx) + (bz - pz) * (bvz - pvz)) / dist : 0;
      sectorApprVel[sectorIdx] += Math.max(-2.0, Math.min(2.0, radialVel / vMax));
    }

    let offset = 13;
    for (let k = 0; k < 6; k++) {
      const bIdx = top6Idx[k];
      if (bIdx >= 0 && bIdx < activeCount) {
        const bx = this.swarmField.boidX[bIdx];
        const by = this.swarmField.boidY[bIdx];
        const bz = this.swarmField.boidZ[bIdx];
        const bvx = this.swarmField.boidVx[bIdx];
        const bvy = this.swarmField.boidVy[bIdx];
        const bvz = this.swarmField.boidVz[bIdx];
        const dx = bx - px;
        const dy = by - py;
        const dz = bz - pz;
        const dist = Math.sqrt(top6DistSq[k]);

        const predDx = bx + bvx * tauPrediction - px;
        const predDy = by + bvy * tauPrediction - py;
        const predDz = bz + bvz * tauPrediction - pz;

        obs[offset++] = Math.max(-2.0, Math.min(2.0, dx / W));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, dy / H));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, dz / W));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, (bvx - pvx) / vMax));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, (bvy - pvy) / vMax));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, (bvz - pvz) / vMax));
        obs[offset++] = Math.max(0.0, Math.min(2.0, dist / W));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, predDx / W));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, predDy / H));
        obs[offset++] = Math.max(-2.0, Math.min(2.0, predDz / W));
      } else {
        obs[offset++] = 0;
        obs[offset++] = 0;
        obs[offset++] = 0;
        obs[offset++] = 0;
        obs[offset++] = 0;
        obs[offset++] = 0;
        obs[offset++] = 2.0; // max dist
        obs[offset++] = 0;
        obs[offset++] = 0;
        obs[offset++] = 0;
      }
    }

    // 4. 8-Sector Egocentric Radar (24 floats: 73..96)
    for (let s = 0; s < 8; s++) {
      obs[73 + s] = Math.min(1.0, sectorCounts[s] / 25.0); // Sector Density
      obs[81 + s] = sectorMinDists[s]; // Sector Min Distance
      obs[89 + s] = sectorCounts[s] > 0 ? Math.max(-1.0, Math.min(1.0, sectorApprVel[s] / sectorCounts[s])) : 0; // Sector Approach Velocity
    }

    // 5. Snake Body Proprioception & Spine Curvature (6 floats: 97..102)
    const body = this.player.segments;
    const bodyLen = body.length;
    if (bodyLen >= 3) {
      const pHead = this.player.position;
      const pMid = body[Math.floor(bodyLen / 2)];
      const pTail = body[bodyLen - 1];

      const angleHeadMid = Math.atan2(pMid.z - pHead.z, pMid.x - pHead.x) - heading;
      const angleMidTail = Math.atan2(pTail.z - pMid.z, pTail.x - pMid.x) - Math.atan2(pMid.z - pHead.z, pMid.x - pHead.x);
      obs[97] = Math.sin(angleHeadMid);
      obs[98] = Math.cos(angleHeadMid);
      obs[99] = Math.sin(angleMidTail);

      // Bounding envelope
      let minX = pHead.x, maxX = pHead.x, minY = pHead.y, maxY = pHead.y, minZ = pHead.z, maxZ = pHead.z;
      for (let i = 0; i < bodyLen; i++) {
        const seg = body[i];
        if (seg.x < minX) minX = seg.x; if (seg.x > maxX) maxX = seg.x;
        if (seg.y < minY) minY = seg.y; if (seg.y > maxY) maxY = seg.y;
        if (seg.z < minZ) minZ = seg.z; if (seg.z > maxZ) maxZ = seg.z;
      }
      obs[100] = Math.min(1.0, (maxX - minX) / W);
      obs[101] = Math.min(1.0, (maxY - minY) / H);
      obs[102] = Math.min(1.0, (maxZ - minZ) / W);
    } else {
      obs[97] = 0; obs[98] = 1.0; obs[99] = 0;
      obs[100] = 0.1; obs[101] = 0.1; obs[102] = 0.1;
    }

    // 6. Swarm Centroid Kinematics & Threat Vector (7 floats: 103..109)
    const centroid = this.swarmField.swarmCentroid;
    const toCentroidX = centroid.x - px;
    const toCentroidY = centroid.y - py;
    const toCentroidZ = centroid.z - pz;
    const centroidDist = Math.hypot(toCentroidX, toCentroidZ) || 1.0;

    obs[103] = Math.max(-2.0, Math.min(2.0, toCentroidX / W));
    obs[104] = Math.max(-2.0, Math.min(2.0, toCentroidY / H));
    obs[105] = Math.max(-2.0, Math.min(2.0, toCentroidZ / W));
    obs[106] = 0; // Centroid relative vx
    obs[107] = 0; // Centroid relative vy
    obs[108] = 0; // Centroid relative vz
    obs[109] = Math.min(2.0, centroidDist / W);

    // 7. Local Swarm Density (1 float: 110)
    obs[110] = Math.min(1.0, localDensityCount / 60.0);

    // 8. Previous 7D Action Vector (7 floats: 111..117)
    for (let a = 0; a < 7; a++) {
      obs[111 + a] = this.sacCurrentAction[a] ?? 0;
    }

    // 9. Recent Hits and Shortening Damage (2 floats: 118..119)
    obs[118] = Math.min(1.0, this.player.lastStepHits / 4.0);
    obs[119] = Math.min(1.0, this.player.shortenCount / 20.0);

    // 10. Tactical Cooldowns & State (6 floats: 120..125)
    const repelCdMax = this.config.playerRepelCooldown ?? 1.8;
    const dashCdMax = this.config.playerDashCooldown ?? 2.8;
    obs[120] = Math.max(0.0, Math.min(1.0, this.player.repelCooldown / repelCdMax));
    obs[121] = Math.max(0.0, Math.min(1.0, this.player.dashCooldown / dashCdMax));
    obs[122] = Math.min(1.0, (this.swarmField.attackState ?? 0) / 3.0);

    // Check if player is shielded by a raised terrain barrier
    let isShielded = false;
    for (const b of this.terrainDeformations) {
      if (b.heightChange > 0) {
        const bdx = b.point.x - px;
        const bdz = b.point.z - pz;
        const bDist = Math.hypot(bdx, bdz);
        if (bDist < centroidDist) {
          const dot = (bdx * toCentroidX + bdz * toCentroidZ) / (bDist * centroidDist);
          if (dot > 0.65) {
            isShielded = true;
            break;
          }
        }
      }
    }
    obs[123] = isShielded ? 1.0 : 0.0;

    // Shoelace Winding Area enclosed by snake body
    let windingArea = 0;
    if (bodyLen >= 6) {
      let sum = 0;
      const poly = [{ x: px, z: pz }, ...body];
      for (let i = 0; i < poly.length; i++) {
        const j = (i + 1) % poly.length;
        sum += poly[i].x * poly[j].z - poly[j].x * poly[i].z;
      }
      windingArea = Math.abs(sum) * 0.5;
    }
    obs[124] = Math.min(1.0, windingArea / 120.0);

    // Curriculum stage normalized
    const stages: CurriculumStage[] = [
      'stage_1_survival',
      'stage_2_tactical_harvesting',
      'stage_3_adversarial_mastery',
    ];
    const stageIdx = stages.indexOf(this.sacAgent.curriculumStage);
    obs[125] = stageIdx >= 0 ? stageIdx / 2.0 : 0.5;

    return obs;
  }

  /**
   * Continuous-Control SAC Brain Step
   * Evaluates state observation s_t (126-dim), samples 7D continuous action a_t,
   * performs full tactical ability execution (Repel, Dash, Barrier, Trench),
   * computes reward accounting R_t with PBRS & Curriculum Staging, and executes off-policy SGD updates.
   */
  private stepSACAgent(dt: number) {
    if (this.sacActionHoldCounter <= 0) {
      const nextObs = this.getSnakeObservation();

      // If we have a valid previous transition, compute reward and store in Replay Buffer
      if (this.sacPreviousState !== null) {
        const py = this.player.position.y ?? 0.55;
        const centroidY = this.swarmField.swarmCentroid.y ?? 3.5;
        const altDelta = Math.abs(py - centroidY);
        let altBonus = 0;
        if (altDelta > 3.5) {
          altBonus = (this.config.sacRewardAltitudeBonus ?? 0.25) * 1.2;
        }

        // Action entropy / diversity bonus if agent is utilizing varied tactical dimensions
        let diversityBonus = 0;
        const abilityIntents = [
          Math.abs(this.sacCurrentAction[3]),
          Math.abs(this.sacCurrentAction[4]),
          Math.abs(this.sacCurrentAction[5]),
          Math.abs(this.sacCurrentAction[6]),
        ];
        const activeAbilities = abilityIntents.filter(v => v > 0.25).length;
        if (activeAbilities > 0) {
          diversityBonus += (this.config.sacRewardTacticalBonus ?? 0.35) * (activeAbilities / 4.0);
        }

        // Spear evasion bonus: reward ascending or dodging away while spear is formed/thrusting
        let spearEvasionBonus = 0;
        const attackSt = this.swarmField.attackState ?? 0;
        if (attackSt === 1 || attackSt === 2) {
          // If spear is active and player maintains vertical altitude separation or high angular deviation
          if (altDelta > 3.0 || this.sacCurrentAction[1] > 0.3) {
            spearEvasionBonus = 0.65;
          }
        }

        // Cluster bait bonus: reward luring large groups before repelling
        let clusterBaitBonus = 0;
        if (this.sacStepRepelled >= 6) {
          clusterBaitBonus = 0.75;
        }

        // Encirclement loop trapping bonus
        const windingAreaNorm = nextObs[124] ?? 0;
        let encirclementBonus = 0;
        if (windingAreaNorm > 0.15 && this.player.continuousLength > 20) {
          encirclementBonus = windingAreaNorm * 0.8;
        }

        // Compute Potential-Based Reward Shaping (PBRS)
        const pbrsShaping = this.sacAgent.computePBRS(
          this.sacPreviousState,
          nextObs,
          this.sacAgent.gamma
        );

        let reward = this.sacAgent.calculateReward(
          this.sacOldLength,
          this.player.continuousLength,
          this.sacCurrentAction,
          this.sacPreviousAction,
          this.config.sacLengthScale ?? 50.0,
          this.config.sacRewardVariant ?? 'variant_c_combined',
          this.config.sacAlphaWeight ?? 1.0,
          this.config.sacBetaWeight ?? 1.0,
          this.config.sacLambdaSmoothness ?? 0.005,
          {
            hitsCount: this.player.lastStepHits,
            repelledCount: this.sacStepRepelled,
            dashExecuted: this.sacStepDash,
            barrierPlaced: this.sacStepBarrier,
            trenchCarved: this.sacStepTrench,
            altitudeClearanceBonus: altBonus,
            actionDiversityBonus: diversityBonus,
            spearEvasionBonus,
            clusterBaitBonus,
            encirclementBonus,
            pbrsShaping,
          }
        );

        this.player.lastStepReward = reward;

        // Reset step action flags
        this.sacStepRepelled = 0;
        this.sacStepDash = false;
        this.sacStepBarrier = false;
        this.sacStepTrench = false;

        // Store transition in SAC Replay Buffer (done = false for continuous non-terminating task)
        this.sacAgent.storeExperience(
          this.sacPreviousState,
          this.sacCurrentAction,
          reward,
          nextObs,
          false
        );

        // Perform SAC policy and critic neural network update
        this.sacAgent.learn(this.config.sacBatchSize ?? 128);
      }

      // Choose next 7D continuous action vector a_t in [-1, 1]^7
      const isEval = this.config.sacIsEvaluation ?? false;
      const sample = this.sacAgent.chooseAction(nextObs, isEval);

      // Exponential Moving Average (EMA) action filter for butter-smooth trajectory
      for (let i = 0; i < 7; i++) {
        this.sacSmoothedAction[i] += (sample.action[i] - this.sacSmoothedAction[i]) * Math.min(1.0, dt * 10.0);
      }

      this.sacPreviousAction.set(this.sacCurrentAction);
      this.sacCurrentAction.set(sample.action);
      this.sacPreviousState = nextObs;
      this.sacOldLength = this.player.continuousLength;

      // =========================================================================
      // SAC TACTICAL ABILITY DISPATCHER: Connect ALL Snake Abilities with RL!
      // a[3]: Repel Intent | a[4]: Dash Intent | a[5]: Barrier Intent | a[6]: Trench Intent
      // =========================================================================
      const aRepel = sample.action[3];
      const aDash = sample.action[4];
      const aBarrier = sample.action[5];
      const aTrench = sample.action[6];

      const px = this.player.position.x;
      const pz = this.player.position.z;
      const activeCount = this.swarmField.count;
      const centroid = this.swarmField.swarmCentroid;
      const toCentroidX = centroid.x - px;
      const toCentroidZ = centroid.z - pz;
      const centroidDist = Math.hypot(toCentroidX, toCentroidZ) || 1.0;

      let boidsInRepelZone = 0;
      let boidsNearCount = 0;
      for (let i = 0; i < activeCount; i++) {
        const d = Math.hypot(this.swarmField.boidX[i] - px, this.swarmField.boidZ[i] - pz);
        if (d < this.player.repelRadius) boidsInRepelZone++;
        if (d < 7.0) boidsNearCount++;
      }

      // 1. Kinetic Repel Blast Trigger (a[3])
      if (aRepel > 0.22 && this.player.repelCooldown <= 0 && this.abilityDebounce.repel <= 0) {
        const repelled = this.manualRepel();
        if (repelled > 0) {
          this.abilityDebounce.repel = 0.40;
          this.sacStepRepelled += repelled;
          this.sacAgent.abilityExecutions.repels++;
          this.stats.repelCount++;
          this.stats.totalRepelled += repelled;
        }
      }

      // 2. Surge Dash Trigger (a[4])
      if (aDash > 0.30 && this.player.dashCooldown <= 0 && this.abilityDebounce.dash <= 0) {
        if (centroidDist < 16.0 || boidsNearCount > 0 || aDash > 0.70) {
          this.abilityDebounce.dash = 0.45;
          const steerX = sample.action[0];
          const steerZ = sample.action[2];
          const steerLen = Math.hypot(steerX, steerZ);
          if (steerLen > 0.3) {
            this.player.dashDirection = {
              x: steerX / steerLen,
              y: 0,
              z: steerZ / steerLen,
            };
          } else {
            const invDist = 1.0 / centroidDist;
            this.player.dashDirection = {
              x: -toCentroidX * invDist,
              y: 0,
              z: -toCentroidZ * invDist,
            };
          }
          this.manualDash();
          this.sacStepDash = true;
          this.sacAgent.abilityExecutions.dashes++;
        }
      }

      // 3. Tactical Trench Carving (a[6])
      if (aTrench > 0.35 && (this.player.position.y ?? 0.55) < 5.5 && this.abilityDebounce.trench <= 0) {
        if (boidsNearCount > 0 || aTrench > 0.70) {
          this.abilityDebounce.trench = 0.50;
          this.carveTrench();
          this.sacStepTrench = true;
          this.sacAgent.abilityExecutions.trenches++;
        }
      }

      // Reset action persistence hold counter (default 4 physics ticks)
      this.sacActionHoldCounter = Math.max(1, this.config.sacActionPersistence ?? 4);
      this.sacRolloutStepCount++;

      // Periodic rollout window reset for long-term metrics tracking
      const maxRollout = this.config.sacRolloutSteps ?? 5000;
      if (this.sacRolloutStepCount >= maxRollout) {
        this.sacRolloutStepCount = 0;
        this.sacAgent.resetTrajectory(this.player.snakeLength);
      }
    } else {
      this.sacActionHoldCounter--;
    }
  }

  private updatePlayerPhysics(dt: number) {
    // Ability Debounce timers
    if (this.abilityDebounce.repel > 0) this.abilityDebounce.repel -= dt;
    if (this.abilityDebounce.dash > 0) this.abilityDebounce.dash -= dt;
    if (this.abilityDebounce.barrier > 0) this.abilityDebounce.barrier -= dt;
    if (this.abilityDebounce.trench > 0) this.abilityDebounce.trench -= dt;

    if (this.player.repelCooldown > 0) {
      this.player.repelCooldown = Math.max(0, this.player.repelCooldown - dt);
    }
    if (this.player.dashCooldown > 0) {
      this.player.dashCooldown = Math.max(0, this.player.dashCooldown - dt);
    }
    if (this.player.flashTimer > 0) {
      this.player.flashTimer = Math.max(0, this.player.flashTimer - dt);
    }
    if (this.player.damageFlashTimer > 0) {
      this.player.damageFlashTimer = Math.max(0, this.player.damageFlashTimer - dt);
    }
    if (this.player.growthPulseTimer > 0) {
      this.player.growthPulseTimer = Math.max(0, this.player.growthPulseTimer - dt);
    }
    if (this.player.lastShortenedTimer > 0) {
      this.player.lastShortenedTimer = Math.max(0, this.player.lastShortenedTimer - dt);
    }

    // Dash execution
    if (this.player.isDashing) {
      this.player.dashTimer -= dt;
      const dashSpd = this.config.playerDashSpeed;
      this.player.velocity.x = this.player.dashDirection.x * dashSpd;
      this.player.velocity.z = this.player.dashDirection.z * dashSpd;

      if (this.player.dashTimer <= 0) {
        this.player.isDashing = false;
      }
    }

    // =========================================================================
    // CONTINUOUS 360° OMNIDIRECTIONAL & 3D FLIGHT STEERING KINEMATICS
    // =========================================================================
    const omegaMax = this.config.maxTurnRate ?? (Math.PI / 2);
    const forwardVel = this.player.isDashing
      ? this.config.playerDashSpeed
      : (this.config.forwardVelocity ?? this.config.playerSpeed ?? 5.4);
    const vSpeed = this.config.playerVerticalSpeed ?? 5.0;
    const minAltitude = this.config.snakeMinAltitude ?? 0.45;
    const maxAltitude = this.config.snakeMaxAltitude ?? 36.0;
    const minCruisingSpeed = Math.max(3.6, forwardVel * 0.75); // Absolute non-zero velocity floor

    // Dynamic steering toward target heading (applicable to both RL agent and Manual Player)
    if (this.player.targetHeading !== undefined) {
      // Calculate shortest angular delta in [-PI, PI]
      let angleDiff = this.player.targetHeading - this.player.heading;
      while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
      while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;

      const maxAngleStep = omegaMax * dt;
      if (Math.abs(angleDiff) > 1e-3) {
        const rawSteer = angleDiff / Math.max(1e-4, maxAngleStep);
        this.player.steeringInput = Math.max(-1.0, Math.min(1.0, rawSteer));
      } else {
        this.player.steeringInput = 0;
      }
    }

    // Active Arena Boundary Avoidance (Ensures snake smoothly curves away before reaching perimeter)
    const boundLimit = 24.5;
    const wallMargin = 5.8;
    const px = this.player.position.x;
    const pz = this.player.position.z;

    if (Math.abs(px) > (boundLimit - wallMargin) || Math.abs(pz) > (boundLimit - wallMargin)) {
      // Direct heading back toward arena center (0, 0)
      const toCenterX = -px;
      const toCenterZ = -pz;
      const centerAngle = Math.atan2(toCenterX, toCenterZ);
      
      let wallAngleDiff = centerAngle - this.player.heading;
      while (wallAngleDiff < -Math.PI) wallAngleDiff += Math.PI * 2;
      while (wallAngleDiff > Math.PI) wallAngleDiff -= Math.PI * 2;

      const steerCorrection = Math.max(-1.0, Math.min(1.0, wallAngleDiff / (omegaMax * dt * 1.5)));
      this.player.steeringInput = steerCorrection;
    }

    // Angular velocity omega_t = a_t * omega_max
    this.player.angularVelocity = this.player.steeringInput * omegaMax;

    // Heading update: theta_{t+1} = wrap(theta_t + omega_t * dt)
    this.player.heading += this.player.angularVelocity * dt;
    while (this.player.heading < -Math.PI) this.player.heading += Math.PI * 2;
    while (this.player.heading > Math.PI) this.player.heading -= Math.PI * 2;

    this.player.facingAngle = this.player.heading;

    // -------------------------------------------------------------------------
    // 3D Altitudinal Ground Clearance & Obstacle Detection
    // -------------------------------------------------------------------------
    let baseGroundY = minAltitude;
    for (let b = 0; b < this.terrainDeformations.length; b++) {
      const deform = this.terrainDeformations[b];
      if (deform.heightChange > 0) {
        const halfW = (deform.width ?? deform.cubeSize) * 0.5 + 0.3;
        const halfD = (deform.depth ?? deform.cubeSize) * 0.5 + 0.3;
        if (
          Math.abs(this.player.position.x - deform.point.x) < halfW &&
          Math.abs(this.player.position.z - deform.point.z) < halfD
        ) {
          const cubeTop = (deform.baseY ?? 0) + (deform.height ?? deform.cubeSize) + 0.45;
          if (cubeTop > baseGroundY) {
            baseGroundY = cubeTop;
          }
        }
      }
    }

    // -------------------------------------------------------------------------
    // 3D Altitude, Pitch, and Vertical Flight Dynamics
    // -------------------------------------------------------------------------
    if (this.player.targetPitch !== undefined) {
      const pitchDelta = this.player.targetPitch - this.player.pitch;
      this.player.pitch += pitchDelta * Math.min(1.0, dt * 7.0);
      this.player.velocity.y = Math.sin(this.player.pitch) * vSpeed * 1.2;
    } else if (this.player.targetAltitude !== undefined) {
      const dy = this.player.targetAltitude - this.player.position.y;
      const targetVy = Math.max(-vSpeed, Math.min(vSpeed, dy * 3.5));
      this.player.velocity.y += (targetVy - this.player.velocity.y) * Math.min(1.0, dt * 8.0);
      this.player.pitch = Math.atan2(this.player.velocity.y, forwardVel);
    } else if (this.player.controlMode === 'rl_agent') {
      // Autonomous RL Mode: 3D cruising with harmonic altitude waves & tactical aerial evasions
      const organicWave = Math.sin(this.totalElapsedTime * 1.5 + this.player.walkCycle * 0.25) * 2.2;
      const baseTargetY = baseGroundY + 1.6 + Math.max(0, organicWave);

      let altitudeAdjustment = 0;
      const centroid = this.swarmField.swarmCentroid;
      const distToCentroid = Math.hypot(centroid.x - this.player.position.x, centroid.z - this.player.position.z);
      if (distToCentroid < 11.0) {
        // Snake dynamically climbs into the sky to soar over the oncoming swarm
        altitudeAdjustment = 4.5;
      }

      const desiredY = Math.max(baseGroundY, Math.min(maxAltitude - 2.0, baseTargetY + altitudeAdjustment));
      const dy = desiredY - this.player.position.y;
      const targetVy = Math.max(-vSpeed, Math.min(vSpeed, dy * 3.0));
      this.player.velocity.y += (targetVy - this.player.velocity.y) * Math.min(1.0, dt * 6.0);
      this.player.pitch = Math.atan2(this.player.velocity.y, forwardVel);
    } else {
      // Manual Player Mode: Smoothly level out pitch and vertical speed, holding current altitude
      this.player.velocity.y *= Math.pow(0.82, dt * 30);
      this.player.pitch *= Math.pow(0.82, dt * 30);
    }

    // 3D Velocity Vector handling
    if (this.player.controlMode === 'rl_agent' && this.config.sacEnabled) {
      // SAC Continuous Control: 3D Velocity Vector guided by smoothed actor policy
      const vMax = Math.max(4.5, this.config.sacMaxSpeed ?? this.config.playerSpeed ?? 7.5);
      const vVert = Math.max(3.5, this.config.playerVerticalSpeed ?? 6.0);

      // SAC action vector [ax, ay, az] filtered with EMA
      let ax = this.sacSmoothedAction[0];
      let ay = this.sacSmoothedAction[1];
      let az = this.sacSmoothedAction[2];

      // Sanitize against NaN / Infinity
      if (!Number.isFinite(ax)) ax = Math.sin(this.player.heading);
      if (!Number.isFinite(ay)) ay = 0;
      if (!Number.isFinite(az)) az = Math.cos(this.player.heading);

      // Active Soft Wall Repulsion in SAC Mode (smooth parabolic push away from boundaries)
      let wallPushX = 0;
      let wallPushZ = 0;
      if (this.player.position.x > boundLimit - wallMargin) {
        wallPushX = -Math.pow((this.player.position.x - (boundLimit - wallMargin)) / wallMargin, 1.4);
      } else if (this.player.position.x < -boundLimit + wallMargin) {
        wallPushX = Math.pow((-boundLimit + wallMargin - this.player.position.x) / wallMargin, 1.4);
      }

      if (this.player.position.z > boundLimit - wallMargin) {
        wallPushZ = -Math.pow((this.player.position.z - (boundLimit - wallMargin)) / wallMargin, 1.4);
      } else if (this.player.position.z < -boundLimit + wallMargin) {
        wallPushZ = Math.pow((-boundLimit + wallMargin - this.player.position.z) / wallMargin, 1.4);
      }

      if (wallPushX !== 0 || wallPushZ !== 0) {
        ax += wallPushX * 2.2;
        az += wallPushZ * 2.2;
      }

      // Ensure snake never stalls or freezes: cruise smoothly
      const actHorizLen = Math.hypot(ax, az);
      if (actHorizLen < 0.15 || !Number.isFinite(actHorizLen)) {
        const toCenterX = -this.player.position.x;
        const toCenterZ = -this.player.position.z;
        const toCenterDist = Math.hypot(toCenterX, toCenterZ);
        if (toCenterDist > 4.0) {
          ax = toCenterX / toCenterDist;
          az = toCenterZ / toCenterDist;
        } else {
          ax = Math.sin(this.player.heading);
          az = Math.cos(this.player.heading);
        }
      } else {
        ax /= actHorizLen;
        az /= actHorizLen;
      }

      // Add harmonic 3D vertical wave to encourage volumetric aerial exploration
      const waveY = Math.sin(this.totalElapsedTime * 1.4 + this.player.walkCycle * 0.22) * 1.0;

      const targetVx = ax * Math.max(minCruisingSpeed, vMax);
      const targetVy = Math.max(-vVert, Math.min(vVert, ay * vVert + waveY));
      const targetVz = az * Math.max(minCruisingSpeed, vMax);

      // Smooth inertia damping for velocity transition
      this.player.velocity.x += (targetVx - this.player.velocity.x) * Math.min(1.0, dt * 6.5);
      this.player.velocity.y += (targetVy - this.player.velocity.y) * Math.min(1.0, dt * 6.0);
      this.player.velocity.z += (targetVz - this.player.velocity.z) * Math.min(1.0, dt * 6.5);

      this.sacAgent.currentVelocity = [this.player.velocity.x, this.player.velocity.y, this.player.velocity.z];

      // Smooth continuous rotational steering without 180-degree instant snapping
      const horizSpeed = Math.hypot(this.player.velocity.x, this.player.velocity.z);
      if (horizSpeed > 0.08) {
        const targetHeading = Math.atan2(this.player.velocity.x, this.player.velocity.z);
        let diffAngle = targetHeading - this.player.heading;
        while (diffAngle < -Math.PI) diffAngle += Math.PI * 2;
        while (diffAngle > Math.PI) diffAngle -= Math.PI * 2;

        const maxTurnStep = (this.config.maxTurnRate ?? (Math.PI * 1.2)) * dt * 2.0;
        const turnStep = Math.max(-maxTurnStep, Math.min(maxTurnStep, diffAngle * Math.min(1.0, dt * 8.0)));
        this.player.heading += turnStep;
        while (this.player.heading < -Math.PI) this.player.heading += Math.PI * 2;
        while (this.player.heading > Math.PI) this.player.heading -= Math.PI * 2;
        this.player.facingAngle = this.player.heading;
      }

      const targetPitch = Math.atan2(this.player.velocity.y, Math.max(0.1, horizSpeed));
      this.player.pitch += (targetPitch - this.player.pitch) * Math.min(1.0, dt * 6.5);
      this.player.steeringInput = this.sacSmoothedAction[0];
    } else {
      // Kinematic Heading & Pitch formulation for Manual Player or legacy Q-learning
      const horizSpeed = Math.max(minCruisingSpeed, forwardVel * Math.cos(this.player.pitch));
      if (!this.player.isDashing) {
        this.player.velocity.x = Math.sin(this.player.heading) * horizSpeed;
        this.player.velocity.z = Math.cos(this.player.heading) * horizSpeed;
      }
    }

    // Move Snake Head in Full 3D Space
    this.player.position.x += this.player.velocity.x * dt;
    this.player.position.y += this.player.velocity.y * dt;
    this.player.position.z += this.player.velocity.z * dt;

    // Altitude Clamping & Ground Clearance
    if (this.player.position.y < baseGroundY) {
      this.player.position.y = baseGroundY;
      if (this.player.velocity.y < 0) this.player.velocity.y = 0;
    } else if (this.player.position.y > maxAltitude) {
      this.player.position.y = maxAltitude;
      if (this.player.velocity.y > 0) this.player.velocity.y = 0;
    }

    this.player.altitude = this.player.position.y;
    this.player.verticalVelocity = this.player.velocity.y;

    // Smooth soft boundary containment without instant violent bounce flipping
    if (this.player.position.x >= boundLimit) {
      this.player.position.x = boundLimit;
      if (this.player.velocity.x > 0) {
        this.player.velocity.x = -Math.max(minCruisingSpeed * 0.7, Math.abs(this.player.velocity.x) * 0.6);
      }
      this.sacSmoothedAction[0] = -0.85;
    } else if (this.player.position.x <= -boundLimit) {
      this.player.position.x = -boundLimit;
      if (this.player.velocity.x < 0) {
        this.player.velocity.x = Math.max(minCruisingSpeed * 0.7, Math.abs(this.player.velocity.x) * 0.6);
      }
      this.sacSmoothedAction[0] = 0.85;
    }

    if (this.player.position.z >= boundLimit) {
      this.player.position.z = boundLimit;
      if (this.player.velocity.z > 0) {
        this.player.velocity.z = -Math.max(minCruisingSpeed * 0.7, Math.abs(this.player.velocity.z) * 0.6);
      }
      this.sacSmoothedAction[2] = -0.85;
    } else if (this.player.position.z <= -boundLimit) {
      this.player.position.z = -boundLimit;
      if (this.player.velocity.z < 0) {
        this.player.velocity.z = Math.max(minCruisingSpeed * 0.7, Math.abs(this.player.velocity.z) * 0.6);
      }
      this.sacSmoothedAction[2] = 0.85;
    }

    // Keep heading wrapped [-PI, PI]
    while (this.player.heading < -Math.PI) this.player.heading += Math.PI * 2;
    while (this.player.heading > Math.PI) this.player.heading -= Math.PI * 2;

    // Autonomous Anti-Stall Watchdog: Rapidly detect and resolve stalls or stationary states
    this.unstuckTimer += dt;
    if (this.unstuckTimer >= 0.6) {
      const moveDist = Math.hypot(
        this.player.position.x - this.lastUnstuckPos.x,
        this.player.position.z - this.lastUnstuckPos.z
      );
      // If moving < 0.6m in 0.6s, the snake is stalled or blocked
      if (moveDist < 0.6) {
        this.stuckDuration += this.unstuckTimer;
        if (this.stuckDuration >= 0.6) {
          // Smooth Unstuck Impulse: propel toward open arena space with elevation vault
          const toCenterX = -this.player.position.x;
          const toCenterZ = -this.player.position.z;
          const centerDist = Math.hypot(toCenterX, toCenterZ);
          const centerAngle = centerDist > 1.0 ? Math.atan2(toCenterX, toCenterZ) : this.player.heading + 0.8;
          const escapeAngle = centerAngle + (Math.random() - 0.5) * 0.5;

          const escX = Math.sin(escapeAngle);
          const escZ = Math.cos(escapeAngle);
          const boostSpeed = Math.max(forwardVel * 1.15, minCruisingSpeed * 1.25, 6.0);

          this.player.velocity.x = escX * boostSpeed;
          this.player.velocity.z = escZ * boostSpeed;
          this.player.velocity.y = Math.max(this.player.velocity.y, 2.5); // Vault smoothly over terrain obstacles

          this.sacSmoothedAction[0] = escX;
          this.sacSmoothedAction[1] = 0.35;
          this.sacSmoothedAction[2] = escZ;
          this.player.heading = escapeAngle;
          this.player.facingAngle = escapeAngle;
          this.stuckDuration = 0;
        }
      } else {
        this.stuckDuration = 0;
      }

      this.lastUnstuckPos = {
        x: this.player.position.x,
        y: this.player.position.y,
        z: this.player.position.z,
      };
      this.unstuckTimer = 0;
    }

    // Absolute Coordinate & Velocity Finite Guard
    if (
      !Number.isFinite(this.player.position.x) ||
      !Number.isFinite(this.player.position.y) ||
      !Number.isFinite(this.player.position.z)
    ) {
      this.player.position.x = 0;
      this.player.position.y = 1.2;
      this.player.position.z = 0;
      this.player.velocity.x = 0;
      this.player.velocity.y = 0;
      this.player.velocity.z = forwardVel;
      this.player.heading = 0;
    }

    // Continuous undulation walk cycle in 3D (gentle, organic frequency)
    const moveLen3D = Math.hypot(this.player.velocity.x, this.player.velocity.y, this.player.velocity.z);
    this.player.walkCycle += Math.max(minCruisingSpeed, moveLen3D) * dt * 2.6;

    // Continuous Length Dynamics: l_{t+1} = l_t + g * dt
    const gRate = this.config.growthRate ?? 0.5;
    this.player.continuousLength = Math.min(
      this.player.snakeMaxCap,
      this.player.continuousLength + gRate * dt
    );
    this.player.snakeLength = Math.max(
      this.config.snakeMinLength ?? 10,
      Math.round(this.player.continuousLength)
    );
    if (this.player.snakeLength > this.player.peakLength) {
      this.player.peakLength = this.player.snakeLength;
    }

    // =========================================================================
    // 3D BODY POSITION HISTORY & SERPENTINE INVERSE KINEMATICS (IK)
    // =========================================================================
    if (!this.player.bodyHistory) this.player.bodyHistory = [];
    this.player.bodyHistory.unshift({
      x: this.player.position.x,
      y: this.player.position.y,
      z: this.player.position.z,
    });
    const maxHistoryPoints = Math.max(10, Math.ceil(this.player.continuousLength * 6));
    if (this.player.bodyHistory.length > maxHistoryPoints) {
      this.player.bodyHistory.length = maxHistoryPoints;
    }

    const spacing = this.config.snakeSegmentSpacing ?? 0.42;
    const currentSegments = this.player.segments;
    const targetCount = this.player.snakeLength;

    while (currentSegments.length < targetCount) {
      const last = currentSegments[currentSegments.length - 1] || {
        x: this.player.position.x,
        y: this.player.position.y,
        z: this.player.position.z,
        angle: this.player.facingAngle,
        pitch: this.player.pitch,
      };
      currentSegments.push({
        x: last.x,
        y: last.y,
        z: last.z - spacing,
        angle: last.angle,
        pitch: last.pitch ?? 0,
      });
    }
    if (currentSegments.length > targetCount) {
      currentSegments.length = targetCount;
    }

    let prevX = this.player.position.x;
    let prevY = this.player.position.y;
    let prevZ = this.player.position.z;

    for (let i = 0; i < currentSegments.length; i++) {
      const seg = currentSegments[i];
      if (!seg) continue;
      const dx = seg.x - prevX;
      const dy = seg.y - prevY;
      const dz = seg.z - prevZ;
      const dist3D = Math.hypot(dx, dy, dz) || 0.001;

      const targetX = prevX + (dx / dist3D) * spacing;
      const targetY = prevY + (dy / dist3D) * spacing;
      const targetZ = prevZ + (dz / dist3D) * spacing;

      // Smooth segment following
      seg.x += (targetX - seg.x) * Math.min(1.0, dt * 22.0);
      seg.y += (targetY - seg.y) * Math.min(1.0, dt * 22.0);
      seg.z += (targetZ - seg.z) * Math.min(1.0, dt * 22.0);

      // Gentle local normal undulation (scaled organically)
      const wavePhase = this.player.walkCycle - i * 0.25;
      const waveAmp = (moveLen3D > 0.2 ? 0.035 : 0.012) * Math.sin(wavePhase);
      const segTangent = Math.atan2(prevX - seg.x, prevZ - seg.z);
      const normX = -Math.sin(segTangent);
      const normZ = Math.cos(segTangent);
      seg.x += normX * waveAmp * dt * 4.0;
      seg.z += normZ * waveAmp * dt * 4.0;

      if (seg.y < minAltitude) {
        seg.y = minAltitude;
      }

      prevX = seg.x;
      prevY = seg.y;
      prevZ = seg.z;
    }

    // =========================================================================
    // 3D SNAKE BODY SELF-INTERACTION & NON-PENETRATION VOLUMETRIC SOLVER (PBD)
    // Ensures the snake's body and head never clip or pass through itself
    // =========================================================================
    const growthFactor = Math.max(1.0, (this.player.snakeLength || 10) / 10);
    const thicknessScale = Math.pow(growthFactor, 0.18);
    const baseSegRadius = spacing * 0.55 * thicknessScale;
    const headColRadius = baseSegRadius * 1.35;

    // Helper to calculate realistic anatomical taper radius along the spine
    const getSegRadius = (idx: number, total: number) => {
      const u = idx / Math.max(1, total - 1);
      let taper = 1.0;
      if (u < 0.10) {
        taper = 0.88 + (u / 0.10) * 0.22; // Neck taper up to full girth
      } else if (u < 0.65) {
        taper = 1.10 - ((u - 0.10) / 0.55) * 0.15; // Thick mid-body
      } else {
        taper = 0.95 * Math.pow(1.0 - (u - 0.65) / 0.35, 0.70) + 0.22; // Taper to tail
      }
      return baseSegRadius * taper;
    };

    // Fast O(N) Position-Based Dynamics & Articulation Solver
    const pbdIterations = 2;
    for (let iter = 0; iter < pbdIterations; iter++) {
      // 1. Head-to-Body Volumetric Repulsion (Head slides and collides against coiled body in O(N))
      const segStride = currentSegments.length > 80 ? 2 : 1;
      for (let j = 4; j < currentSegments.length; j += segStride) {
        const segJ = currentSegments[j];
        const radJ = getSegRadius(j, currentSegments.length);
        const minHeadDist = headColRadius + radJ;

        const hdx = this.player.position.x - segJ.x;
        const hdy = this.player.position.y - segJ.y;
        const hdz = this.player.position.z - segJ.z;
        const headDistSq = hdx * hdx + hdy * hdy + hdz * hdz;

        if (headDistSq < minHeadDist * minHeadDist && headDistSq > 0.00001) {
          const headDist = Math.sqrt(headDistSq);
          const overlap = minHeadDist - headDist;
          const nx = hdx / headDist;
          const ny = hdy / headDist;
          const nz = hdz / headDist;

          // Push head away (60%) and push body segment away (40%)
          this.player.position.x += nx * overlap * 0.60;
          this.player.position.y += ny * overlap * 0.60;
          this.player.position.z += nz * overlap * 0.60;

          segJ.x -= nx * overlap * 0.40;
          segJ.y -= ny * overlap * 0.40;
          segJ.z -= nz * overlap * 0.40;

          // Deflect head velocity smoothly along the body surface
          const vDotN = this.player.velocity.x * nx + this.player.velocity.y * ny + this.player.velocity.z * nz;
          if (vDotN < 0) {
            this.player.velocity.x -= vDotN * nx * 0.90;
            this.player.velocity.y -= vDotN * ny * 0.90;
            this.player.velocity.z -= vDotN * nz * 0.90;
          }
        }
      }

      // 2. Vertebral Distance & Articulation Anchor Propagation (O(N))
      let anchorX = this.player.position.x;
      let anchorY = this.player.position.y;
      let anchorZ = this.player.position.z;

      for (let i = 0; i < currentSegments.length; i++) {
        const seg = currentSegments[i];
        const dx = seg.x - anchorX;
        const dy = seg.y - anchorY;
        const dz = seg.z - anchorZ;
        const dist = Math.hypot(dx, dy, dz) || 0.001;

        if (Math.abs(dist - spacing) > 0.001) {
          const correction = (dist - spacing) / dist;
          seg.x -= dx * correction * 0.85;
          seg.y -= dy * correction * 0.85;
          seg.z -= dz * correction * 0.85;
        }

        if (seg.y < minAltitude) {
          seg.y = minAltitude;
        }

        anchorX = seg.x;
        anchorY = seg.y;
        anchorZ = seg.z;
      }
    }

    // Final Segment Rotation and Angle Smoothing
    let angPrevX = this.player.position.x;
    let angPrevY = this.player.position.y;
    let angPrevZ = this.player.position.z;

    for (let i = 0; i < currentSegments.length; i++) {
      const seg = currentSegments[i];
      const hDist = Math.hypot(angPrevX - seg.x, angPrevZ - seg.z) || 0.001;
      const targetSegAngle = Math.atan2(angPrevX - seg.x, angPrevZ - seg.z);
      let dSegAngle = targetSegAngle - (seg.angle ?? targetSegAngle);
      while (dSegAngle < -Math.PI) dSegAngle += Math.PI * 2;
      while (dSegAngle > Math.PI) dSegAngle -= Math.PI * 2;
      seg.angle = (seg.angle ?? targetSegAngle) + dSegAngle * Math.min(1.0, dt * 18.0);

      const targetSegPitch = Math.atan2(angPrevY - seg.y, hDist);
      seg.pitch = (seg.pitch ?? targetSegPitch) + (targetSegPitch - (seg.pitch ?? targetSegPitch)) * Math.min(1.0, dt * 18.0);

      angPrevX = seg.x;
      angPrevY = seg.y;
      angPrevZ = seg.z;
    }

    // =========================================================================
    // 3-SECOND MILESTONE REWARD PULSE
    // =========================================================================
    this.player.elongateTimer += dt;
    if (this.player.elongateTimer >= this.player.elongateInterval) {
      this.player.elongateTimer = 0;

      if (this.player.snakeLength < this.player.snakeMaxCap) {
        const growAmt = Math.min(
          this.player.growthAmount,
          this.player.snakeMaxCap - this.player.snakeLength
        );
        this.player.growthPulseTimer = 0.8;
        this.rlAccumulatedReward += this.config.rlRewardGrowth;

        if (this.config.soundEnabled) {
          simulationSound.playRepelBlast();
        }
      }
    }

    // =========================================================================
    // SNAKE HEAD ARTICULATION & 3D EYE GAZE TRACKING
    // =========================================================================
    const headX = this.player.position.x;
    const headY = this.player.position.y;
    const headZ = this.player.position.z;

    // 1. Find closest swarm boid for ocular saccade 3D gaze tracking
    let closestBoidDist = 999.0;
    let closestBoidIdx = -1;
    const activeBoidCount = this.swarmField.count;

    for (let i = 0; i < activeBoidCount; i++) {
      const bx = this.swarmField.boidX[i];
      const by = this.swarmField.boidY[i];
      const bz = this.swarmField.boidZ[i];
      const dist = Math.hypot(bx - headX, by - headY, bz - headZ);
      if (dist < closestBoidDist) {
        closestBoidDist = dist;
        closestBoidIdx = i;
      }
    }

    if (closestBoidIdx >= 0 && closestBoidDist < 40.0) {
      this.player.eyeTargetPos = {
        x: this.swarmField.boidX[closestBoidIdx],
        y: this.swarmField.boidY[closestBoidIdx],
        z: this.swarmField.boidZ[closestBoidIdx],
      };
    } else {
      this.player.eyeTargetPos = null;
    }

    // 2. Head State Machine Evaluation
    if (this.player.damageFlashTimer > 0) {
      this.player.headState = 'damage_recoil';
      this.player.targetJawAngle = 0.45;
    } else if (this.player.isDashing) {
      this.player.headState = 'strike_lunge';
      this.player.targetJawAngle = 0.60;
    } else if (closestBoidDist < 7.5) {
      this.player.headState = 'hunting_track';
      this.player.targetJawAngle = 0.20;
    } else {
      this.player.headState = 'idle_cruise';
      this.player.targetJawAngle = 0.04;
    }

    // 3. Smooth Jaw Articulation Kinematics
    this.player.jawAngle += (this.player.targetJawAngle - this.player.jawAngle) * Math.min(1.0, dt * 18.0);

    // 4. Kinetic Crown Charge Passive Dissipation
    this.player.kineticCrownCharge = Math.max(0, this.player.kineticCrownCharge - dt * 0.025);
  }

  /**
    * Q-Learning Brain Tick for Red Snake Survival & Length Maximization
    * Continuous Observation Vector s_t: [l_t, theta_t, delta_x_1, delta_z_1, delta_vx_1, delta_vz_1, ...]
    */
  private stepRLAgent(dt: number) {
    this.rlDecisionTimer += dt;
    if (this.rlDecisionTimer < this.config.rlDecisionInterval) {
      return;
    }

    this.rlDecisionTimer = 0;

    // 1. Calculate Continuous Environment Observables
    const px = this.player.position.x;
    const pz = this.player.position.z;
    const pvx = this.player.velocity.x;
    const pvz = this.player.velocity.z;
    const activeCount = this.swarmField.count;
    const nearestCount = this.config.nearestEnemiesCount ?? 3;

    // Zero-allocation tracking for nearest boids
    const topKDist = [Infinity, Infinity, Infinity];
    const topKIdx = [-1, -1, -1];
    let nearestBoidDist = 999.0;
    let localBoidsNear = 0;
    let boidsInRepelRadius = 0;

    for (let i = 0; i < activeCount; i++) {
      const bx = this.swarmField.boidX[i];
      const bz = this.swarmField.boidZ[i];
      const dx = bx - px;
      const dz = bz - pz;
      const dist = Math.hypot(dx, dz);

      if (dist < nearestBoidDist) {
        nearestBoidDist = dist;
      }
      if (dist < 7.0) {
        localBoidsNear++;
      }
      if (dist < this.player.repelRadius) {
        boidsInRepelRadius++;
      }

      if (dist < topKDist[2]) {
        let ins = 2;
        while (ins > 0 && dist < topKDist[ins - 1]) {
          topKDist[ins] = topKDist[ins - 1];
          topKIdx[ins] = topKIdx[ins - 1];
          ins--;
        }
        topKDist[ins] = dist;
        topKIdx[ins] = i;
      }
    }

    // Formulate continuous state vector s_t
    const obsVector: number[] = [this.player.continuousLength, this.player.heading];
    for (let n = 0; n < nearestCount; n++) {
      const bIdx = topKIdx[n];
      if (bIdx >= 0 && bIdx < activeCount) {
        const bx = this.swarmField.boidX[bIdx];
        const bz = this.swarmField.boidZ[bIdx];
        const bvx = this.swarmField.boidVx[bIdx];
        const bvz = this.swarmField.boidVz[bIdx];
        obsVector.push(bx - px, bz - pz, bvx - pvx, bvz - pvz);
      } else {
        obsVector.push(999.0, 999.0, 0, 0);
      }
    }
    this.player.observationVector = obsVector;

    // Check if player is shielded by a raised terrain barrier
    const centroid = this.swarmField.swarmCentroid;
    const toCentroidX = centroid.x - px;
    const toCentroidZ = centroid.z - pz;
    const centroidDist = Math.hypot(toCentroidX, toCentroidZ) || 1.0;

    let isShielded = false;
    for (const b of this.terrainDeformations) {
      if (b.heightChange > 0) {
        const bdx = b.point.x - px;
        const bdz = b.point.z - pz;
        const bDist = Math.hypot(bdx, bdz);
        if (bDist < centroidDist) {
          const dot = (bdx * toCentroidX + bdz * toCentroidZ) / (bDist * centroidDist);
          if (dot > 0.65) {
            isShielded = true;
            break;
          }
        }
      }
    }

    // 2. Formulate State Key for Tabular/RL Brain (including 3D altitude state)
    const lengthRatio = this.player.snakeLength / this.player.snakeMaxCap;
    const { stateKey } = this.rlAgent.getStateKey(
      nearestBoidDist,
      localBoidsNear,
      this.player.repelCooldown <= 0,
      isShielded,
      lengthRatio > 0.05 ? 1.0 : 0.2,
      this.player.position.y ?? 0.55
    );

    // 3. Complete Previous Action's Bellman Update with exact Mathematical Reward Function:
    // r_t = (alpha * l_t) - (beta * k_t) - (gamma * d_t)
    if (this.rlPreviousAction !== null && this.rlPreviousStateKey !== '') {
      const alpha = this.config.rlAlphaLengthReward ?? 0.1;
      const beta = this.config.rlBetaHitPenalty ?? 10.0;
      const gamma = this.config.rlGammaTerminalPenalty ?? 100.0;
      const dtInterval = this.config.rlDecisionInterval;

      const continuousSurvivalReward = alpha * this.player.continuousLength * (dtInterval / 1.0);
      const hitDeduction = beta * this.player.lastStepHits;
      const deathDeduction = this.player.isDead ? gamma : 0;

      let stepReward = continuousSurvivalReward - hitDeduction - deathDeduction;
      if (nearestBoidDist > 6.0) {
        stepReward += 0.5; // Safe distancing bonus
      }

      // Vertical clearance bonus for Q-Learning agent
      const playerY = this.player.position.y ?? 0.55;
      const centroidY = centroid.y ?? 3.5;
      if (centroidDist < 12.0 && Math.abs(playerY - centroidY) > 3.5) {
        stepReward += 0.4;
      }

      stepReward += this.rlAccumulatedReward;
      this.rlAccumulatedReward = 0;
      this.player.lastStepReward = stepReward;

      this.rlAgent.learn(this.rlPreviousStateKey, this.rlPreviousAction, stepReward, stateKey);
    }

    // 4. Choose Next Action via Policy
    const { action } = this.rlAgent.chooseAction(stateKey);
    this.rlPreviousAction = action;
    this.rlPreviousStateKey = stateKey;

    // 5. Execute Action (steer continuously in 360° 3D space)
    this.executeRLAction(action, toCentroidX, toCentroidZ, centroidDist, boidsInRepelRadius);
  }

  private executeRLAction(
    action: RLAction,
    toCentroidX: number,
    toCentroidZ: number,
    centroidDist: number,
    boidsInRepelRadius: number
  ) {
    const invDist = 1.0 / Math.max(0.1, centroidDist);
    const dirX = toCentroidX * invDist;
    const dirZ = toCentroidZ * invDist;

    // Determine evasion vector: if swarm is close, evade threat; if far, roam smoothly
    const isSwarmClose = centroidDist < 14.0;
    const evadeAngle = isSwarmClose
      ? Math.atan2(-dirX, -dirZ)
      : this.player.heading + 0.25 * Math.sin(this.totalElapsedTime * 1.2);

    switch (action) {
      case 'evade_kite': {
        this.player.targetHeading = evadeAngle;
        if (isSwarmClose) {
          this.player.targetAltitude = 8.5; // Dynamic cruising altitude
        }
        break;
      }
      case 'aerial_soar': {
        // Steep 3D ascension to upper airspace (16.0 - 24.0m) to soar over ground/mid boid murmurations
        this.player.targetAltitude = 18.5 + 4.0 * Math.sin(this.totalElapsedTime * 0.8);
        this.player.targetPitch = 0.45; // Positive upward pitch
        this.player.targetHeading = evadeAngle + 0.35;
        this.rlAccumulatedReward += 0.35;
        break;
      }
      case 'ground_dive': {
        // Rapid altitudinal descent to hug terrain trenches and ground contours (1.0 - 2.2m)
        this.player.targetAltitude = 1.2;
        this.player.targetPitch = -0.45; // Downward dive pitch
        this.player.targetHeading = evadeAngle - 0.35;
        this.rlAccumulatedReward += 0.30;
        break;
      }
      case 'altitudinal_corkscrew': {
        // 3D spiral corkscrew maneuver: continuous tight banking while ascending/descending
        const spiralSpeed = 2.4;
        this.player.targetHeading = this.player.heading + spiralSpeed * 0.25;
        this.player.targetAltitude = 10.0 + 7.0 * Math.sin(this.totalElapsedTime * 2.2);
        this.player.targetPitch = 0.35 * Math.cos(this.totalElapsedTime * 2.2);
        this.rlAccumulatedReward += 0.40;
        break;
      }
      case 'flank_reposition': {
        const perpX = -dirZ;
        const perpZ = dirX;
        const flankAngle = isSwarmClose
          ? Math.atan2(perpX, perpZ)
          : this.player.heading + 0.8;
        this.player.targetHeading = flankAngle;
        if (isSwarmClose) {
          this.player.targetAltitude = 6.0;
        }
        break;
      }
      case 'blast_repel': {
        if (this.player.repelCooldown <= 0) {
          const repelled = this.manualRepel();
          if (repelled > 0) {
            this.stats.repelCount++;
            this.stats.totalRepelled += repelled;
            this.rlAccumulatedReward += (this.config.rlRewardRepelKill ?? 0.2) * repelled;
          } else {
            this.rlAccumulatedReward += (this.config.rlPenaltyWastedRepel ?? -1.5);
          }
        }
        this.player.targetHeading = evadeAngle;
        break;
      }
      case 'carve_trench': {
        if ((this.player.position.y ?? 0.55) < 5.0) {
          this.carveTrench();
          this.rlAccumulatedReward += 0.5;
        }
        this.player.targetHeading = evadeAngle;
        break;
      }
      case 'tactical_dash': {
        if (this.player.dashCooldown <= 0) {
          this.player.dashDirection = { x: -dirX, y: 0, z: -dirZ };
          this.manualDash();
          this.rlAccumulatedReward += 0.8;
        }
        this.player.targetHeading = evadeAngle;
        break;
      }
    }
  }

  /**
   * Check Swarm collisions with Snake:
   * Dynamic hit penalty c: l_{t+1} = max(minLen, l_t - c * k_t)
   */
  private checkSwarmCollisions(dt: number) {
    const px = this.player.position.x;
    const py = this.player.position.y ?? 0.55;
    const pz = this.player.position.z;
    const activeCount = this.swarmField.count;
    const hitRadiusSq = 1.15 * 1.15;

    let bitesThisFrame = 0;
    const segments = this.player.segments;
    const segCheckStride = Math.max(1, Math.floor(segments.length / 24));

    // Compute Snake Bounding Box with padding for fast O(1) boid rejection
    let minX = px;
    let maxX = px;
    let minY = py;
    let maxY = py;
    let minZ = pz;
    let maxZ = pz;

    for (let s = 0; s < segments.length; s += 4) {
      const seg = segments[s];
      if (!seg) continue;
      if (seg.x < minX) minX = seg.x;
      if (seg.x > maxX) maxX = seg.x;
      const sy = seg.y ?? 0.55;
      if (sy < minY) minY = sy;
      if (sy > maxY) maxY = sy;
      if (seg.z < minZ) minZ = seg.z;
      if (seg.z > maxZ) maxZ = seg.z;
    }

    const pad = 2.0;
    const bbMinX = minX - pad;
    const bbMaxX = maxX + pad;
    const bbMinY = minY - pad;
    const bbMaxY = maxY + pad;
    const bbMinZ = minZ - pad;
    const bbMaxZ = maxZ + pad;

    for (let i = activeCount - 1; i >= 0; i--) {
      const bx = this.swarmField.boidX[i];
      const by = this.swarmField.boidY[i];
      const bz = this.swarmField.boidZ[i];

      // Fast AABB rejection: If boid is outside snake's bounding volume, skip all collision tests
      if (bx < bbMinX || bx > bbMaxX || bz < bbMinZ || bz > bbMaxZ || by < bbMinY || by > bbMaxY) {
        continue;
      }

      // If Snake is dashing, it destroys boids with kinetic impact and decaying explosion
      if (this.player.isDashing) {
        const hdx = bx - px;
        const hdy = by - py;
        const hdz = bz - pz;
        if (hdx * hdx + hdy * hdy + hdz * hdz < hitRadiusSq * 2.2) {
          this.swarmField.damageBoid(i, 100.0, (deadX, deadY, deadZ, bvx, bvy, bvz) => {
            this.spawnDecayExplosion(deadX, deadY, deadZ, bvx, bvy, bvz, 16);
          });
          continue;
        }
      }

      // 1. Check 3D collision with Snake Head
      const hdx = bx - px;
      const hdy = by - py;
      const hdz = bz - pz;
      if (hdx * hdx + hdy * hdy + hdz * hdz < hitRadiusSq) {
        bitesThisFrame++;
        this.swarmField.boidVx[i] = hdx * 8.0;
        this.swarmField.boidVy[i] = Math.max(2.0, hdy * 6.0);
        this.swarmField.boidVz[i] = hdz * 8.0;
        continue;
      }

      // 2. Check 3D collision with Snake Body Segments
      for (let s = 0; s < segments.length; s += segCheckStride) {
        const seg = segments[s];
        if (!seg) continue;
        const sdx = bx - seg.x;
        const sdy = by - (seg.y ?? 0.55);
        const sdz = bz - seg.z;
        if (sdx * sdx + sdy * sdy + sdz * sdz < hitRadiusSq) {
          bitesThisFrame++;
          this.swarmField.boidVx[i] = sdx * 8.0;
          this.swarmField.boidVy[i] = Math.max(2.0, sdy * 6.0);
          this.swarmField.boidVz[i] = sdz * 8.0;
          break;
        }
      }
    }

    this.player.lastStepHits = bitesThisFrame;

    // Apply continuous length deduction: l_{t+1} = max(minLen, l_t - c * k_t)
    if (bitesThisFrame > 0 && !this.player.isDashing) {
      const minLength = this.config.snakeMinLength ?? 10;
      const hitPenaltyC = this.config.hitPenalty ?? 5.0;
      const rawDeduction = bitesThisFrame * (this.config.snakeShortenPerBite ?? 1) * (hitPenaltyC / 5.0);

      const oldContinuous = this.player.continuousLength;
      this.player.continuousLength = Math.max(minLength, this.player.continuousLength - rawDeduction);
      const actualShortened = Math.round(oldContinuous - this.player.continuousLength);

      this.player.snakeLength = Math.max(minLength, Math.round(this.player.continuousLength));
      this.player.segments.length = this.player.snakeLength;
      this.player.shortenCount += bitesThisFrame;
      this.player.lastShortenedTimer = 0.25;
      this.player.damageFlashTimer = 0.2;

      if (actualShortened > 0) {
        if (this.config.soundEnabled && Math.random() < 0.4) {
          simulationSound.playDamageHit();
        }
      }
    }
  }

  private updateEffects(dt: number) {
    // Shockwaves
    for (let i = this.shockwaves.length - 1; i >= 0; i--) {
      const sw = this.shockwaves[i];
      sw.life -= dt;
      sw.radius = sw.maxRadius * (1.0 - sw.life / sw.maxLife);
      if (sw.life <= 0) {
        this.shockwaves.splice(i, 1);
      }
    }

    // Particles & Decaying Creature Ash Dissolution
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.life -= dt;

      const drag = p.drag ?? 0.95;
      p.velocity.x *= drag;
      p.velocity.z *= drag;
      const gravity = p.gravity ?? 4.9;
      p.velocity.y -= gravity * dt;

      p.position.x += p.velocity.x * dt;
      p.position.y = Math.max(0.04, p.position.y + p.velocity.y * dt);
      p.position.z += p.velocity.z * dt;

      // Color decay simulation: Pale Grey (#b8bcc6) -> Slate Grey (#6b7280) -> Deep Soot Charcoal (#09090b)
      if (p.type === 'decay_creature_explosion') {
        const progress = Math.max(0, Math.min(1, p.life / p.maxLife));
        if (progress > 0.6) {
          const norm = (progress - 0.6) / 0.4;
          const r = Math.round(107 + norm * (184 - 107));
          const g = Math.round(114 + norm * (188 - 114));
          const b = Math.round(128 + norm * (198 - 128));
          p.color = `rgb(${r},${g},${b})`;
        } else if (progress > 0.25) {
          const norm = (progress - 0.25) / 0.35;
          const r = Math.round(39 + norm * (107 - 39));
          const g = Math.round(39 + norm * (114 - 39));
          const b = Math.round(42 + norm * (128 - 42));
          p.color = `rgb(${r},${g},${b})`;
        } else {
          const norm = progress / 0.25;
          const r = Math.round(9 + norm * (39 - 9));
          const g = Math.round(9 + norm * (39 - 9));
          const b = Math.round(11 + norm * (42 - 11));
          p.color = `rgb(${r},${g},${b})`;
        }
      }

      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }

    // Floating Numbers
    for (let i = this.floatingNumbers.length - 1; i >= 0; i--) {
      const fn = this.floatingNumbers[i];
      fn.life -= dt;
      fn.position.y += fn.velocity.y * dt;
      if (fn.life <= 0) {
        this.floatingNumbers.splice(i, 1);
      }
    }
  }

  private syncStats() {
    const px = this.player.position.x;
    const pz = this.player.position.z;
    const activeCount = this.swarmField.count;

    let boidsNear = 0;
    let boidsInBlast = 0;

    for (let i = 0; i < activeCount; i++) {
      const dist = Math.hypot(this.swarmField.boidX[i] - px, this.swarmField.boidZ[i] - pz);
      if (dist < 5.0) boidsNear++;
      if (dist < this.player.repelRadius) boidsInBlast++;
    }

    const nextElongate = Math.max(0, this.player.elongateInterval - this.player.elongateTimer);

    this.stats = {
      snakeLength: this.player.snakeLength,
      continuousLength: this.player.continuousLength,
      heading: this.player.heading,
      pitch: this.player.pitch,
      altitude: this.player.altitude,
      verticalVelocity: this.player.verticalVelocity,
      steeringInput: this.player.steeringInput,
      angularVelocity: this.player.angularVelocity,
      lastStepReward: this.player.lastStepReward,
      lastStepHits: this.player.lastStepHits,
      observationVector: this.player.observationVector,
      snakeMaxCap: this.player.snakeMaxCap,
      peakLength: this.player.peakLength,
      nextElongateCountdown: nextElongate,
      bitesTakenCount: this.player.shortenCount,
      playerHealth: this.player.health,
      playerMaxHealth: this.player.maxHealth,
      playerIsDead: false,
      playerSurvivalTime: this.player.survivalTime,
      bestSurvivalTime: Math.max(this.bestSurvivalTime, this.player.survivalTime),
      episodeNumber: this.episodeCount,
      playerScore: Math.floor(this.player.snakeLength * 10 + this.player.survivalTime * 5),
      playerControlMode: this.player.controlMode,
      playerRepelCooldownRemaining: this.player.repelCooldown,
      playerDashCooldownRemaining: this.player.dashCooldown,
      lastDamageEvent: null,
      activeBoids: activeCount,
      totalSpawned: activeCount,
      totalRepelled: this.stats.totalRepelled,
      repelCount: this.stats.repelCount,
      boidsNearPlayer: boidsNear,
      boidsInBlastZone: boidsInBlast,
      totalBarriersPlaced: this.stats.totalBarriersPlaced,
      totalTrenchesCarved: this.stats.totalTrenchesCarved,
      playerAttractionWeight: this.config.playerAttractionWeight,
      avgBoidSurvivalTime: this.swarmField.avgSurvivalTime,
      maxBoidSurvivalTime: this.swarmField.maxSurvivalTime,
      swarmAttackState: this.swarmField.attackState,
      swarmAttackStateName:
        this.swarmField.attackState === 3
          ? 'Dispersing / Burst'
          : this.swarmField.attackState === 2
          ? 'Thrusting Spear'
          : this.swarmField.attackState === 1
          ? 'Forming Spear'
          : 'Flocking Murmuration',
      swarmDemographics: { ...this.swarmField.demographics },
      sacMetrics: this.sacAgent.getMetrics(
        this.config.sacEnabled,
        this.config.sacIsEvaluation,
        this.config.sacRewardVariant
      ),
      rlAgent: this.rlAgent.getStats(this.config.rlEnabled),
      headTelemetry: {
        state: this.player.headState,
        jawAngleDeg: (this.player.jawAngle * 180) / Math.PI,
        eyeTargetDistance: this.player.eyeTargetPos
          ? Math.hypot(
              this.player.eyeTargetPos.x - this.player.position.x,
              this.player.eyeTargetPos.y - this.player.position.y,
              this.player.eyeTargetPos.z - this.player.position.z
            )
          : null,
        kineticCrownCharge: this.player.kineticCrownCharge,
        fangErectRatio: Math.min(1.0, this.player.jawAngle / 0.45),
        tongueDartSpeed: this.player.headState === 'hunting_track' ? 22.0 : 14.0,
      },
      elapsedTime: this.totalElapsedTime,
      fps: 60,
    };
  }

  /**
   * Export learned RL weights and configuration to a downloadable JSON file
   */
  public exportModelToFile(type: 'sac' | 'qtable' | 'all' = 'all') {
    let data: any;
    let filename = '';
    const dateStr = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    if (type === 'sac') {
      data = this.sacAgent.exportModelJSON();
      filename = `sac_continuous_model_${dateStr}.json`;
    } else if (type === 'qtable') {
      data = this.rlAgent.exportModelJSON();
      filename = `qtable_model_${dateStr}.json`;
    } else {
      data = {
        version: 1,
        bundleType: 'URSINA_RL_LEARNED_BUNDLE',
        timestamp: Date.now(),
        savedAt: new Date().toISOString(),
        sacModel: this.sacAgent.exportModelJSON(),
        qTableModel: this.rlAgent.exportModelJSON(),
      };
      filename = `ursina_rl_learned_bundle_${dateStr}.json`;
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /**
   * Import learned model weights from a JSON file into active RL agents
   */
  public async importModelFromFile(file: File): Promise<{ success: boolean; message: string }> {
    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (data.bundleType === 'URSINA_RL_LEARNED_BUNDLE') {
        let sacOk = false;
        let qOk = false;
        if (data.sacModel) {
          sacOk = this.sacAgent.importModelJSON(data.sacModel);
          this.sacAgent.saveToLocalStorage();
        }
        if (data.qTableModel) {
          qOk = this.rlAgent.importModelJSON(data.qTableModel);
          this.rlAgent.saveToLocalStorage();
        }
        this.syncStats();
        return {
          success: sacOk || qOk,
          message: `Imported full RL bundle (SAC: ${sacOk ? 'OK' : 'Skipped'}, Q-Table: ${qOk ? 'OK' : 'Skipped'})`,
        };
      } else if (data.modelType === 'SAC_CONTINUOUS_3D') {
        const ok = this.sacAgent.importModelJSON(data);
        if (ok) {
          this.sacAgent.saveToLocalStorage();
          this.syncStats();
          return { success: true, message: 'SAC Continuous 3D weights imported successfully' };
        }
        return { success: false, message: 'Invalid SAC model file format' };
      } else if (data.modelType === 'TABULAR_Q_LEARNING' || data.qTable) {
        const ok = this.rlAgent.importModelJSON(data);
        if (ok) {
          this.rlAgent.saveToLocalStorage();
          this.syncStats();
          return { success: true, message: 'Tabular Q-Learning state imported successfully' };
        }
        return { success: false, message: 'Invalid Q-Table model file format' };
      } else {
        return { success: false, message: 'Unrecognized RL model file structure' };
      }
    } catch (err: any) {
      return { success: false, message: `Failed to load file: ${err?.message || 'Invalid JSON'}` };
    }
  }

  /**
   * Set active Emergent Behavior Profile
   */
  public setEmergentBehaviorProfile(profile: EmergentBehaviorProfile) {
    this.sacAgent.behaviorProfile = profile;
    this.sacAgent.initializeEmergentPriors(profile);
    this.syncStats();
  }

  /**
   * Load optimized pre-trained emergent neural policy
   */
  public loadOptimizedEmergentPolicy(profile: EmergentBehaviorProfile = 'adaptive_predator') {
    this.sacAgent.resetAll(profile);
    this.sacAgent.saveToLocalStorage();
    this.syncStats();
  }

  /**
   * Reset learned model states and clear persistent storage
   */
  public resetRLModel(type: 'sac' | 'qtable' | 'all') {
    if (type === 'sac' || type === 'all') {
      this.sacAgent.resetAll();
      this.sacAgent.clearLocalStorage();
    }
    if (type === 'qtable' || type === 'all') {
      this.rlAgent.resetQTable();
      this.rlAgent.clearLocalStorage();
    }
    this.syncStats();
  }
}
