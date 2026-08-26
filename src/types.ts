export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface SnakeSegment {
  x: number;
  y: number;
  z: number;
  angle: number;
  pitch?: number;
}

export interface FloatingDamageNumber {
  id: string;
  position: Vector3D;
  text: string;
  color: string;
  life: number;
  maxLife: number;
  velocity: Vector3D;
  isCrit?: boolean;
}

export interface ParticleEffect {
  id: string;
  position: Vector3D;
  velocity: Vector3D;
  color: string;
  size: number;
  life: number;
  maxLife: number;
  type?: 'decay_creature_explosion' | 'dash' | 'generic';
  drag?: number;
  gravity?: number;
}

export interface ShockwaveEffect {
  id: string;
  position: Vector3D;
  radius: number;
  maxRadius: number;
  life: number;
  maxLife: number;
  color: string;
}

// Tactical Cube & Trench Terrain Deformation Records
export interface TerrainDeformationRecord {
  id: string;
  point: Vector3D; // center position on XZ squared ground plane
  cubeSize: number; // width in X / Z
  width?: number; // width (X-axis)
  height?: number; // vertical height (Y-axis)
  depth?: number; // depth (Z-axis)
  baseY?: number; // base vertical elevation Y (0 for ground tier, > 0 for stacked tiers)
  stackLevel?: number; // 1 = base ground cube, 2 = stacked on 1st, etc.
  cubeType: 'small' | 'big' | 'cuboid';
  heightChange: number; // positive = raised cube, negative = carved trench
  isCarved?: boolean;
  timestamp: number;
}

export type TerrainBrushMode =
  | 'small_cube_raise'
  | 'big_cube_raise'
  | 'custom_cuboid_raise'
  | 'small_cube_carve'
  | 'big_cube_carve'
  | 'raise_cube'
  | 'carve_cube'
  | 'none';

// =====================================================================
// Soft Actor-Critic (SAC) Continuous Control RL Types for 3D Snake
// =====================================================================
export type SACRewardVariant = 'variant_c_combined' | 'variant_a_net_growth' | 'variant_b_max_size';

export type EmergentTactic =
  | 'soar_spear_evade'
  | 'orbital_flank_peel'
  | 'bait_kinetic_blast'
  | 'trench_chokepoint_trap'
  | 'barrier_shield_deflect'
  | 'surge_dash_escape'
  | 'spiral_coil_defense'
  | 'volumetric_dive_bomb'
  | 'free_adaptive_cruise';

export type EmergentBehaviorProfile =
  | 'adaptive_predator'
  | 'aerial_spear_hunter'
  | 'bait_blast_specialist'
  | 'trench_architect'
  | 'spiral_coil_tank'
  | 'max_growth_runner';

export type CurriculumStage =
  | 'stage_1_survival'
  | 'stage_2_tactical_harvesting'
  | 'stage_3_adversarial_mastery';

export interface SACActionVector {
  ax: number;
  ay: number;
  az: number;
}

export interface SACTransition {
  state: Float32Array;
  action: Float32Array;
  reward: number;
  nextState: Float32Array;
  done: boolean;
  gammaN?: number;
  nStep?: number;
  priority?: number;
  treeIndex?: number;
}

export interface SACMetrics {
  enabled: boolean;
  isEvaluation: boolean;
  actorLoss: number;
  criticLoss: number;
  alpha: number;
  targetEntropy: number;
  replayBufferSize: number;
  replayBufferCapacity: number;
  // Prioritized Experience Replay (PER) & Multi-Step metrics
  usePER: boolean;
  perAlpha?: number;
  perBeta: number;
  meanTDError: number;
  maxTDError: number;
  useNStep: boolean;
  nStep: number;
  meanLength: number;
  maxLength: number;
  damagePer1000Steps: number;
  growthPerStep: number;
  netGrowth: number;
  swarmHitsTotal: number;
  qMinMean: number;
  currentAction: number[]; // [ax, ay, az, a_repel, a_dash, a_barrier, a_trench]
  currentVelocity: [number, number, number];
  currentEmergentTactic?: string;
  emergentConfidence?: number;
  behaviorProfile?: string;
  tacticalIntentScores?: { [key: string]: number };
  abilityIntents?: {
    repel: number;
    dash: number;
    barrier: number;
    trench: number;
  };
  abilityExecutions?: {
    repels: number;
    dashes: number;
    barriers: number;
    trenches: number;
  };
  lastStepReward: number;
  totalAccumulatedReward: number;
  stepCount: number;
  rolloutEpisode: number;
  rewardVariant: SACRewardVariant;
  recentLossHistory: { step: number; actorLoss: number; criticLoss: number; length: number; alpha: number }[];
  // Enhanced RL Architecture & Perception Telemetry
  curriculumStage?: CurriculumStage;
  pbrsShapingReward?: number;
  enclosedWindingArea?: number;
  domainRandomizationEnabled?: boolean;
  layerNormEnabled?: boolean;
  radarSectorDensities?: number[];
}

export type RLAction =
  | 'evade_kite'
  | 'aerial_soar'
  | 'ground_dive'
  | 'altitudinal_corkscrew'
  | 'blast_repel'
  | 'carve_trench'
  | 'tactical_dash'
  | 'flank_reposition';

export type RLProximityState = 'close' | 'mid' | 'far';
export type RLDensityState = 'low' | 'high' | 'critical';
export type RLCooldownState = 'ready' | 'cooling';
export type RLBarrierState = 'shielded' | 'exposed';
export type RLHealthState = 'critical' | 'healthy';
export type RLAltitudeState = 'ground' | 'mid' | 'aerial';

export interface QTableRecord {
  stateKey: string;
  proximityState: RLProximityState;
  densityState: RLDensityState;
  cooldownState: RLCooldownState;
  barrierState: RLBarrierState;
  healthState: RLHealthState;
  altitudeState?: RLAltitudeState;
  actions: Record<RLAction, number>;
  bestAction: RLAction;
  visitCount: number;
}

export interface SurvivalEpisodeRecord {
  episode: number;
  survivalTime: number;
  peakLength: number;
  repelsCount: number;
  boidsEliminated: number;
  totalReward: number;
  barriersDeployed: number;
  causeOfDeath: string;
}

export interface RLAgentStats {
  enabled: boolean;
  epsilon: number;
  alpha: number;
  gamma: number;
  totalReward: number;
  lastReward: number;
  lastAction: RLAction | null;
  lastStateKey: string;
  isExploring: boolean;
  decisionCount: number;
  episodesCount: number;
  qTable: Record<string, Record<RLAction, number>>;
  actionCounts: Record<RLAction, number>;
  bestSurvivalTime: number;
  recentEpisodes: SurvivalEpisodeRecord[];
}

// =====================================================================
// Snake Head Anatomical Mechanics & Jaw State Machine
// =====================================================================
export type SnakeHeadState =
  | 'idle_cruise'
  | 'hunting_track'
  | 'strike_lunge'
  | 'kinetic_roar'
  | 'damage_recoil';

export interface SnakeHeadTelemetry {
  state: SnakeHeadState;
  jawAngleDeg: number;
  eyeTargetDistance: number | null;
  kineticCrownCharge: number;
  fangErectRatio: number;
  tongueDartSpeed: number;
}

// Single Player: Red Snake Entity
export interface RedPlayerEntity {
  position: Vector3D;
  velocity: Vector3D;
  targetPos: Vector3D;
  facingAngle: number;
  turnSpeed: number;
  walkCycle: number;
  speed: number;
  scale: number;
  // Continuous Kinematics & Omnidirectional State Vector
  heading: number; // theta_t in [-pi, pi]
  targetHeading?: number; // target heading angle for 360-degree omnidirectional vector control
  pitch: number; // vertical pitch angle in radians (elevation angle)
  targetPitch?: number; // target pitch angle
  altitude: number; // current vertical altitude Y
  targetAltitude?: number; // target vertical altitude Y
  verticalVelocity: number; // current vertical velocity along Y
  steeringInput: number; // a_t in [-1, 1]
  angularVelocity: number; // omega_t = a_t * omega_max
  continuousLength: number; // l_t in R+
  bodyHistory: Vector3D[]; // queue of past positions representing continuous body
  lastStepReward: number; // r_t = alpha * l_t - beta * k_t - gamma * d_t
  lastStepHits: number; // k_t number of swarm hits
  observationVector: number[]; // s_t = [l_t, theta_t, rel_x_1, rel_z_1, rel_vx_1, rel_vz_1, ...]
  // Snake Body & Elongation Mechanics
  segments: SnakeSegment[];
  snakeLength: number; // Current length in units/segments (capped at 1000)
  snakeMaxCap: number; // 1000 units
  elongateTimer: number; // Tracks 3-second cycle
  elongateInterval: number; // 3.0s
  growthAmount: number; // Units grown per cycle
  peakLength: number; // Highest length reached
  shortenCount: number; // How many times bitten & shortened
  lastShortenedTimer: number; // Visual flash when shortened
  growthPulseTimer: number; // Visual ripple when grown
  // Snake Head Mechanics & Articulated Jaw State
  headState: SnakeHeadState;
  jawAngle: number; // Current jaw opening angle in radians (0 to 0.75 rad)
  targetJawAngle: number;
  kineticCrownCharge: number; // [0, 1] Cranial crest charge
  eyeTargetPos: Vector3D | null; // 3D gaze tracking point
  // Health & Defense
  health: number;
  maxHealth: number;
  isDead: boolean; // Snake cannot be killed by boids, but can be shortened
  damageFlashTimer: number;
  score: number;
  lastDamageTaken: number;
  // Kinetic Repel Blast
  repelCooldown: number;
  maxRepelCooldown: number;
  repelRadius: number;
  flashTimer: number;
  // Dash & Agility
  isDashing: boolean;
  dashTimer: number;
  dashCooldown: number;
  maxDashCooldown: number;
  dashDirection: Vector3D;
  // Control Mode
  controlMode: 'rl_agent' | 'manual_player';
  // Telemetry
  survivalTime: number;
  boidsRepelledCount: number;
  barriersPlacedCount: number;
  trenchesCarvedCount: number;
  damageTakenTotal: number;
}

export type SwarmColorMode =
  | 'survival_age'
  | 'active_state'
  | 'hybrid_age_state'
  | 'state_dynamic'
  | 'generation_lineage'
  | 'kinetic_energy';

export interface SimulationConfig {
  // Red Snake Player Settings
  playerSpeed: number; // default 5.2
  playerDashSpeed: number; // default 15.0
  playerMaxHealth: number; // default 100
  playerRepelRadius: number; // default 6.5
  playerRepelCooldown: number; // default 1.8
  playerDashCooldown: number; // default 2.8
  playerAutoHealRate: number; // default 1.2
  snakeInitialLength: number; // default 24
  snakeMaxCap: number; // default 1000
  snakeElongateInterval: number; // default 3.0s (elongates every 3 seconds)
  snakeGrowthAmount: number; // default 5 units per 3 seconds
  snakeMinLength: number; // default 10 units (cannot be killed, only shortened to min)
  snakeShortenPerBite: number; // default 1 unit per bite
  snakeSegmentSpacing: number; // default 0.42

  // Continuous Omnidirectional Kinematics & 3D Flight Formulation
  continuousSteering: boolean; // default true
  enable3DFlight: boolean; // default true (3D altitudinal soaring & diving)
  playerVerticalSpeed: number; // default 5.0 (vertical speed along Y)
  snakeMinAltitude: number; // default 0.45
  snakeMaxAltitude: number; // default 36.0
  maxTurnRate: number; // omega_max in rad/s (default Math.PI / 2 ≈ 1.57 rad/s, max 3.14)
  growthRate: number; // g in length units per second (default 0.5)
  hitPenalty: number; // c in length units deducted per swarm intersection (default 5.0)
  forwardVelocity: number; // v constant forward locomotion velocity (default 5.2)
  rlAlphaLengthReward: number; // alpha: baseline survival reward scaled by length (default 0.1)
  rlBetaHitPenalty: number; // beta: severe penalty for taking a hit (default 10.0)
  rlGammaTerminalPenalty: number; // gamma: terminal death penalty (default 100.0)
  nearestEnemiesCount: number; // n nearest swarm enemies in state vector (default 3)
  manualControlStyle: 'omnidirectional_vector' | 'tank_steering' | 'pointer_aim';

  // Autonomous Swarm Dynamics & Flow
  boidsEnabled: boolean;
  playerAttractionWeight: number; // 3D Snake attraction and homing weight (default 2.2, range 0.0 to 6.0)
  boidsAttackWeight: number; // Kept for backward compatibility
  boidsSeparationWeight: number; // default 2.0
  boidsAlignmentWeight: number; // default 1.4
  boidsCohesionWeight: number; // default 1.1
  boidsSeparationRadius: number; // default 1.5m
  boidsNeighborRadius: number; // default 4.0m
  boidsMaxSpeed: number; // default 8.5m/s
  boidsMaxForce: number; // default 15.0
  minionReplicationInterval: number; // default 0.01s
  minionReplicationMultiplier: number; // default 4
  maxPointsCap: number; // buffer limit (default 4000)

  // O(N) 3D Perlin Flow Field & Organic Murmuration
  flowFieldEnabled: boolean;
  flowFieldWeight: number; // default 2.6
  flowFieldScale: number; // default 0.002
  flowFieldSpeed: number; // default 0.10
  anchorClusterWeight: number; // default 1.4
  individualWiggleWeight: number; // default 1.8

  // Real-Time 3D Deformable Cubic Terrain Barriers
  terrainDeformationEnabled: boolean;
  terrainBrushMode: TerrainBrushMode;
  terrainCubeSize: number; // default 3.0
  terrainCubeHeight: number; // default 3.0
  terrainCubeDepth: number; // default 2.5
  terrainWireframe: boolean;
  terrainGridResolution: number; // default 50

  // Soft Actor-Critic (SAC) Continuous 3D Control Configuration
  sacEnabled: boolean;
  sacLearningRate: number; // default 3e-4 (0.0003)
  sacDiscountFactor: number; // gamma (default 0.99)
  sacTau: number; // Polyak target update rate (default 0.005)
  sacBatchSize: number; // default 128
  sacReplayCapacity: number; // default 50000
  sacActionPersistence: number; // Hold action for N physics frames (default 4)
  sacRewardVariant: SACRewardVariant; // 'variant_c_combined' | 'variant_a_net_growth' | 'variant_b_max_size'
  sacLengthScale: number; // L_s normalization scale (default 50.0)
  sacAlphaWeight: number; // alpha weight for size reward (default 1.0)
  sacBetaWeight: number; // beta weight for growth delta (default 1.0)
  sacLambdaSmoothness: number; // lambda weight for action smoothness (default 0.005)
  sacNearestEnemiesCount: number; // K nearest swarm agents (default 8)
  sacPredictionHorizon: number; // tau frames for kinematic future prediction (default 10)
  sacDensityRadius: number; // radius r for local swarm density estimation (default 8.0)
  sacIsEvaluation: boolean; // deterministic evaluation vs stochastic exploration
  sacMaxSpeed: number; // max velocity magnitude (default 7.5)
  sacRolloutSteps: number; // max steps per rollout window before trajectory reset (default 5000)
  sacRewardAltitudeBonus?: number; // reward for dynamic 3D vertical flight & altitudinal evasion (default 0.25)
  sacRewardTacticalBonus?: number; // reward for emergent ability coordination & defensive timing (default 0.35)
  sacBehaviorProfile?: EmergentBehaviorProfile; // profile presets for distinct emergent tactical styles
  sacEmergentRewardScale?: number; // multiplier for rich emergent reward terms (default 1.2)
  // Prioritized Experience Replay (PER) & Multi-Step Returns Configuration
  sacUsePER?: boolean; // toggle Prioritized Experience Replay (default true)
  sacPerAlpha?: number; // priority exponent alpha (default 0.6)
  sacPerBeta?: number; // importance sampling exponent beta (default 0.4 -> 1.0)
  sacUseNStep?: boolean; // toggle N-step returns bootstrapping (default true)
  sacNStep?: number; // N-step horizon depth (default 3, range 1-10)
  sacCurriculumStage?: CurriculumStage; // Active curriculum learning stage
  sacUsePBRS?: boolean; // Potential-Based Reward Shaping (PBRS) toggle
  sacPBRSWeight?: number; // PBRS gamma-discounted potential scale (default 0.45)
  sacUseLayerNorm?: boolean; // Layer Normalization in MLP networks (default true)
  sacDomainRandomization?: boolean; // Dynamic per-episode environment randomization
  sacEncirclementBonusWeight?: number; // Weight for trapping swarm inside closed body loop (default 0.8)

  // Legacy / Hybrid Q-Learning Configuration
  rlEnabled: boolean;
  rlLearningRate: number; // alpha (default 0.12)
  rlDiscountFactor: number; // gamma (default 0.92)
  rlEpsilon: number; // exploration rate (default 0.35)
  rlEpsilonDecay: number; // decay (default 0.996)
  rlMinEpsilon: number; // floor (default 0.03)
  rlDecisionInterval: number; // interval in sec (default 0.25s)

  // RL Reward Weights
  rlRewardSurvivalPerSec: number; // default +0.25
  rlRewardGrowth: number; // default +2.5 per 3s elongation
  rlRewardRepelKill: number; // default +0.2 per boid
  rlRewardBarrierDeflect: number; // default +1.2
  rlPenaltyShortened: number; // default -2.0 per segment bitten
  rlPenaltyDamage: number; // default -1.0
  rlPenaltyWastedRepel: number; // default -1.5
  rlPenaltyDeath: number; // default -25.0

  // Snake Head Anatomy & Visual Kinematics
  snakeHeadEyeTracking?: boolean; // default true (3D ocular saccades tracking nearest boids)
  snakeHeadJawStyle?: 'predator_fang' | 'cyber_plasma' | 'abyssal_viper';
  snakeHeadHornStyle?: 'swept_horns' | 'tactile_whiskers' | 'crown_crest' | 'minimal';

  // Visuals & Environment
  showSnakeTailTrail?: boolean; // default true (trailing particle effect along snake tail path)
  showRepelRadius: boolean;
  showPanicZone: boolean;
  showMinionTrails: boolean;
  showHealthBars: boolean;
  showDamageNumbers: boolean;
  showQDecisionHUD: boolean;
  arenaGridSize: number;
  cameraPreset: 'overhead_follow_red' | 'free_orbit' | 'top_down' | 'side_view';
  soundEnabled: boolean;
  bloomIntensity: number;

  // Swarm Visuals & Survival Feedback
  swarmColorMode?: SwarmColorMode; // default 'survival_age'
  swarmAgeMaxThreshold?: number; // default 20.0s (range 5s - 60s)
  swarmDynamicSizeByAge?: boolean; // default true
}

export interface SimulationStats {
  // Red Snake Player Status
  snakeLength: number;
  continuousLength: number; // l_t continuous length in R+
  heading: number; // theta_t continuous heading in [-pi, pi]
  pitch: number; // vertical pitch angle in radians
  altitude: number; // current vertical altitude Y
  verticalVelocity: number; // current vertical velocity along Y
  steeringInput: number; // a_t continuous steering in [-1, 1]
  angularVelocity: number; // omega_t = a_t * omega_max
  lastStepReward: number; // r_t reward
  lastStepHits: number; // k_t hits this step
  observationVector: number[]; // s_t continuous observation vector
  snakeMaxCap: number;
  peakLength: number;
  nextElongateCountdown: number;
  bitesTakenCount: number;
  playerHealth: number;
  playerMaxHealth: number;
  playerIsDead: boolean;
  playerSurvivalTime: number;
  bestSurvivalTime: number;
  episodeNumber: number;
  playerScore: number;
  playerControlMode: 'rl_agent' | 'manual_player';
  playerRepelCooldownRemaining: number;
  playerDashCooldownRemaining: number;
  lastDamageEvent: string | null;

  // Swarm & Combat Status
  activeBoids: number;
  totalSpawned: number;
  totalRepelled: number;
  repelCount: number;
  boidsNearPlayer: number; // Boids within danger zone (< 5m)
  boidsInBlastZone: number; // Boids within repel radius
  totalBarriersPlaced: number;
  totalTrenchesCarved: number;
  playerAttractionWeight: number;

  // Swarm Survival Demographics & Active State Telemetry
  avgBoidSurvivalTime: number;
  maxBoidSurvivalTime: number;
  swarmAttackState: number;
  swarmAttackStateName: string;
  swarmDemographics: {
    newborn: number;
    youth: number;
    mature: number;
    veteran: number;
    ancient: number;
  };

  // SAC Continuous Reinforcement Learning Telemetry
  sacMetrics: SACMetrics;

  // Q-Learning RL Agent Telemetry (legacy)
  rlAgent: RLAgentStats;

  // Snake Head Anatomical Telemetry
  headTelemetry?: SnakeHeadTelemetry;

  // Performance
  elapsedTime: number;
  fps: number;
}
