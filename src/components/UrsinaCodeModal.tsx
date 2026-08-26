import React, { useState } from 'react';
import {
  X,
  Copy,
  Check,
  Terminal,
  Brain,
  Layers,
  Cpu,
  Calculator,
  Compass,
  Sparkles,
  Zap,
  Code2,
} from 'lucide-react';

interface UrsinaCodeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const MATHEMATICAL_FORMULATION_DOC = `# =====================================================================
# CONTINUOUS 360° OMNIDIRECTIONAL SNAKE RL MATHEMATICAL FORMULATION
# =====================================================================

1. KINEMATIC STATE SPACE s_t ∈ R^(2 + 4n):
   ---------------------------------------------------------------------
   • Snake Continuous Length:  l_t ∈ [l_min, l_max] ⊂ R+
   • Snake Heading Angle:     θ_t ∈ [-π, π]
   • Relative Swarm Coordinates (for n nearest boids):
     Δx_{i,t} = x_{i,t}^{boid} - x_t^{snake}
     Δz_{i,t} = z_{i,t}^{boid} - z_t^{snake}
   • Relative Swarm Velocities (for n nearest boids):
     Δvx_{i,t} = vx_{i,t}^{boid} - vx_t^{snake}
     Δvz_{i,t} = vz_{i,t}^{boid} - vz_t^{snake}

   Observation Vector:
   s_t = [ l_t, θ_t, Δx_{1,t}, Δz_{1,t}, Δvx_{1,t}, Δvz_{1,t}, ..., Δx_{n,t}, Δz_{n,t}, Δvx_{n,t}, Δvz_{n,t} ]

2. CONTINUOUS ACTION SPACE a_t ∈ [-1.0, 1.0]:
   ---------------------------------------------------------------------
   • Steering Input:          a_t ∈ [-1.0, 1.0]  (Normalized turn command)
   • Angular Velocity:        ω_t = a_t · ω_max
     where ω_max is the maximum angular velocity in rad/s.

3. TRANSITION DYNAMICS & 360° CONTINUOUS KINEMATICS:
   ---------------------------------------------------------------------
   Given time step Δt and constant forward velocity v:
   
   • Heading Update:
     θ_{t+1} = wrap_to_pi( θ_t + ω_t · Δt )
     
   • 2D Position Update:
     x_{t+1} = x_t + v · sin(θ_{t+1}) · Δt
     z_{t+1} = z_t + v · cos(θ_{t+1}) · Δt

4. PASSIVE CONTINUOUS GROWTH & HIT SHORTENING DYNAMICS:
   ---------------------------------------------------------------------
   • Growth Rate:             g (units per second)
   • Hit Penalty:             c (units lost per swarm bite)
   • Active Bites in Step:    k_t = Σ I(||p_boid - p_seg|| < r_hit)
   
   Continuous Length Update:
   l_{t+1} = max( l_min, l_t + g · Δt - c · k_t )

5. REINFORCEMENT LEARNING REWARD FUNCTION:
   ---------------------------------------------------------------------
   r_t = (α · l_t) - (β · k_t) - (γ · d_t)
   
   Where:
   • α > 0 : Reward weight for maintaining length and surviving over time
   • β > 0 : Penalty weight for each swarm bite received
   • γ > 0 : Large terminal penalty if the snake perishes (d_t ∈ {0, 1})

6. BODY HISTORY & SERPENTINE INVERSE KINEMATICS (DEQUE TRACKING):
   ---------------------------------------------------------------------
   • Head positions are stored in a FIFO queue / deque: Q = [p_0, p_1, ..., p_m]
   • Segment i is positioned at arc-length distance d_i = i · s along the trajectory trace:
     ||p_i - p_{i-1}|| = spacing
`;

const GYMNASIUM_PYTHON_ENV_CODE = `# =====================================================================
# CONTINUOUS 360° SNAKE GYMNASIUM ENVIRONMENT (Python / Gymnasium)
# Requirements: pip install gymnasium numpy torch matplotlib
# =====================================================================

import math
import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ContinuousSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        n_nearest: int = 3,
        forward_velocity: float = 6.0,
        max_turn_rate: float = math.pi,  # 180 deg/s
        dt: float = 0.05,
        arena_bound: float = 28.0,
        growth_rate: float = 0.5,
        hit_penalty: float = 5.0,
        alpha_length: float = 0.1,
        beta_hit: float = 10.0,
        gamma_death: float = 100.0,
    ):
        super().__init__()
        self.n_nearest = n_nearest
        self.v = forward_velocity
        self.omega_max = max_turn_rate
        self.dt = dt
        self.bound = arena_bound
        self.growth_rate = growth_rate
        self.hit_penalty = hit_penalty
        self.alpha = alpha_length
        self.beta = beta_hit
        self.gamma = gamma_death

        # Continuous Action Space: a_t in [-1, 1] (Steering rate)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Continuous Observation Space: [length, heading, (dx, dz, dvx, dvz) * n]
        obs_dim = 2 + 4 * self.n_nearest
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.head_pos = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.length = 10.0
        self.body_history = collections.deque(maxlen=2000)

        # Simulated Swarm
        self.num_boids = 40
        self.boid_pos = np.zeros((self.num_boids, 2), dtype=np.float32)
        self.boid_vel = np.zeros((self.num_boids, 2), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.head_pos = np.zeros(2, dtype=np.float32)
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.length = 10.0
        self.body_history.clear()
        self.body_history.append(self.head_pos.copy())

        # Spawn swarm in ring perimeter
        angles = self.np_random.uniform(-math.pi, math.pi, size=self.num_boids)
        radii = self.np_random.uniform(12.0, 22.0, size=self.num_boids)
        self.boid_pos[:, 0] = np.cos(angles) * radii
        self.boid_pos[:, 1] = np.sin(angles) * radii
        self.boid_vel = self.np_random.uniform(-1.0, 1.0, size=(self.num_boids, 2))

        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate relative coordinates and velocities of nearest boids
        diff = self.boid_pos - self.head_pos
        dists = np.linalg.norm(diff, axis=1)
        nearest_indices = np.argsort(dists)[: self.n_nearest]

        obs = [self.length, self.heading]
        snake_vel = np.array([math.sin(self.heading) * self.v, math.cos(self.heading) * self.v])

        for idx in nearest_indices:
            dx, dz = diff[idx]
            dvx = self.boid_vel[idx, 0] - snake_vel[0]
            dvz = self.boid_vel[idx, 1] - snake_vel[1]
            obs.extend([dx, dz, dvx, dvz])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        a_t = float(np.clip(action[0], -1.0, 1.0))

        # 1. Continuous Heading Update: theta_{t+1} = theta_t + omega_t * dt
        omega_t = a_t * self.omega_max
        self.heading = (self.heading + omega_t * self.dt + math.pi) % (2 * math.pi) - math.pi

        # 2. Continuous 360-degree Position Update
        self.head_pos[0] += math.sin(self.heading) * self.v * self.dt
        self.head_pos[1] += math.cos(self.heading) * self.v * self.dt

        # Arena bounds clamping
        self.head_pos = np.clip(self.head_pos, -self.bound, self.bound)
        self.body_history.appendleft(self.head_pos.copy())

        # 3. Swarm Boids Motion towards snake
        to_head = self.head_pos - self.boid_pos
        dist_to_head = np.linalg.norm(to_head, axis=1, keepdims=True) + 1e-4
        dir_to_head = to_head / dist_to_head
        self.boid_vel += dir_to_head * 5.0 * self.dt
        self.boid_vel *= 0.98
        self.boid_pos += self.boid_vel * self.dt

        # 4. Check Swarm Collisions
        hits = int(np.sum(dist_to_head.flatten() < 1.0))
        
        # 5. Continuous Length Dynamics: l_{t+1} = max(l_min, l_t + g*dt - c*k_t)
        self.length = max(3.0, self.length + self.growth_rate * self.dt - self.hit_penalty * hits)

        # 6. Reward Function: r_t = (alpha * l_t) - (beta * k_t) - (gamma * d_t)
        terminated = bool(self.length <= 3.0)
        d_t = 1.0 if terminated else 0.0
        reward = (self.alpha * self.length) - (self.beta * hits) - (self.gamma * d_t)

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {"hits": hits, "length": self.length}
`;

const PYTORCH_PPO_TRAINING_CODE = `# =====================================================================
# CONTINUOUS PPO REINFORCEMENT LEARNING TRAINER (PyTorch)
# =====================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor: outputs mean mu and log_std for continuous steering a_t in [-1, 1]
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: outputs state-value V(s)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        feat = self.shared(state)
        mu = self.actor_mean(feat)
        std = torch.exp(self.actor_log_std)
        value = self.critic(feat)
        return Normal(mu, std), value

def train_continuous_snake():
    env = ContinuousSnakeEnv()
    policy = ContinuousActorCritic(env.observation_space.shape[0])
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    print("Starting Continuous Snake PPO Training...")
    for epoch in range(100):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            dist, value = policy(state_t)
            action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0).numpy()[0]

            next_state, reward, terminated, truncated, info = env.step(action_clamped)
            total_reward += reward
            done = terminated or truncated
            state = next_state

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Reward = {total_reward:.2f}, Final Length = {info['length']:.1f}")

if __name__ == "__main__":
    train_continuous_snake()
`;

const URSINA_CONTINUOUS_SCRIPT = `# =====================================================================
# Red vs Swarm: Continuous 360° Steering Survival Engine (Ursina / Panda3D)
# Requirements: pip install ursina numpy
# =====================================================================

import math
import random
import numpy as np
from ursina import *

app = Ursina(title="Red Continuous Snake vs Swarm", vsync=True)
window.color = color.rgb(18, 22, 28)

# ---------------------------------------------------------
# Continuous 360° Snake Entity
# ---------------------------------------------------------
class ContinuousSnake(Entity):
    def __init__(self):
        super().__init__(model='sphere', color=color.rgb(239, 68, 68), scale=1.4, position=(0, 0.6, 0))
        self.heading = 0.0  # theta in [-pi, pi]
        self.v = 6.0        # constant forward velocity
        self.omega_max = math.pi  # max turn rate (rad/s)
        self.continuous_length = 8.0
        self.segments = []
        self.history = []
        self.spacing = 0.42
        self.repel_radius = 6.5
        self.repel_cooldown = 0.0

    def update_kinematics(self, steering_input, dt):
        # 1. Angular velocity: omega_t = a_t * omega_max
        omega = steering_input * self.omega_max

        # 2. Heading: theta_{t+1} = theta_t + omega_t * dt
        self.heading += omega * dt
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        self.rotation_y = -math.degrees(self.heading)

        # 3. Position: x_{t+1} = x_t + v*sin(theta)*dt, z_{t+1} = z_t + v*cos(theta)*dt
        self.x += math.sin(self.heading) * self.v * dt
        self.z += math.cos(self.heading) * self.v * dt

        # Arena bounds
        self.x = clamp(self.x, -28, 28)
        self.z = clamp(self.z, -28, 28)

        # Update History & Segments
        self.history.insert(0, Vec3(self.x, self.y, self.z))
        if len(self.history) > 500:
            self.history.pop()

        # Follower segments
        target_count = int(self.continuous_length)
        while len(self.segments) < target_count:
            seg = Entity(model='sphere', color=color.rgb(220, 38, 38), scale=1.1)
            self.segments.append(seg)
        while len(self.segments) > target_count:
            destroy(self.segments.pop())

        for i, seg in enumerate(self.segments):
            hist_idx = min(len(self.history) - 1, int((i + 1) * 3))
            seg.position = self.history[hist_idx]

snake = ContinuousSnake()

# Autonomous Swarm Boids
swarm = [
    Entity(model='sphere', color=color.rgb(250, 204, 21), scale=0.35, position=Vec3(random.uniform(-20, 20), 1.0, random.uniform(-20, 20)))
    for _ in range(40)
]

# Arena
ground = Entity(model='plane', scale=60, color=color.rgb(24, 28, 36), collider='box')
grid = Entity(model=Grid(60, 60), rotation_x=90, y=0.01, color=color.rgba(255, 255, 255, 25))

camera.position = (0, 32, -24)
camera.rotation_x = 55

def update():
    dt = time.dt
    steer = 0.0
    if held_keys['a'] or held_keys['left arrow']:
        steer -= 1.0
    if held_keys['d'] or held_keys['right arrow']:
        steer += 1.0

    snake.update_kinematics(steer, dt)

    # Passive growth
    snake.continuous_length += 0.5 * dt

    # Swarm attraction
    for b in swarm:
        diff = (snake.position - b.position).normalized()
        b.position += diff * 4.5 * dt
        if distance(b.position, snake.position) < 1.0:
            snake.continuous_length = max(3.0, snake.continuous_length - 1.5 * dt)

app.run()
`;

const BOID_SWARM_TUTORIAL_CODE = `// =====================================================================
// BOID SWARM ENGINE (FROM SCRATCH: NO SUMMONER, NO SPEAR)
// Standalone Single-File / Module Implementation
// =====================================================================

import * as THREE from 'three';

// === STEP 2: SOA DATA LAYOUT (STRUCT-OF-ARRAYS) ===
const MAX_BOIDS = 10000;

const boidX    = new Float32Array(MAX_BOIDS);
const boidY    = new Float32Array(MAX_BOIDS);
const boidZ    = new Float32Array(MAX_BOIDS);
const boidVx   = new Float32Array(MAX_BOIDS);
const boidVy   = new Float32Array(MAX_BOIDS);
const boidVz   = new Float32Array(MAX_BOIDS);
const boidAge  = new Float32Array(MAX_BOIDS);   // survival age in seconds
const boidGen  = new Float32Array(MAX_BOIDS);   // generation number
const boidSeed = new Float32Array(MAX_BOIDS);   // per-boid phase seed for wiggle

let boidCount = 0;

// === STEP 3: 3D PERLIN NOISE (KEN PERLIN IMPROVED NOISE) ===
const PERM = new Uint8Array(512);
const GRAD3 = [
    1,1,0, -1,1,0, 1,-1,0, -1,-1,0,
    1,0,1, -1,0,1, 1,0,-1, -1,0,-1,
    0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1,
    1,1,0, 0,-1,1, -1,1,0, 0,-1,-1
];

(function initPerm() {
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    let seed = 42;
    for (let i = 255; i > 0; i--) {
        seed = (seed * 16807) % 2147483647;
        const j = Math.floor((seed / 2147483647) * (i + 1));
        [p[i], p[j]] = [p[j], p[i]];
    }
    for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
})();

function fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
function lerp(a, b, t) { return a + t * (b - a); }

function grad3D(hash, x, y, z) {
    const h = (hash & 15) * 3;
    return GRAD3[h] * x + GRAD3[h+1] * y + GRAD3[h+2] * z;
}

export function perlin3D(x, y, z) {
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const Z = Math.floor(z) & 255;
    const fx = x - Math.floor(x);
    const fy = y - Math.floor(y);
    const fz = z - Math.floor(z);
    const u = fade(fx), v = fade(fy), w = fade(fz);

    const A  = PERM[X] + Y;
    const AA = PERM[A] + Z;
    const AB = PERM[A+1] + Z;
    const B  = PERM[X+1] + Y;
    const BA = PERM[B] + Z;
    const BB = PERM[B+1] + Z;

    return lerp(
        lerp(lerp(grad3D(PERM[AA],   fx,   fy,   fz),
                  grad3D(PERM[BA],   fx-1, fy,   fz), u),
             lerp(grad3D(PERM[AB],   fx,   fy-1, fz),
                  grad3D(PERM[BB],   fx-1, fy-1, fz), u), v),
        lerp(lerp(grad3D(PERM[AA+1], fx,   fy,   fz-1),
                  grad3D(PERM[BA+1], fx-1, fy,   fz-1), u),
             lerp(grad3D(PERM[AB+1], fx,   fy-1, fz-1),
                  grad3D(PERM[BB+1], fx-1, fy-1, fz-1), u), v),
        w
    );
}

// === STEP 4: SPAWNING BOIDS ===
export function spawnBoid(x, y, z, vx = 0, vy = 0, vz = 0, generation = 0) {
    if (boidCount >= MAX_BOIDS) return;
    
    const i = boidCount;
    boidX[i]    = x;
    boidY[i]    = Math.max(0.35, y);   // clamp above ground
    boidZ[i]    = z;
    boidVx[i]   = vx;
    boidVy[i]   = vy;
    boidVz[i]   = vz;
    boidAge[i]  = 0;
    boidGen[i]  = generation;
    boidSeed[i] = Math.random() * 1000;   // unique phase offset

    boidCount++;
}

// === STEP 5: THE UPDATE LOOP (THE 4 EQUATIONS) ===
const DAMPING       = 0.92;     // velocity decay per frame
const ACCEL         = 15.0;     // flow field acceleration
const MAX_SPEED     = 6.5;      // terminal velocity
const CELL_SIZE     = 2.0;      // spatial grid cell size
const DENSITY_LIMIT = 6;        // max boids per cell before jitter
const ATTACK_WEIGHT = 0.8;      // pursuit pull toward target

export function updateBoids(dt, simTime, targetX = 0, targetZ = 0) {
    if (boidCount === 0) return;

    const dtClamped = Math.min(dt, 0.05);  // prevent spiral of death
    const spatialFreq = 0.002;
    const timeFreq = 0.1;

    // Spatial grid for O(1) density separation
    const HALF_WORLD = 30.0;
    const GRID_COLS = Math.ceil(60.0 / CELL_SIZE);
    const cellCounts = new Uint16Array(GRID_COLS * GRID_COLS);

    let writeIdx = 0;

    for (let i = 0; i < boidCount; i++) {
        let x  = boidX[i];
        let y  = boidY[i];
        let z  = boidZ[i];
        let vx = boidVx[i];
        let vy = boidVy[i];
        let vz = boidVz[i];

        // === EQUATION 1: Flow field angle from Perlin noise ===
        const angle = perlin3D(
            x * spatialFreq,
            z * spatialFreq,
            simTime * timeFreq
        ) * Math.PI * 4.0;

        // Per-boid wiggle for individuality
        const wiggle = Math.sin(simTime * 3.0 + boidSeed[i]) * 0.35;

        // === EQUATION 2: Damped velocity integration ===
        vx = (vx * DAMPING) + Math.cos(angle + wiggle) * ACCEL * dtClamped;
        vz = (vz * DAMPING) + Math.sin(angle + wiggle) * ACCEL * dtClamped;
        vy = (vy * DAMPING) + Math.sin(simTime * 2.0 + boidSeed[i] * 0.5) * (ACCEL * 0.25) * dtClamped;

        // === EXTRAS: Pursuit force toward target ===
        if (ATTACK_WEIGHT > 0) {
            const dxA = targetX - x;
            const dzA = targetZ - z;
            const distA = Math.sqrt(dxA * dxA + dzA * dzA) || 0.001;
            const invDistA = 1.0 / distA;
            vx += dxA * invDistA * (ATTACK_WEIGHT * 1.8) * dtClamped;
            vz += dzA * invDistA * (ATTACK_WEIGHT * 1.8) * dtClamped;
        }

        // === EQUATION 3: Spatial grid density check (O(1) separation) ===
        const cX = Math.min(Math.max(Math.floor((x + HALF_WORLD) / CELL_SIZE), 0), GRID_COLS - 1);
        const cZ = Math.min(Math.max(Math.floor((z + HALF_WORLD) / CELL_SIZE), 0), GRID_COLS - 1);
        const cellIdx = cX + cZ * GRID_COLS;

        cellCounts[cellIdx]++;
        if (cellCounts[cellIdx] > DENSITY_LIMIT) {
            vx += (Math.random() - 0.5) * 1.5;
            vz += (Math.random() - 0.5) * 1.5;
        }

        // === EQUATION 4: Terminal velocity cap ===
        const speedSq = vx * vx + vy * vy + vz * vz;
        if (speedSq > MAX_SPEED * MAX_SPEED) {
            const scale = MAX_SPEED / Math.sqrt(speedSq);
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        // === Position integration ===
        x += vx * dtClamped * 4.0;
        y += vy * dtClamped;
        z += vz * dtClamped * 4.0;

        // === Arena bounds wrapping ===
        if (x < -HALF_WORLD) x = HALF_WORLD - 0.5;
        if (x >  HALF_WORLD) x = -HALF_WORLD + 0.5;
        if (z < -HALF_WORLD) z = HALF_WORLD - 0.5;
        if (z >  HALF_WORLD) z = -HALF_WORLD + 0.5;

        // === Ground collision ===
        if (y < 0.35) {
            y = 0.35;
            vy = Math.max(0, vy);
        }

        // === Compact: keep this boid ===
        boidX[writeIdx]  = x;
        boidY[writeIdx]  = y;
        boidZ[writeIdx]  = z;
        boidVx[writeIdx] = vx;
        boidVy[writeIdx] = vy;
        boidVz[writeIdx] = vz;

        writeIdx++;
    }

    boidCount = writeIdx;
}

// === STEP 6: CONTINUOUS REPLICATION (1 -> 4 SPLITTING) ===
let replicationTimer = 0;
const REPLICATION_INTERVAL = 0.01;  // every 10ms
const REPLICATION_MULTIPLIER = 4;   // 1 parent -> 4 total (+3 children)

export function updateReplication(dt) {
    replicationTimer += dt;

    if (replicationTimer >= REPLICATION_INTERVAL && boidCount > 0 && boidCount < MAX_BOIDS) {
        replicationTimer = 0;
        const initialCount = boidCount;
        const childrenPerParent = REPLICATION_MULTIPLIER - 1; // 3

        for (let i = 0; i < initialCount; i++) {
            if (boidCount >= MAX_BOIDS) break;

            const px = boidX[i], py = boidY[i], pz = boidZ[i];
            const pvx = boidVx[i], pvy = boidVy[i], pvz = boidVz[i];
            const gen = boidGen[i] + 1;

            for (let k = 0; k < childrenPerParent; k++) {
                if (boidCount >= MAX_BOIDS) break;

                const jitter = 0.45;
                const nx = px + (Math.random() - 0.5) * jitter;
                const ny = Math.max(0.35, py + (Math.random() - 0.5) * jitter);
                const nz = pz + (Math.random() - 0.5) * jitter;

                const angle = Math.random() * Math.PI * 2;
                const nvx = pvx * 0.5 + Math.cos(angle) * 2;
                const nvy = pvy * 0.5 + (Math.random() - 0.5) * 0.8;
                const nvz = pvz * 0.5 + Math.sin(angle) * 2;

                spawnBoid(nx, ny, nz, nvx, nvy, nvz, gen);
            }
        }
    }
}
`;

export const UrsinaCodeModal: React.FC<UrsinaCodeModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'boids' | 'math' | 'gym' | 'ppo' | 'ursina'>('boids');
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const getContent = () => {
    switch (activeTab) {
      case 'boids':
        return BOID_SWARM_TUTORIAL_CODE;
      case 'math':
        return MATHEMATICAL_FORMULATION_DOC;
      case 'gym':
        return GYMNASIUM_PYTHON_ENV_CODE;
      case 'ppo':
        return PYTORCH_PPO_TRAINING_CODE;
      case 'ursina':
        return URSINA_CONTINUOUS_SCRIPT;
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(getContent());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-fade-in select-none">
      <div className="relative w-full max-w-5xl max-h-[92vh] bg-slate-950 border border-slate-800 rounded-3xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/80">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-amber-500 to-rose-600 flex items-center justify-center border border-amber-400/40 shadow-lg shadow-amber-950/40">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <span>Swarm & Kinematics Engine Architecture</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  Boid SoA + 360° RL
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Boid Swarm Engine (SoA & 4 Equations), Gymnasium Env, PyTorch PPO & Panda3D Script
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors shadow-sm"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-300">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 px-6 py-2.5 bg-slate-900/50 border-b border-slate-800/80 overflow-x-auto text-xs font-medium">
          <button
            onClick={() => setActiveTab('boids')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-colors ${
              activeTab === 'boids'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Boid Swarm (SoA + 4 Eqs)</span>
          </button>

          <button
            onClick={() => setActiveTab('math')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-colors ${
              activeTab === 'math'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Calculator className="w-3.5 h-3.5" />
            <span>Snake RL Math Formulation</span>
          </button>

          <button
            onClick={() => setActiveTab('gym')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-colors ${
              activeTab === 'gym'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>Gymnasium Env (Python)</span>
          </button>

          <button
            onClick={() => setActiveTab('ppo')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-colors ${
              activeTab === 'ppo'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Brain className="w-3.5 h-3.5" />
            <span>PyTorch PPO Trainer</span>
          </button>

          <button
            onClick={() => setActiveTab('ursina')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-colors ${
              activeTab === 'ursina'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>Ursina / Panda3D Script</span>
          </button>
        </div>

        {/* Code View */}
        <div className="flex-1 p-4 overflow-y-auto bg-slate-950 font-mono text-xs text-slate-300 select-text">
          <pre className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800/90 overflow-x-auto leading-relaxed shadow-inner">
            <code>{getContent()}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
