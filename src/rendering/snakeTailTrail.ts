import * as THREE from 'three';
import { Vector3D } from '../types';

/**
 * Procedural 64x64 Luminous Tail Ember & Spark Texture
 * Generates an optical flare with an intense diamond-white core,
 * electric cyan-silver inner glow, and smooth additive feathery falloff.
 */
function createTailParticleTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)'); // Diamond white core
    grad.addColorStop(0.18, 'rgba(215, 245, 255, 0.95)'); // Luminous cyan-silver
    grad.addColorStop(0.42, 'rgba(56, 189, 248, 0.65)'); // Electric sky flare
    grad.addColorStop(0.70, 'rgba(30, 64, 175, 0.25)'); // Deep indigo halo
    grad.addColorStop(1.0, 'rgba(15, 23, 42, 0.0)'); // Transparent edge

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return texture;
}

export interface SnakeTailTrailConfig {
  maxParticles?: number;
  baseSize?: number;
  trailLengthSec?: number;
  swirlIntensity?: number;
  buoyancy?: number;
  emissionRateScale?: number;
  glowColorHex?: number;
}

/**
 * High-Performance Zero-GC Trailing Particle System for the 3D Snake's Tail Path.
 * Ejects living sparks, ethereal embers, and kinetic vortices along the 3D wake of the tail.
 */
export class SnakeTailTrailSystem {
  public group: THREE.Group;
  private pointsMesh: THREE.Points;
  private geometry: THREE.BufferGeometry;
  private material: THREE.PointsMaterial;

  // Max capacity
  private maxParticles: number;

  // Zero-GC Flat typed arrays
  private positions: Float32Array;
  private colors: Float32Array;
  private velocities: Float32Array;
  private lifeArray: Float32Array;
  private maxLifeArray: Float32Array;
  private sizeScaleArray: Float32Array;
  private swirlPhases: Float32Array;
  private particleTypes: Uint8Array; // 0 = standard spark, 1 = radiant diamond ember, 2 = kinetic wake flare

  // Active particle tracker
  private activeCount: number = 0;

  // Tail path tracking
  private prevTailPos: THREE.Vector3 = new THREE.Vector3();
  private hasPrevTailPos: boolean = false;
  private emissionAccumulator: number = 0;

  // Reusable scratch vectors
  private vSpawnPos = new THREE.Vector3();
  private vSpawnVel = new THREE.Vector3();
  private vTailDelta = new THREE.Vector3();
  private vOrthogonal = new THREE.Vector3();

  // Optional ribbon trail for continuous ethereal wake
  private ribbonMesh: THREE.Line;
  private ribbonGeometry: THREE.BufferGeometry;
  private ribbonPositions: Float32Array;
  private ribbonColors: Float32Array;
  private maxRibbonPoints: number = 48;
  private ribbonHistory: { x: number; y: number; z: number; time: number }[] = [];

  constructor(config: SnakeTailTrailConfig = {}) {
    this.maxParticles = config.maxParticles ?? 1800;
    this.group = new THREE.Group();
    this.group.name = 'snake_tail_trailing_particles';

    // Allocate flat buffers
    this.positions = new Float32Array(this.maxParticles * 3);
    this.colors = new Float32Array(this.maxParticles * 3);
    this.velocities = new Float32Array(this.maxParticles * 3);
    this.lifeArray = new Float32Array(this.maxParticles);
    this.maxLifeArray = new Float32Array(this.maxParticles);
    this.sizeScaleArray = new Float32Array(this.maxParticles);
    this.swirlPhases = new Float32Array(this.maxParticles);
    this.particleTypes = new Uint8Array(this.maxParticles);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    this.geometry.setDrawRange(0, 0);

    const texture = createTailParticleTexture();

    this.material = new THREE.PointsMaterial({
      size: config.baseSize ?? 0.85,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      map: texture,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.pointsMesh = new THREE.Points(this.geometry, this.material);
    this.pointsMesh.frustumCulled = false;
    this.group.add(this.pointsMesh);

    // Ethereal wake ribbon line
    this.ribbonPositions = new Float32Array(this.maxRibbonPoints * 3);
    this.ribbonColors = new Float32Array(this.maxRibbonPoints * 3);
    this.ribbonGeometry = new THREE.BufferGeometry();
    this.ribbonGeometry.setAttribute('position', new THREE.BufferAttribute(this.ribbonPositions, 3));
    this.ribbonGeometry.setAttribute('color', new THREE.BufferAttribute(this.ribbonColors, 3));
    this.ribbonGeometry.setDrawRange(0, 0);

    const ribbonMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      linewidth: 2,
    });

    this.ribbonMesh = new THREE.Line(this.ribbonGeometry, ribbonMat);
    this.ribbonMesh.frustumCulled = false;
    this.group.add(this.ribbonMesh);
  }

  /**
   * Spawns a single particle in the pre-allocated flat array
   */
  private spawnParticle(
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    life: number,
    type: number = 0,
    sizeScale: number = 1.0
  ) {
    if (this.activeCount >= this.maxParticles) {
      // Overwrite oldest or skip if buffer full
      return;
    }

    const idx = this.activeCount;
    const i3 = idx * 3;

    this.positions[i3 + 0] = x;
    this.positions[i3 + 1] = y;
    this.positions[i3 + 2] = z;

    this.velocities[i3 + 0] = vx;
    this.velocities[i3 + 1] = vy;
    this.velocities[i3 + 2] = vz;

    this.lifeArray[idx] = life;
    this.maxLifeArray[idx] = life;
    this.sizeScaleArray[idx] = sizeScale;
    this.swirlPhases[idx] = Math.random() * Math.PI * 2;
    this.particleTypes[idx] = type;

    // Initial color (bright spark)
    this.colors[i3 + 0] = 0.95;
    this.colors[i3 + 1] = 0.98;
    this.colors[i3 + 2] = 1.0;

    this.activeCount++;
  }

  /**
   * Updates the trailing particle system every animation frame:
   * 1. Interpolates tail path between previous and current frame positions
   * 2. Emits living trailing particles with velocity inheritance, swirl, and radial dispersion
   * 3. Updates particle physics (drag, buoyancy, micro-turbulence swirl, lifespan decay, and color gradient)
   * 4. Updates WebGL BufferGeometry attributes
   */
  public update(
    tailPos: Vector3D | THREE.Vector3,
    tailTangent?: Vector3D | THREE.Vector3,
    dt: number = 0.016,
    isDashing: boolean = false,
    speed: number = 5.0,
    elapsedTime: number = 0
  ) {
    const tx = tailPos.x;
    const ty = typeof tailPos.y === 'number' ? tailPos.y : 0.55;
    const tz = tailPos.z;

    if (!this.hasPrevTailPos) {
      this.prevTailPos.set(tx, ty, tz);
      this.hasPrevTailPos = true;
    }

    // Measure distance travelled by tail this frame
    this.vTailDelta.set(tx - this.prevTailPos.x, ty - this.prevTailPos.y, tz - this.prevTailPos.z);
    const frameTailDist = this.vTailDelta.length();
    const tailSpeed = dt > 0 ? frameTailDist / dt : speed;

    // Determine emission intensity based on movement & dash status
    const isMoving = frameTailDist > 0.005 || tailSpeed > 0.3;
    const baseRate = isDashing ? 320 : isMoving ? 140 : 45;
    const emissionCount = baseRate * dt;
    this.emissionAccumulator += emissionCount;

    // Emit new particles interpolated along the segment path between prevTailPos and tailPos
    while (this.emissionAccumulator >= 1.0) {
      this.emissionAccumulator -= 1.0;
      const subAlpha = Math.random();

      // Interpolate spawn point along the path
      this.vSpawnPos.lerpVectors(this.prevTailPos, new THREE.Vector3(tx, ty, tz), subAlpha);

      // Radial micro-dispersion around the tail diameter
      const angle = Math.random() * Math.PI * 2;
      const radiusSpread = 0.08 + Math.random() * (isDashing ? 0.28 : 0.14);
      const perpX = Math.cos(angle) * radiusSpread;
      const perpY = (Math.random() - 0.4) * radiusSpread * 0.7;
      const perpZ = Math.sin(angle) * radiusSpread;

      this.vSpawnPos.x += perpX;
      this.vSpawnPos.y += perpY;
      this.vSpawnPos.z += perpZ;

      // Base backward velocity along tail wake
      let bvx = 0;
      let bvy = 0.15 + Math.random() * 0.25; // Gentle upward drift
      let bvz = 0;

      if (tailTangent) {
        // Eject in direction opposite of tail tangent
        bvx = -tailTangent.x * (0.8 + Math.random() * 1.2);
        bvy += -tailTangent.y * 0.6;
        bvz = -tailTangent.z * (0.8 + Math.random() * 1.2);
      } else if (frameTailDist > 0.0001) {
        bvx = -this.vTailDelta.x * 2.0;
        bvz = -this.vTailDelta.z * 2.0;
      }

      // Add gentle radial burst velocity
      bvx += perpX * (isDashing ? 4.5 : 1.8);
      bvy += perpY * (isDashing ? 3.0 : 1.2);
      bvz += perpZ * (isDashing ? 4.5 : 1.8);

      const life = isDashing
        ? 0.9 + Math.random() * 0.7
        : 0.65 + Math.random() * 0.65;

      const pType = isDashing
        ? (Math.random() > 0.4 ? 2 : 1)
        : Math.random() > 0.75
        ? 1
        : 0;

      const sizeScale = isDashing ? 1.4 : 0.9 + Math.random() * 0.4;

      this.spawnParticle(
        this.vSpawnPos.x,
        this.vSpawnPos.y,
        this.vSpawnPos.z,
        bvx,
        bvy,
        bvz,
        life,
        pType,
        sizeScale
      );
    }

    this.prevTailPos.set(tx, ty, tz);

    // -------------------------------------------------------------------------
    // Update Active Particles Physics & Color Transitions
    // -------------------------------------------------------------------------
    const dragFactor = Math.max(0, 1.0 - 1.7 * dt);
    const buoyancy = 0.42 * dt;
    const timeSec = elapsedTime;

    for (let i = this.activeCount - 1; i >= 0; i--) {
      this.lifeArray[i] -= dt;

      if (this.lifeArray[i] <= 0) {
        // Remove dead particle by swapping with last active
        const lastIdx = this.activeCount - 1;
        if (i !== lastIdx) {
          const i3 = i * 3;
          const l3 = lastIdx * 3;

          this.positions[i3 + 0] = this.positions[l3 + 0];
          this.positions[i3 + 1] = this.positions[l3 + 1];
          this.positions[i3 + 2] = this.positions[l3 + 2];

          this.velocities[i3 + 0] = this.velocities[l3 + 0];
          this.velocities[i3 + 1] = this.velocities[l3 + 1];
          this.velocities[i3 + 2] = this.velocities[l3 + 2];

          this.colors[i3 + 0] = this.colors[l3 + 0];
          this.colors[i3 + 1] = this.colors[l3 + 1];
          this.colors[i3 + 2] = this.colors[l3 + 2];

          this.lifeArray[i] = this.lifeArray[lastIdx];
          this.maxLifeArray[i] = this.maxLifeArray[lastIdx];
          this.sizeScaleArray[i] = this.sizeScaleArray[lastIdx];
          this.swirlPhases[i] = this.swirlPhases[lastIdx];
          this.particleTypes[i] = this.particleTypes[lastIdx];
        }
        this.activeCount--;
        continue;
      }

      const i3 = i * 3;
      const progress = this.lifeArray[i] / this.maxLifeArray[i]; // 1.0 (fresh) -> 0.0 (decayed)
      const pType = this.particleTypes[i];
      const phase = this.swirlPhases[i];

      // Swirl / micro-vortices force
      const swirlX = Math.sin(phase + timeSec * 3.8) * 0.35 * dt;
      const swirlZ = Math.cos(phase + timeSec * 3.8) * 0.35 * dt;

      this.velocities[i3 + 0] = this.velocities[i3 + 0] * dragFactor + swirlX;
      this.velocities[i3 + 1] = this.velocities[i3 + 1] * dragFactor + buoyancy;
      this.velocities[i3 + 2] = this.velocities[i3 + 2] * dragFactor + swirlZ;

      this.positions[i3 + 0] += this.velocities[i3 + 0] * dt;
      this.positions[i3 + 1] = Math.max(0.02, this.positions[i3 + 1] + this.velocities[i3 + 1] * dt);
      this.positions[i3 + 2] += this.velocities[i3 + 2] * dt;

      // -----------------------------------------------------------------------
      // Dynamic Ethereal Color Grading (Diamond-White / Cyan -> Silver-Slate -> Smoky Charcoal)
      // -----------------------------------------------------------------------
      let r = 1.0, g = 1.0, b = 1.0;

      if (pType === 2) {
        // High-energy Kinetic Dash Burst (Hyper-luminous Plasma Cyan & Hot White)
        if (progress > 0.65) {
          r = 0.95;
          g = 0.98;
          b = 1.0;
        } else if (progress > 0.30) {
          const t = (progress - 0.30) / 0.35;
          r = 0.20 + t * (0.95 - 0.20);
          g = 0.85 + t * (0.98 - 0.85);
          b = 1.00;
        } else {
          const t = progress / 0.30;
          r = 0.05 + t * 0.15;
          g = 0.30 + t * 0.55;
          b = 0.60 + t * 0.40;
        }
      } else if (pType === 1) {
        // Radiant Diamond Ember (Bright Silver-Pearl with Golden Core)
        if (progress > 0.60) {
          r = 0.98;
          g = 0.95;
          b = 0.85;
        } else if (progress > 0.25) {
          const t = (progress - 0.25) / 0.35;
          r = 0.65 + t * (0.98 - 0.65);
          g = 0.72 + t * (0.95 - 0.72);
          b = 0.85 + t * (0.85 - 0.85);
        } else {
          const t = progress / 0.25;
          r = 0.25 * t;
          g = 0.30 * t;
          b = 0.40 * t;
        }
      } else {
        // Standard Living Tail Spark (Luminous Cyan-White -> Silver Slate -> Soft Fade)
        if (progress > 0.65) {
          const t = (progress - 0.65) / 0.35;
          r = 0.65 + t * (0.95 - 0.65);
          g = 0.85 + t * (0.98 - 0.85);
          b = 1.0;
        } else if (progress > 0.25) {
          const t = (progress - 0.25) / 0.40;
          r = 0.28 + t * (0.65 - 0.28);
          g = 0.48 + t * (0.85 - 0.48);
          b = 0.75 + t * (1.00 - 0.75);
        } else {
          const t = progress / 0.25;
          r = 0.08 * t;
          g = 0.18 * t;
          b = 0.35 * t;
        }
      }

      // Fade intensity proportional to remaining life
      const alphaFalloff = Math.sin(progress * Math.PI * 0.5);
      this.colors[i3 + 0] = r * alphaFalloff;
      this.colors[i3 + 1] = g * alphaFalloff;
      this.colors[i3 + 2] = b * alphaFalloff;
    }

    // Sync particle points draw range and flag GPU update
    this.geometry.setDrawRange(0, this.activeCount);
    (this.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;

    // -------------------------------------------------------------------------
    // Update Ethereal Wake Ribbon Path
    // -------------------------------------------------------------------------
    this.updateRibbon(tx, ty, tz, timeSec, isDashing);
  }

  /**
   * Records and renders a fluid continuous wake ribbon connecting recent tail history
   */
  private updateRibbon(tx: number, ty: number, tz: number, timeSec: number, isDashing: boolean) {
    // Add current point to history if moved or periodically
    const last = this.ribbonHistory[0];
    const shouldAdd = !last || Math.hypot(tx - last.x, ty - last.y, tz - last.z) > 0.22;

    if (shouldAdd) {
      this.ribbonHistory.unshift({ x: tx, y: ty, z: tz, time: timeSec });
      if (this.ribbonHistory.length > this.maxRibbonPoints) {
        this.ribbonHistory.pop();
      }
    }

    const count = this.ribbonHistory.length;
    for (let i = 0; i < count; i++) {
      const pt = this.ribbonHistory[i];
      const i3 = i * 3;
      this.ribbonPositions[i3 + 0] = pt.x;
      this.ribbonPositions[i3 + 1] = pt.y + 0.04;
      this.ribbonPositions[i3 + 2] = pt.z;

      const alpha = Math.max(0, 1.0 - i / (count - 1 || 1));
      const colScale = alpha * (isDashing ? 0.85 : 0.45);
      this.ribbonColors[i3 + 0] = 0.22 * colScale;
      this.ribbonColors[i3 + 1] = 0.75 * colScale;
      this.ribbonColors[i3 + 2] = 0.95 * colScale;
    }

    this.ribbonGeometry.setDrawRange(0, count);
    (this.ribbonGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.ribbonGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  }

  /**
   * Resets all active particles
   */
  public reset() {
    this.activeCount = 0;
    this.hasPrevTailPos = false;
    this.emissionAccumulator = 0;
    this.ribbonHistory = [];
    this.geometry.setDrawRange(0, 0);
    this.ribbonGeometry.setDrawRange(0, 0);
  }

  /**
   * Toggles visibility of the tail particle trail
   */
  public setVisible(visible: boolean) {
    this.group.visible = visible;
  }

  /**
   * Clean up WebGL resources
   */
  public dispose() {
    this.geometry.dispose();
    this.material.dispose();
    this.ribbonGeometry.dispose();
    (this.ribbonMesh.material as THREE.Material).dispose();
  }
}
