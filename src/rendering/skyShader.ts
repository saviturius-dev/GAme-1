import * as THREE from 'three';

/**
 * Animated Scrolling Noise Cloud Sky Shader
 * Implements dual-layer drifting noise with non-repeating swirling patterns,
 * smoothstep contrast adjustments, and vertical horizon-to-zenith gradient blending.
 *
 * Color Palette:
 * - Color_Base (Mist): #7A8B99 (vec3(0.478, 0.545, 0.600))
 * - Color_Storm (Dark Storm Gray): #2E3440 (vec3(0.180, 0.204, 0.251))
 */

export interface SkyDomeHandle {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  update: (time: number, cameraPosition?: THREE.Vector3) => void;
  dispose: () => void;
}

const vertexShader = `
varying vec3 vWorldPosition;
varying vec2 vUv;
varying vec3 vNormal;

void main() {
  vUv = uv;
  vNormal = normalize(normalMatrix * normal);
  vec4 worldPosition = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPosition.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPosition;
}
`;

const fragmentShader = `
uniform float uTime;
uniform vec3 uColorMist;
uniform vec3 uColorStorm;
uniform vec3 uColorZenith;
uniform vec2 uVelocity1;
uniform vec2 uVelocity2;
uniform float uNoiseScale;
uniform float uCloudDensity;
uniform float uContrast;

varying vec3 vWorldPosition;
varying vec2 vUv;
varying vec3 vNormal;

// Simplex 2D Noise Implementation (Stefan Gustavson & Ian McEwan, Ashima Arts)
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

float snoise(vec2 v) {
  const vec4 C = vec4(
    0.211324865405187,   // (3.0 - sqrt(3.0)) / 6.0
    0.366025403784439,   // 0.5 * (sqrt(3.0) - 1.0)
   -0.577350269189626,   // -1.0 + 2.0 * C.x
    0.024390243902439    // 1.0 / 41.0
  );
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// 2-Octave Fractal Noise with Dual Scrolling Vectors
float fbm(vec2 st, vec2 offset) {
  float v = 0.0;
  float a = 0.65;
  vec2 shift = vec2(100.0);
  mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
  for (int i = 0; i < 2; ++i) {
    v += a * (snoise(st + offset) * 0.5 + 0.5);
    st = rot * st * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

void main() {
  // Planar continuous projection based on sky direction to prevent polar UV pinching
  vec3 dir = normalize(vWorldPosition);
  float height = clamp(dir.y, 0.0, 1.0);

  // Scaled UV coordinate space for smooth atmospheric turbulence
  vec2 skyCoords = vec2(dir.x / (dir.y + 0.28), dir.z / (dir.y + 0.28)) * uNoiseScale;

  // Step 1: Compute time drift velocity offsets
  vec2 offset1 = uTime * uVelocity1;
  vec2 offset2 = uTime * uVelocity2;

  // Step 2: Sample dual noise layers
  // Layer 1: Primary cloud mass drift
  float n1 = (snoise(skyCoords + offset1) * 0.5 + 0.5);
  
  // Layer 2: Secondary cross-drift at 1.4x scale
  float n2 = (snoise((skyCoords * 1.4) + offset2) * 0.5 + 0.5);

  // Micro turbulence layer for fine whisps
  float n3 = (snoise((skyCoords * 3.2) - offset1 * 1.8) * 0.5 + 0.5) * 0.28;

  // Step 3: Combine noise layers via multiplication + additive whisps
  float combined_noise = clamp((n1 * n2 * 2.1) + n3, 0.0, 1.0);

  // Fine-tuning: Contrast adjustment with smoothstep and power curve
  combined_noise = smoothstep(0.12, 0.88, combined_noise);
  combined_noise = pow(combined_noise, uContrast);

  // Step 4: Vertical gradient mixing (Horizon mist to zenith dark charcoal)
  // Base Color = #7A8B99 (misty gray)
  // Storm Color = #2E3440 (dark storm gray)
  vec3 horizonColor = mix(uColorMist * 1.08, uColorMist, height);
  vec3 zenithStorm = mix(uColorStorm, uColorZenith, height);

  // Modulate cloud density with altitude
  float altitudeDensity = smoothstep(0.02, 0.45, height);
  float cloudFactor = combined_noise * altitudeDensity * uCloudDensity;

  // Final color interpolation
  vec3 finalColor = mix(horizonColor, zenithStorm, cloudFactor);

  // Subtle atmospheric edge glow at the horizon line
  float horizonGlow = pow(1.0 - height, 4.0) * 0.15;
  finalColor += uColorMist * horizonGlow;

  gl_FragColor = vec4(finalColor, 1.0);
}
`;

export function createProceduralSkyDome(radius: number = 420): SkyDomeHandle {
  const geometry = new THREE.SphereGeometry(radius, 48, 32);

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uColorMist: { value: new THREE.Color(0x7a8b99) },   // #7A8B99
      uColorStorm: { value: new THREE.Color(0x2e3440) },  // #2E3440
      uColorZenith: { value: new THREE.Color(0x1a202c) }, // Dark zenith
      uVelocity1: { value: new THREE.Vector2(0.015, 0.008) },
      uVelocity2: { value: new THREE.Vector2(-0.008, 0.020) },
      uNoiseScale: { value: 0.35 },
      uCloudDensity: { value: 1.0 },
      uContrast: { value: 1.15 },
    },
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = -1000;

  return {
    mesh,
    material,
    update: (time: number, cameraPosition?: THREE.Vector3) => {
      material.uniforms.uTime.value = time;
      if (cameraPosition) {
        mesh.position.set(cameraPosition.x, 0, cameraPosition.z);
      }
    },
    dispose: () => {
      geometry.dispose();
      material.dispose();
    },
  };
}
