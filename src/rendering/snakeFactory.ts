import * as THREE from 'three';
import { SnakeSegment, Vector3D, SnakeHeadState } from '../types';

/**
 * Anatomical Taper Profile for Real-World Viper / Python
 * Uses Catmull-Rom continuous cubic spline interpolation for organic curvature
 */
export interface TaperPoint {
  t: number;
  radius: number;
}

export const TAPER_PROFILE: TaperPoint[] = [
  { t: 0.00, radius: 0.72 }, // Anatomical neck constriction behind wide venom lobes
  { t: 0.06, radius: 0.88 }, // Anterior muscular neck flare
  { t: 0.18, radius: 1.05 }, // Heavy muscular front thorax
  { t: 0.40, radius: 1.00 }, // Robust midbody
  { t: 0.65, radius: 0.90 }, // Rear torso
  { t: 0.80, radius: 0.72 }, // Pre-cloaca transition
  { t: 0.90, radius: 0.42 }, // Anterior tail
  { t: 0.96, radius: 0.16 }, // Posterior needle taper
  { t: 1.00, radius: 0.02 }, // Ultra-fine tail apex
];

export const BASE_RADIUS = 0.50;
export const DORSAL_FLATTEN = 0.88; // Subtle arch along dorsal spine with keeled crest
export const VENTRAL_FLATTEN = 0.62; // Flatter ventral belly for broad gastrosteges
export const NUM_RINGS = 144; // Ultra high-resolution longitudinal rings for silky smooth spine curvature
export const RADIAL_SEGMENTS = 36; // 36 radial segments for seamless, smooth cross-sectional geometry
export const VERTS_PER_RING = RADIAL_SEGMENTS + 1; // 37 verts per ring for seamless UV wrapping

/**
 * Catmull-Rom cubic spline interpolation through control points
 */
export function getCatmullRomRadiusAtT(t: number, profile: TaperPoint[] = TAPER_PROFILE): number {
  const clampedT = Math.max(0, Math.min(1, t));
  const n = profile.length;
  if (clampedT <= profile[0].t) return profile[0].radius;
  if (clampedT >= profile[n - 1].t) return profile[n - 1].radius;

  // Find bounding segment
  let i = 0;
  while (i < n - 1 && profile[i + 1].t < clampedT) {
    i++;
  }

  const p1 = profile[i];
  const p2 = profile[i + 1];
  const p0 = profile[Math.max(0, i - 1)];
  const p3 = profile[Math.min(n - 1, i + 2)];

  const span = Math.max(0.0001, p2.t - p1.t);
  const u = (clampedT - p1.t) / span;

  // Catmull-Rom spline formula
  const u2 = u * u;
  const u3 = u2 * u;

  const y0 = p0.radius;
  const y1 = p1.radius;
  const y2 = p2.radius;
  const y3 = p3.radius;

  const val = 0.5 * (
    (2 * y1) +
    (-y0 + y2) * u +
    (2 * y0 - 5 * y1 + 4 * y2 - y3) * u2 +
    (-y0 + 3 * y1 - 3 * y2 + y3) * u3
  );

  return Math.max(0.01, val);
}

/**
 * Real-world snake species pattern & coloration configuration
 */
export interface SnakePatternConfig {
  speciesName: string;
  dorsalBaseColor: string; // Base dorsal slate-grey tone
  dorsalDarkSaddle: string; // Dark obsidian-black diamond / hourglass saddle markings
  saddleHighlightColor: string; // Bright silver-ash / pearl-grey scale bevel highlight
  saddleSecondaryAccent: string; // Mid-tone steel-slate transition accent
  lateralBlotchColor: string; // Flank charcoal-black spot coloration
  scaleMarginColor: string; // Inter-scale groove & margin shadow
  ventralColor: string; // Transverse belly gastrosteges base (silver-slate)
  ventralSpotColor: string; // Lateral paired spots on belly scutes
  patternRepeat: number; // Number of repeating diamond saddles along length
  keeledRidgeIntensity: number; // 0 to 1
}

export const DEFAULT_SNAKE_PATTERN: SnakePatternConfig = {
  speciesName: 'Obsidian & Slate Black-Diamond Viper',
  dorsalBaseColor: '#323843', // Rich gunmetal slate-grey base
  dorsalDarkSaddle: '#06070a', // Deep jet obsidian / pitch-black diamond saddles
  saddleHighlightColor: '#d6dde7', // Brilliant silver-ash / pearl-grey scale bevel highlight
  saddleSecondaryAccent: '#5e6878', // Medium steel-slate transition tone
  lateralBlotchColor: '#0e1116', // Pitch-charcoal lateral flank blotches
  scaleMarginColor: '#050608', // Deep recessed scale grooves
  ventralColor: '#cbd3de', // Pale silver-slate ventral gastrosteges
  ventralSpotColor: '#161920', // Dark carbon-black lateral paired markings on belly scutes
  patternRepeat: 24, // Repeating diamond saddles along spine
  keeledRidgeIntensity: 0.95,
};

/**
 * Generates a high-resolution (2048x1024) realistic PBR diffuse/albedo texture
 * featuring a striking high-contrast grey-and-black snake pattern: overlapping imbricate diamond scales
 * with individual scale lighting gradients, obsidian-black diamondback saddles bordered by silver-ash bevels,
 * lateral flank spots, and segmented silver-slate ventral gastrosteges.
 */
export function generateSnakeBodyTexture(config: SnakePatternConfig = DEFAULT_SNAKE_PATTERN): THREE.CanvasTexture {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    // 1. Base dorsal background fill with rich organic slate-grey tone
    ctx.fillStyle = config.dorsalBaseColor;
    ctx.fillRect(0, 0, width, height);

    const dorsalMidY = height * 0.50; // Exact Center = TOP OF SNAKE (Dorsal Spine)
    const bellyEdgeTop = height * 0.14; // Top canvas margin = Ventral Belly Seam (k=24 / v=1.0)
    const bellyEdgeBottom = height * 0.86; // Bottom canvas margin = Ventral Belly Seam (k=0 / v=0.0)

    // Longitudinal bilateral tone modulation centered at dorsal spine
    const baseGrad = ctx.createLinearGradient(0, 0, 0, height);
    baseGrad.addColorStop(0.0, 'rgba(203, 211, 222, 0.95)'); // Upper belly seam
    baseGrad.addColorStop(0.12, 'rgba(10, 12, 16, 0.85)'); // Lateral belly border
    baseGrad.addColorStop(0.24, 'rgba(24, 28, 35, 0.45)'); // Flank transition
    baseGrad.addColorStop(0.38, 'rgba(50, 56, 67, 0.15)'); // Mid-dorsal slate
    baseGrad.addColorStop(0.50, 'rgba(12, 14, 18, 0.40)'); // Dorsal spine midline
    baseGrad.addColorStop(0.62, 'rgba(50, 56, 67, 0.15)'); // Mid-dorsal slate
    baseGrad.addColorStop(0.76, 'rgba(24, 28, 35, 0.45)'); // Flank transition
    baseGrad.addColorStop(0.88, 'rgba(10, 12, 16, 0.85)'); // Lateral belly border
    baseGrad.addColorStop(1.0, 'rgba(203, 211, 222, 0.95)'); // Lower belly seam
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, width, height);

    const repeat = Math.max(1, config.patternRepeat);
    const saddleWidth = width / repeat;
    const halfH = height * 0.28; // Extends across upper dorsal back

    // 2. High-Contrast Black & Gray Diamondback Saddles ON TOP OF SNAKE (Centered at dorsalMidY)
    for (let b = 0; b < repeat; b++) {
      const startX = b * saddleWidth;
      const midX = startX + saddleWidth * 0.5;
      const endX = startX + saddleWidth;

      // 2a. Diamond Saddle Outer Halo (Crisp pearl-silver scale border)
      ctx.fillStyle = config.saddleHighlightColor;
      ctx.beginPath();
      ctx.moveTo(startX + saddleWidth * 0.05, dorsalMidY);
      ctx.lineTo(midX, dorsalMidY - halfH);
      ctx.lineTo(endX - saddleWidth * 0.05, dorsalMidY);
      ctx.lineTo(midX, dorsalMidY + halfH);
      ctx.closePath();
      ctx.fill();

      // 2b. Secondary Steel-Slate Intermediate Ring
      ctx.fillStyle = config.saddleSecondaryAccent;
      ctx.beginPath();
      ctx.moveTo(startX + saddleWidth * 0.12, dorsalMidY);
      ctx.lineTo(midX, dorsalMidY - halfH * 0.82);
      ctx.lineTo(endX - saddleWidth * 0.12, dorsalMidY);
      ctx.lineTo(midX, dorsalMidY + halfH * 0.82);
      ctx.closePath();
      ctx.fill();

      // 2c. Diamond Saddle Inner Core (Obsidian pitch-black)
      ctx.fillStyle = config.dorsalDarkSaddle;
      ctx.beginPath();
      ctx.moveTo(startX + saddleWidth * 0.19, dorsalMidY);
      ctx.lineTo(midX, dorsalMidY - halfH * 0.64);
      ctx.lineTo(endX - saddleWidth * 0.19, dorsalMidY);
      ctx.lineTo(midX, dorsalMidY + halfH * 0.64);
      ctx.closePath();
      ctx.fill();

      // 2d. Diamond Center Silver/Slate Accent (Diamondback jewel center)
      ctx.fillStyle = config.saddleHighlightColor;
      ctx.beginPath();
      ctx.ellipse(midX, dorsalMidY, saddleWidth * 0.09, halfH * 0.28, 0, 0, Math.PI * 2);
      ctx.fill();

      // Fine inner obsidian jewel dot
      ctx.fillStyle = config.dorsalDarkSaddle;
      ctx.beginPath();
      ctx.ellipse(midX, dorsalMidY, saddleWidth * 0.04, halfH * 0.12, 0, 0, Math.PI * 2);
      ctx.fill();

      // 2e. Bilateral Lateral Flank Blotches (Staggered on both upper and lower flanks)
      // Upper Flank Ocellus (Left Flank, Y ~ dorsalMidY - height * 0.24)
      const upperFlankY = dorsalMidY - height * 0.24;
      ctx.fillStyle = 'rgba(214, 221, 231, 0.65)';
      ctx.beginPath();
      ctx.ellipse(startX, upperFlankY, saddleWidth * 0.16, height * 0.09, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(94, 104, 120, 0.75)';
      ctx.beginPath();
      ctx.ellipse(startX, upperFlankY, saddleWidth * 0.13, height * 0.07, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = config.lateralBlotchColor;
      ctx.beginPath();
      ctx.ellipse(startX, upperFlankY, saddleWidth * 0.09, height * 0.05, 0, 0, Math.PI * 2);
      ctx.fill();

      // Lower Flank Ocellus (Right Flank, Y ~ dorsalMidY + height * 0.24)
      const lowerFlankY = dorsalMidY + height * 0.24;
      ctx.fillStyle = 'rgba(214, 221, 231, 0.65)';
      ctx.beginPath();
      ctx.ellipse(startX, lowerFlankY, saddleWidth * 0.16, height * 0.09, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(94, 104, 120, 0.75)';
      ctx.beginPath();
      ctx.ellipse(startX, lowerFlankY, saddleWidth * 0.13, height * 0.07, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = config.lateralBlotchColor;
      ctx.beginPath();
      ctx.ellipse(startX, lowerFlankY, saddleWidth * 0.09, height * 0.05, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // 3. Ventral Gastrosteges (Segmented Silver-Slate Belly Plates at Top & Bottom Seams)
    // Top belly band [0 .. bellyEdgeTop]
    const topBellyGrad = ctx.createLinearGradient(0, 0, 0, bellyEdgeTop);
    topBellyGrad.addColorStop(0.0, config.ventralColor);
    topBellyGrad.addColorStop(0.85, config.ventralColor);
    topBellyGrad.addColorStop(1.0, 'rgba(20, 24, 30, 0.95)');
    ctx.fillStyle = topBellyGrad;
    ctx.fillRect(0, 0, width, bellyEdgeTop);

    // Bottom belly band [bellyEdgeBottom .. height]
    const botBellyGrad = ctx.createLinearGradient(0, bellyEdgeBottom, 0, height);
    botBellyGrad.addColorStop(0.0, 'rgba(20, 24, 30, 0.95)');
    botBellyGrad.addColorStop(0.15, config.ventralColor);
    botBellyGrad.addColorStop(1.0, config.ventralColor);
    ctx.fillStyle = botBellyGrad;
    ctx.fillRect(0, bellyEdgeBottom, width, height - bellyEdgeBottom);

    // Transverse Belly Scute Seams and Paired Lateral Spots
    const scuteWidth = 14;
    for (let x = 0; x < width; x += scuteWidth) {
      // Top belly seams
      ctx.strokeStyle = 'rgba(10, 12, 16, 0.95)';
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, bellyEdgeTop);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(248, 250, 254, 0.75)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x + 2, 0);
      ctx.lineTo(x + 2, bellyEdgeTop - 2);
      ctx.stroke();

      // Top belly paired lateral spot
      ctx.fillStyle = config.ventralSpotColor;
      ctx.beginPath();
      ctx.ellipse(x + scuteWidth * 0.5, bellyEdgeTop - 12, 3.5, 5.5, 0, 0, Math.PI * 2);
      ctx.fill();

      // Bottom belly seams
      ctx.strokeStyle = 'rgba(10, 12, 16, 0.95)';
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(x, bellyEdgeBottom);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(248, 250, 254, 0.75)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x + 2, bellyEdgeBottom + 2);
      ctx.lineTo(x + 2, height);
      ctx.stroke();

      // Bottom belly paired lateral spot
      ctx.fillStyle = config.ventralSpotColor;
      ctx.beginPath();
      ctx.ellipse(x + scuteWidth * 0.5, bellyEdgeBottom + 12, 3.5, 5.5, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // 4. Overlapping Imbricate Diamond Scales Grid across Dorsal & Flank Zones
    const scaleSize = 9;
    const startRow = Math.floor(bellyEdgeTop / scaleSize);
    const endRow = Math.floor(bellyEdgeBottom / scaleSize) + 1;
    const cols = Math.floor(width / scaleSize) + 2;

    for (let r = startRow; r < endRow; r++) {
      const y = r * scaleSize;
      const offset = (r % 2) * (scaleSize * 0.5);

      for (let c = -1; c < cols; c++) {
        const x = c * scaleSize + offset;

        // Individual diamond scale facet contour
        ctx.strokeStyle = 'rgba(4, 5, 7, 0.75)';
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(x, y - scaleSize * 0.5);
        ctx.lineTo(x + scaleSize * 0.5, y);
        ctx.lineTo(x, y + scaleSize * 0.5);
        ctx.lineTo(x - scaleSize * 0.5, y);
        ctx.closePath();
        ctx.stroke();

        // Scale top facet subtle sheen
        ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.beginPath();
        ctx.moveTo(x, y - scaleSize * 0.35);
        ctx.lineTo(x + scaleSize * 0.35, y);
        ctx.lineTo(x, y + scaleSize * 0.35);
        ctx.lineTo(x - scaleSize * 0.35, y);
        ctx.closePath();
        ctx.fill();

        // Keeled ridge down scale centerline (Crisp silver highlight on dorsal top scales)
        const distFromDorsal = Math.abs(y - dorsalMidY);
        if (distFromDorsal < height * 0.26) {
          ctx.strokeStyle = 'rgba(240, 246, 255, 0.35)';
          ctx.lineWidth = 1.0;
          ctx.beginPath();
          ctx.moveTo(x - scaleSize * 0.38, y);
          ctx.lineTo(x + scaleSize * 0.38, y);
          ctx.stroke();
        }
      }
    }

    // 5. Tactile Micro-Grain Stippling & Keratin Epidermal Noise
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 32;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
    ctx.putImageData(imgData, 0, 0);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a high-definition 2048x1024 bump/normal relief texture
 * with raised keeled scale facets, deep inter-scale grooves, and stepped belly scutes.
 */
export function generateSnakeBumpTexture(): THREE.CanvasTexture {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    // Neutral mid-gray baseline (128, 128, 128)
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, width, height);

    const dorsalMidY = height * 0.50;
    const bellyEdgeTop = height * 0.14;
    const bellyEdgeBottom = height * 0.86;

    // 1. Raised Imbricate Diamond Scales Relief across Dorsal Zone
    const scaleSize = 9;
    const startRow = Math.floor(bellyEdgeTop / scaleSize);
    const endRow = Math.floor(bellyEdgeBottom / scaleSize) + 1;
    const cols = Math.floor(width / scaleSize) + 2;

    for (let r = startRow; r < endRow; r++) {
      const y = r * scaleSize;
      const offset = (r % 2) * (scaleSize * 0.5);

      for (let c = -1; c < cols; c++) {
        const x = c * scaleSize + offset;

        // Recessed deep dark groove border
        ctx.strokeStyle = '#141414';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(x, y - scaleSize * 0.5);
        ctx.lineTo(x + scaleSize * 0.5, y);
        ctx.lineTo(x, y + scaleSize * 0.5);
        ctx.lineTo(x - scaleSize * 0.5, y);
        ctx.closePath();
        ctx.stroke();

        // Raised scale body facet (bright = raised in bump map)
        ctx.fillStyle = '#c0c0c0';
        ctx.beginPath();
        ctx.moveTo(x, y - scaleSize * 0.35);
        ctx.lineTo(x + scaleSize * 0.35, y);
        ctx.lineTo(x, y + scaleSize * 0.35);
        ctx.lineTo(x - scaleSize * 0.35, y);
        ctx.closePath();
        ctx.fill();

        // Keeled center ridge (High bright peak along top dorsal spine)
        const distFromDorsal = Math.abs(y - dorsalMidY);
        if (distFromDorsal < height * 0.28) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(x - scaleSize * 0.38, y);
          ctx.lineTo(x + scaleSize * 0.38, y);
          ctx.stroke();
        }
      }
    }

    // 2. Stepped Transverse Belly Scutes Relief (Top & Bottom Belly Bands)
    const scuteWidth = 14;
    for (let x = 0; x < width; x += scuteWidth) {
      // Top belly band
      ctx.fillStyle = '#101010';
      ctx.fillRect(x, 0, 2.8, bellyEdgeTop);
      ctx.fillStyle = '#cccccc';
      ctx.fillRect(x + 2.8, 2, scuteWidth - 3.8, bellyEdgeTop - 4);
      ctx.fillStyle = '#f6f6f6';
      ctx.fillRect(x + 2.8, 2, 2.2, bellyEdgeTop - 4);

      // Bottom belly band
      ctx.fillStyle = '#101010';
      ctx.fillRect(x, bellyEdgeBottom, 2.8, height - bellyEdgeBottom);
      ctx.fillStyle = '#cccccc';
      ctx.fillRect(x + 2.8, bellyEdgeBottom + 2, scuteWidth - 3.8, height - bellyEdgeBottom - 4);
      ctx.fillStyle = '#f6f6f6';
      ctx.fillRect(x + 2.8, bellyEdgeBottom + 2, 2.2, height - bellyEdgeBottom - 4);
    }

    // 3. Tactile Micro-Roughness Noise
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const n = (Math.random() - 0.5) * 44;
      const val = Math.max(0, Math.min(255, data[i] + n));
      data[i] = val;
      data[i + 1] = val;
      data[i + 2] = val;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates a realistic PBR roughness texture:
 * Scale faces are sleek & waxy (lower roughness ~0.36-0.45), while inter-scale hinges are rougher (~0.88).
 */
export function generateSnakeRoughnessTexture(): THREE.CanvasTexture {
  const width = 1024;
  const height = 512;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    // Base scale roughness (waxy reptile keratin)
    ctx.fillStyle = '#585858';
    ctx.fillRect(0, 0, width, height);

    const bellyEdgeTop = height * 0.14;
    const bellyEdgeBottom = height * 0.86;

    // Belly bands are smoother/glossier
    ctx.fillStyle = '#383838';
    ctx.fillRect(0, 0, width, bellyEdgeTop);
    ctx.fillRect(0, bellyEdgeBottom, width, height - bellyEdgeBottom);

    // Rough crevices between scales
    ctx.strokeStyle = '#d8d8d8';
    ctx.lineWidth = 1.6;
    const scaleStep = 8;
    for (let x = -scaleStep; x < width + scaleStep; x += scaleStep) {
      ctx.beginPath();
      ctx.moveTo(x, bellyEdgeTop);
      ctx.lineTo(x + scaleStep * 1.5, bellyEdgeBottom);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x + scaleStep * 1.5, bellyEdgeTop);
      ctx.lineTo(x, bellyEdgeBottom);
      ctx.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates an ultra-detailed procedural cranial diffuse texture map (1024x1024)
 * with symmetrical cephalic plates, iconic viper post-ocular eye bandit stripes,
 * dorsal arrowhead/chevron markings, parietal diamond saddle, and barred lip scales.
 */
export function generateSnakeHeadTexture(config: SnakePatternConfig = DEFAULT_SNAKE_PATTERN): THREE.CanvasTexture {
  const width = 1024;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    // 1. Base gunmetal slate-grey background
    ctx.fillStyle = config.dorsalBaseColor;
    ctx.fillRect(0, 0, width, height);

    // Bilateral shading & gradient from dorsal spine (U=0.50) to lateral flanks and underbelly seams
    const grad = ctx.createLinearGradient(0, 0, width, 0);
    grad.addColorStop(0.0, 'rgba(203, 211, 222, 0.95)'); // Under-chin ventral seam
    grad.addColorStop(0.10, 'rgba(10, 12, 16, 0.90)'); // Lower mandible margin
    grad.addColorStop(0.22, 'rgba(24, 28, 35, 0.40)'); // Left lateral flank
    grad.addColorStop(0.36, 'rgba(50, 56, 67, 0.15)'); // Mid-dorsal slate
    grad.addColorStop(0.50, 'rgba(12, 14, 18, 0.35)'); // Top dorsal crown / snout bridge
    grad.addColorStop(0.64, 'rgba(50, 56, 67, 0.15)'); // Mid-dorsal slate
    grad.addColorStop(0.78, 'rgba(24, 28, 35, 0.40)'); // Right lateral flank
    grad.addColorStop(0.90, 'rgba(10, 12, 16, 0.90)'); // Lower mandible margin
    grad.addColorStop(1.0, 'rgba(203, 211, 222, 0.95)'); // Under-chin ventral seam
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // 2. Iconic Viper Temporal Post-Ocular Melanistic Bandit Stripes (Left & Right Flanks)
    const drawBanditStripe = (centerX: number, isLeft: boolean) => {
      ctx.save();
      // Outer bright pearl-silver highlight rim
      ctx.strokeStyle = config.saddleHighlightColor;
      ctx.lineWidth = 14;
      ctx.beginPath();
      ctx.moveTo(centerX + (isLeft ? 36 : -36), 180);
      ctx.quadraticCurveTo(centerX + (isLeft ? 12 : -12), 480, centerX, 760);
      ctx.stroke();

      // Deep jet obsidian black stripe body
      ctx.strokeStyle = config.dorsalDarkSaddle;
      ctx.lineWidth = 26;
      ctx.beginPath();
      ctx.moveTo(centerX + (isLeft ? 36 : -36), 180);
      ctx.quadraticCurveTo(centerX + (isLeft ? 12 : -12), 480, centerX, 760);
      ctx.stroke();

      // Intermediate steel-slate secondary tone
      ctx.strokeStyle = config.saddleSecondaryAccent;
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.moveTo(centerX + (isLeft ? 36 : -36), 200);
      ctx.quadraticCurveTo(centerX + (isLeft ? 12 : -12), 480, centerX, 740);
      ctx.stroke();

      // Fine silver central pinstripe
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(centerX + (isLeft ? 36 : -36), 220);
      ctx.quadraticCurveTo(centerX + (isLeft ? 12 : -12), 480, centerX, 720);
      ctx.stroke();
      ctx.restore();
    };

    drawBanditStripe(width * 0.25, true);
    drawBanditStripe(width * 0.75, false);

    // 3. Canthal Streaks (Snout ridge dark streaks leading from eye corner to nostril / snout tip)
    const drawCanthalStreak = (startX: number, endX: number) => {
      ctx.save();
      ctx.strokeStyle = config.dorsalDarkSaddle;
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.moveTo(startX, 620);
      ctx.lineTo(endX, 940);
      ctx.stroke();

      ctx.strokeStyle = config.saddleHighlightColor;
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.moveTo(startX, 620);
      ctx.lineTo(endX, 940);
      ctx.stroke();
      ctx.restore();
    };
    drawCanthalStreak(width * 0.32, width * 0.44);
    drawCanthalStreak(width * 0.68, width * 0.56);

    // 4. Dorsal Head Markings (Top of the Skull, centered at X = 512)
    const midX = width * 0.50;

    // 4a. Parietal Crown Diamond (Y ~ 260 to 520) - matches body diamondback motif
    const crownCenterY = 380;
    const diamondHalfW = 95;
    const diamondHalfH = 115;

    // Outer pearl-silver halo
    ctx.fillStyle = config.saddleHighlightColor;
    ctx.beginPath();
    ctx.moveTo(midX, crownCenterY - diamondHalfH);
    ctx.lineTo(midX + diamondHalfW, crownCenterY);
    ctx.lineTo(midX, crownCenterY + diamondHalfH);
    ctx.lineTo(midX - diamondHalfW, crownCenterY);
    ctx.closePath();
    ctx.fill();

    // Intermediate steel-slate transition ring
    ctx.fillStyle = config.saddleSecondaryAccent;
    ctx.beginPath();
    ctx.moveTo(midX, crownCenterY - diamondHalfH * 0.82);
    ctx.lineTo(midX + diamondHalfW * 0.82, crownCenterY);
    ctx.lineTo(midX, crownCenterY + diamondHalfH * 0.82);
    ctx.lineTo(midX - diamondHalfW * 0.82, crownCenterY);
    ctx.closePath();
    ctx.fill();

    // Jet obsidian inner diamond saddle
    ctx.fillStyle = config.dorsalDarkSaddle;
    ctx.beginPath();
    ctx.moveTo(midX, crownCenterY - diamondHalfH * 0.64);
    ctx.lineTo(midX + diamondHalfW * 0.64, crownCenterY);
    ctx.lineTo(midX, crownCenterY + diamondHalfH * 0.64);
    ctx.lineTo(midX - diamondHalfW * 0.64, crownCenterY);
    ctx.closePath();
    ctx.fill();

    // Crown center jewel accent dot
    ctx.fillStyle = config.saddleHighlightColor;
    ctx.beginPath();
    ctx.ellipse(midX, crownCenterY, 14, 22, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = config.dorsalDarkSaddle;
    ctx.beginPath();
    ctx.ellipse(midX, crownCenterY, 6, 10, 0, 0, Math.PI * 2);
    ctx.fill();

    // 4b. Anterior Arrowhead Spear Chevron pointing towards Snout Apex (Y ~ 540 to 860)
    ctx.fillStyle = config.saddleHighlightColor;
    ctx.beginPath();
    ctx.moveTo(midX, 860); // Spear tip pointing to snout
    ctx.lineTo(midX + 78, 620); // Right wing
    ctx.lineTo(midX + 44, 630);
    ctx.lineTo(midX, 740);
    ctx.lineTo(midX - 44, 630);
    ctx.lineTo(midX - 78, 620); // Left wing
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = config.dorsalDarkSaddle;
    ctx.beginPath();
    ctx.moveTo(midX, 846);
    ctx.lineTo(midX + 66, 632);
    ctx.lineTo(midX + 38, 642);
    ctx.lineTo(midX, 730);
    ctx.lineTo(midX - 38, 642);
    ctx.lineTo(midX - 66, 632);
    ctx.closePath();
    ctx.fill();

    // 5. Cephalic Shield Armor Seams (Polygonal Head Scales)
    ctx.strokeStyle = 'rgba(230, 238, 250, 0.48)';
    ctx.lineWidth = 2.0;

    const scalePolys = [
      // Frontal shield
      [[midX - 50, 550], [midX + 50, 550], [midX + 40, 680], [midX, 720], [midX - 40, 680], [midX - 50, 550]],
      // Left Supraocular shield
      [[midX - 60, 530], [midX - 165, 570], [midX - 145, 710], [midX - 55, 660]],
      // Right Supraocular shield
      [[midX + 60, 530], [midX + 165, 570], [midX + 145, 710], [midX + 55, 660]],
      // Prefrontal shields
      [[midX - 40, 680], [midX - 95, 770], [midX, 820], [midX + 95, 770], [midX + 40, 680]],
      // Internasal & Rostral scales
      [[midX - 60, 830], [midX - 25, 940], [midX + 25, 940], [midX + 60, 830]],
      // Occipital rear plates
      [[midX - 120, 200], [midX - 50, 260], [midX + 50, 260], [midX + 120, 200]],
      [[midX - 160, 240], [midX - 80, 340]],
      [[midX + 160, 240], [midX + 80, 340]],
    ];

    for (const poly of scalePolys) {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i][0], poly[i][1]);
      }
      ctx.stroke();
    }

    // 6. Supralabial and Infralabial Barred Lip Margin Scales
    const drawLabialBars = (marginStartX: number, marginEndX: number) => {
      const barStep = 24;
      for (let y = 260; y < 920; y += barStep) {
        ctx.fillStyle = ((y / barStep) % 2 === 0) ? 'rgba(10, 12, 16, 0.92)' : 'rgba(215, 224, 236, 0.88)';
        ctx.fillRect(marginStartX, y, marginEndX - marginStartX, barStep - 4);

        ctx.strokeStyle = 'rgba(6, 8, 10, 0.95)';
        ctx.lineWidth = 2.0;
        ctx.strokeRect(marginStartX, y, marginEndX - marginStartX, barStep - 4);
      }
    };
    drawLabialBars(width * 0.07, width * 0.17);
    drawLabialBars(width * 0.83, width * 0.93);

    // 7. Micro Keeling Highlight Lines on Dorsal Scales
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.42)';
    ctx.lineWidth = 1.6;
    for (let y = 300; y < 940; y += 36) {
      const wSpan = 140 * (1.0 - (y - 300) / 1200);
      ctx.beginPath();
      ctx.moveTo(midX - wSpan * 0.5, y);
      ctx.lineTo(midX + wSpan * 0.5, y);
      ctx.stroke();
    }

    // 8. Granular Keratin Scale Stippling Noise
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const n = (Math.random() - 0.5) * 18;
      data[i] = Math.max(0, Math.min(255, data[i] + n));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + n));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + n));
    }
    ctx.putImageData(imgData, 0, 0);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates an ultra-detailed procedural bump map for the snake head (1024x1024)
 * providing high tactile relief for cephalic plates, canthal ridges, and scale keels.
 */
export function generateSnakeHeadBumpTexture(): THREE.CanvasTexture {
  const width = 1024;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    // Neutral grey baseline
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, width, height);

    const midX = width * 0.50;

    // 1. Raised Parietal Crown Diamond Relief
    const crownCenterY = 380;
    const diamondHalfW = 95;
    const diamondHalfH = 115;

    ctx.fillStyle = '#b5b5b5';
    ctx.beginPath();
    ctx.moveTo(midX, crownCenterY - diamondHalfH);
    ctx.lineTo(midX + diamondHalfW, crownCenterY);
    ctx.lineTo(midX, crownCenterY + diamondHalfH);
    ctx.lineTo(midX - diamondHalfW, crownCenterY);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#e8e8e8';
    ctx.beginPath();
    ctx.moveTo(midX, crownCenterY - diamondHalfH * 0.64);
    ctx.lineTo(midX + diamondHalfW * 0.64, crownCenterY);
    ctx.lineTo(midX, crownCenterY + diamondHalfH * 0.64);
    ctx.lineTo(midX - diamondHalfW * 0.64, crownCenterY);
    ctx.closePath();
    ctx.fill();

    // High jewel peak
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.ellipse(midX, crownCenterY, 14, 22, 0, 0, Math.PI * 2);
    ctx.fill();

    // 2. Raised Snout Arrowhead Relief
    ctx.fillStyle = '#e0e0e0';
    ctx.beginPath();
    ctx.moveTo(midX, 860);
    ctx.lineTo(midX + 78, 620);
    ctx.lineTo(midX + 44, 630);
    ctx.lineTo(midX, 740);
    ctx.lineTo(midX - 44, 630);
    ctx.lineTo(midX - 78, 620);
    ctx.closePath();
    ctx.fill();

    // Center spine ridge
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4.0;
    ctx.beginPath();
    ctx.moveTo(midX, 220);
    ctx.lineTo(midX, 940);
    ctx.stroke();

    // 3. Cephalic Shield Recessed Grooves
    ctx.strokeStyle = '#181818';
    ctx.lineWidth = 4.0;

    const scalePolys = [
      [[midX - 50, 550], [midX + 50, 550], [midX + 40, 680], [midX, 720], [midX - 40, 680], [midX - 50, 550]],
      [[midX - 60, 530], [midX - 165, 570], [midX - 145, 710], [midX - 55, 660]],
      [[midX + 60, 530], [midX + 165, 570], [midX + 145, 710], [midX + 55, 660]],
      [[midX - 40, 680], [midX - 95, 770], [midX, 820], [midX + 95, 770], [midX + 40, 680]],
      [[midX - 60, 830], [midX - 25, 940], [midX + 25, 940], [midX + 60, 830]],
    ];

    for (const poly of scalePolys) {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i][0], poly[i][1]);
      }
      ctx.stroke();
    }

    // 4. Stepped Labial Lip Scale Bars
    const drawLabialBump = (marginStartX: number, marginEndX: number) => {
      const barStep = 24;
      for (let y = 260; y < 920; y += barStep) {
        ctx.fillStyle = ((y / barStep) % 2 === 0) ? '#303030' : '#e0e0e0';
        ctx.fillRect(marginStartX, y, marginEndX - marginStartX, barStep - 4);

        ctx.strokeStyle = '#050505';
        ctx.lineWidth = 3.0;
        ctx.strokeRect(marginStartX, y, marginEndX - marginStartX, barStep - 4);
      }
    };
    drawLabialBump(width * 0.07, width * 0.17);
    drawLabialBump(width * 0.83, width * 0.93);

    // 5. Tactile Micro-Noise
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const n = (Math.random() - 0.5) * 32;
      const val = Math.max(0, Math.min(255, data[i] + n));
      data[i] = val;
      data[i + 1] = val;
      data[i + 2] = val;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates an ultra-detailed procedural roughness texture for the snake head (1024x1024)
 * with waxy sheen on scale plate facets and higher roughness in scale recesses.
 */
export function generateSnakeHeadRoughnessTexture(): THREE.CanvasTexture {
  const width = 1024;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    // Base waxy scale roughness
    ctx.fillStyle = '#545454';
    ctx.fillRect(0, 0, width, height);

    const midX = width * 0.50;

    // Smoother polished crown diamond and arrowhead
    ctx.fillStyle = '#383838';
    ctx.beginPath();
    ctx.arc(midX, 380, 90, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(midX, 720, 80, 0, Math.PI * 2);
    ctx.fill();

    // Rough inter-scale seams
    ctx.strokeStyle = '#d5d5d5';
    ctx.lineWidth = 3.0;

    const scalePolys = [
      [[midX - 50, 550], [midX + 50, 550], [midX + 40, 680], [midX, 720], [midX - 40, 680], [midX - 50, 550]],
      [[midX - 60, 530], [midX - 165, 570], [midX - 145, 710], [midX - 55, 660]],
      [[midX + 60, 530], [midX + 165, 570], [midX + 145, 710], [midX + 55, 660]],
      [[midX - 40, 680], [midX - 95, 770], [midX, 820], [midX + 95, 770], [midX + 40, 680]],
      [[midX - 60, 830], [midX - 25, 940], [midX + 25, 940], [midX + 60, 830]],
    ];

    for (const poly of scalePolys) {
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i][0], poly[i][1]);
      }
      ctx.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates an authentic high-resolution eye iris texture
 * with striated fiery golden-amber/topaz reptilian iris fibers and predator depth.
 */
function generateReptilianIrisTexture(): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    const cx = size / 2;
    const cy = size / 2;
    const r = size / 2;

    // Base iris color (striking radiant golden amber with deep charcoal limbal ring)
    const baseGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    baseGrad.addColorStop(0.0, '#1a1005'); // Deep pupil boundary
    baseGrad.addColorStop(0.20, '#fef08a'); // Luminous golden-yellow inner ring
    baseGrad.addColorStop(0.50, '#f59e0b'); // Vivid amber-gold iris body
    baseGrad.addColorStop(0.78, '#b45309'); // Deep burnt amber
    baseGrad.addColorStop(0.92, '#1e293b'); // Dark obsidian outer limbal ring
    baseGrad.addColorStop(1.0, '#0a0d14');
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, size, size);

    // Radial striated luminous fibers
    ctx.lineWidth = 1.2;
    const numRays = 220;
    for (let i = 0; i < numRays; i++) {
      const angle = (i / numRays) * Math.PI * 2;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const innerR = r * (0.12 + Math.random() * 0.08);
      const outerR = r * (0.86 + Math.random() * 0.10);

      const grad = ctx.createLinearGradient(
        cx + cosA * innerR,
        cy + sinA * innerR,
        cx + cosA * outerR,
        cy + sinA * outerR
      );
      grad.addColorStop(0.0, 'rgba(255, 255, 255, 0.95)'); // Pure white/gold fiber core
      grad.addColorStop(0.35, 'rgba(254, 240, 138, 0.85)'); // Bright golden filament
      grad.addColorStop(0.70, 'rgba(245, 158, 11, 0.60)');  // Warm amber body
      grad.addColorStop(1.0, 'rgba(30, 41, 59, 0.90)');

      ctx.strokeStyle = grad;
      ctx.beginPath();
      ctx.moveTo(cx + cosA * innerR, cy + sinA * innerR);
      ctx.lineTo(cx + cosA * outerR, cy + sinA * outerR);
      ctx.stroke();
    }

    // Concentric micro-contractile iris ridges
    for (let cr = 35; cr < r * 0.88; cr += 18) {
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.25)';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.arc(cx, cy, cr, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

/**
 * Generates an emissive map for the glowing predator iris
 * making the inner luminous ring and radiating neural fibers glow intensely in 3D.
 */
function generateReptilianIrisEmissiveTexture(): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    const cx = size / 2;
    const cy = size / 2;
    const r = size / 2;

    // Dark black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, size, size);

    // Glowing core ring gradient
    const glowGrad = ctx.createRadialGradient(cx, cy, r * 0.10, cx, cy, r * 0.75);
    glowGrad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
    glowGrad.addColorStop(0.25, 'rgba(254, 240, 138, 0.90)');
    glowGrad.addColorStop(0.60, 'rgba(245, 158, 11, 0.50)');
    glowGrad.addColorStop(0.90, 'rgba(0, 0, 0, 0.0)');
    glowGrad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');
    ctx.fillStyle = glowGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.75, 0, Math.PI * 2);
    ctx.fill();

    // Striated emissive filaments
    ctx.lineWidth = 1.5;
    const numEmissiveRays = 140;
    for (let i = 0; i < numEmissiveRays; i++) {
      const angle = (i / numEmissiveRays) * Math.PI * 2;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const innerR = r * 0.15;
      const outerR = r * (0.65 + Math.random() * 0.12);

      const grad = ctx.createLinearGradient(
        cx + cosA * innerR,
        cy + sinA * innerR,
        cx + cosA * outerR,
        cy + sinA * outerR
      );
      grad.addColorStop(0.0, 'rgba(255, 255, 255, 0.9)');
      grad.addColorStop(0.5, 'rgba(251, 191, 36, 0.7)');
      grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

      ctx.strokeStyle = grad;
      ctx.beginPath();
      ctx.moveTo(cx + cosA * innerR, cy + sinA * innerR);
      ctx.lineTo(cx + cosA * outerR, cy + sinA * outerR);
      ctx.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

export interface SnakeDebugVisuals {
  group: THREE.Group;
  enabled: boolean;
  ringsMesh: THREE.LineSegments;
  spinePointsMesh: THREE.Points;
  framesMesh: THREE.LineSegments;
  buffers: {
    ringPositions: Float32Array;
    spinePointPositions: Float32Array;
    framePositions: Float32Array;
    frameColors: Float32Array;
  };
}

export interface SnakeRig {
  root: THREE.Group;
  headGroup: THREE.Group;
  craniumMesh: THREE.Mesh;
  crownCrestMesh?: THREE.Group | THREE.Mesh;
  jawHingeGroup: THREE.Group;
  jawMesh: THREE.Mesh;
  leftEyeGroup: THREE.Group;
  rightEyeGroup: THREE.Group;
  leftEyeGlobe?: THREE.Mesh;
  rightEyeGlobe?: THREE.Mesh;
  leftPupil: THREE.Mesh;
  rightPupil: THREE.Mesh;
  leftEyeCorona?: THREE.Mesh;
  rightEyeCorona?: THREE.Mesh;
  leftEyeLight?: THREE.PointLight;
  rightEyeLight?: THREE.PointLight;
  leftFang: THREE.Mesh;
  rightFang: THREE.Mesh;
  tongueGroup: THREE.Group;
  tongueMesh?: THREE.Mesh;
  leftHorn?: THREE.Mesh;
  rightHorn?: THREE.Mesh;
  tactileWhiskers?: THREE.Group;
  thermalPits?: THREE.Group;
  bodyMesh: THREE.Mesh;
  bodyGeometry: THREE.BufferGeometry;
  bodyMaterial: THREE.MeshStandardMaterial;
  bodyTexture: THREE.CanvasTexture;
  headTexture?: THREE.CanvasTexture;
  headBumpTexture?: THREE.CanvasTexture;
  headRoughnessTexture?: THREE.CanvasTexture;
  debug: SnakeDebugVisuals;
  materials: {
    head: THREE.MeshStandardMaterial;
    headDorsal: THREE.MeshStandardMaterial;
    headVentral: THREE.MeshStandardMaterial;
    body: THREE.MeshStandardMaterial;
    eye: THREE.MeshStandardMaterial;
    eyeCorona?: THREE.MeshBasicMaterial;
    pupil: THREE.MeshBasicMaterial;
    fang: THREE.MeshStandardMaterial;
    tongue: THREE.MeshStandardMaterial;
    crown?: THREE.MeshStandardMaterial;
    horn?: THREE.MeshStandardMaterial;
    thermalPit?: THREE.MeshStandardMaterial;
  };
  buffers: {
    positions: Float32Array;
    normals: Float32Array;
    spineCenters: Float32Array;
    spineTangents: Float32Array;
    spineNormals: Float32Array;
    spineBinormals: Float32Array;
    arcLengths: Float32Array;
  };
  telemetry: {
    totalArcLength: number;
    headScale: number;
    bodyThickness: number;
    growthFactor: number;
    jawAngleDeg: number;
    eyeGazeAngle: number;
    crownCharge: number;
  };
}

// Reusable scratch vectors for zero-GC spine math
const vNextPos = new THREE.Vector3();
const vPrevPos = new THREE.Vector3();
const vTangent = new THREE.Vector3();
const vUp = new THREE.Vector3();
const vRight = new THREE.Vector3();
const vAxis = new THREE.Vector3();
const vPrevTangent = new THREE.Vector3();
const vPrevUp = new THREE.Vector3();
const vRotatedUp = new THREE.Vector3();
const qRot = new THREE.Quaternion();

// Eye gaze tracking scratch vectors (Zero-GC stereoscopic tracking)
const vGazeTarget = new THREE.Vector3();
const vHeadWorld = new THREE.Vector3();
const vTargetDir = new THREE.Vector3();
const vLocalTarget = new THREE.Vector3();
const vLeftDir = new THREE.Vector3();
const vRightDir = new THREE.Vector3();
const vLeftEyePosLocal = new THREE.Vector3();
const vRightEyePosLocal = new THREE.Vector3();
const vLocalEyeDir = new THREE.Vector3();
const mHeadWorldInv = new THREE.Matrix4();
const cPredatorAmber = new THREE.Color(0xf59e0b);
const cStrikeCrimson = new THREE.Color(0xf43f5e);
const cDamageCyan = new THREE.Color(0x38bdf8);
const cEyeLerp = new THREE.Color();

// Catmull-Rom scratch vectors
const vP0 = new THREE.Vector3();
const vP1 = new THREE.Vector3();
const vP2 = new THREE.Vector3();
const vP3 = new THREE.Vector3();

/**
 * Creates an anatomically realistic 3D snake rig modeled after real vipers and pythons:
 * - Streamlined triangular/spade-shaped skull with wide temporal venom arches and canthal ridges
 * - Authentic supraocular brow scales shielding realistic reptilian eyes with vertical slit pupils
 * - Form-fitting lower mandible with gular throat skin and retractable needle fangs
 * - Chemosensory forked tongue flicking through the sub-rostral notch
 * - Silky smooth 24-sided continuous body with 96 rings and realistic PBR scale textures
 */
export function createSnakeRig(_maxSegments: number = 1000): SnakeRig {
  const root = new THREE.Group();
  root.name = 'real_world_snake_rig_root';

  // ---------------------------------------------------------------------------
  // 1. Realistic PBR Materials & High-Res Textures
  // ---------------------------------------------------------------------------
  const bodyTexture = generateSnakeBodyTexture(DEFAULT_SNAKE_PATTERN);
  const bumpTexture = generateSnakeBumpTexture();
  const roughnessTexture = generateSnakeRoughnessTexture();
  const irisTexture = generateReptilianIrisTexture();
  const irisEmissiveTexture = generateReptilianIrisEmissiveTexture();
  const headTexture = generateSnakeHeadTexture(DEFAULT_SNAKE_PATTERN);
  const headBumpTexture = generateSnakeHeadBumpTexture();
  const headRoughnessTexture = generateSnakeRoughnessTexture();

  // Primary Body Scale Material (Authentic Keratin Reptile Scales in Slate & Obsidian)
  const bodyMaterial = new THREE.MeshStandardMaterial({
    map: bodyTexture,
    bumpMap: bumpTexture,
    bumpScale: 0.085,
    roughnessMap: roughnessTexture,
    roughness: 0.48, // Waxy keratin sheen
    metalness: 0.06,
    emissive: 0x000000,
    emissiveIntensity: 0.0,
    flatShading: false,
    side: THREE.DoubleSide,
  });

  // Primary Head Cranial Scales (Matches Body Slate-Grey Base & Head Markings)
  const headMaterial = new THREE.MeshStandardMaterial({
    map: headTexture,
    bumpMap: headBumpTexture,
    bumpScale: 0.085,
    roughnessMap: headRoughnessTexture,
    roughness: 0.44,
    metalness: 0.06,
    flatShading: false,
  });

  // Dorsal Brow & Crown Plates (Dark obsidian-charcoal armor plates with high bump relief)
  const headDorsalMaterial = new THREE.MeshStandardMaterial({
    map: headTexture,
    bumpMap: headBumpTexture,
    bumpScale: 0.095,
    roughnessMap: headRoughnessTexture,
    roughness: 0.38,
    metalness: 0.08,
    flatShading: false,
  });

  // Ventral Jaw & Gular Throat Shield (Silver-slate ventral scutes)
  const headVentralMaterial = new THREE.MeshStandardMaterial({
    color: 0xcbd3de, // Pale silver-slate
    bumpMap: bumpTexture,
    bumpScale: 0.060,
    roughness: 0.52,
    metalness: 0.04,
    flatShading: false,
  });

  // Mouth Interior / Epithelium Lining (Realistic mucous membrane)
  const mouthLiningMaterial = new THREE.MeshStandardMaterial({
    color: 0x5a4850, // Slate-tinted pinkish-mauve
    roughness: 0.28,
    metalness: 0.08,
  });

  // Retractable Needle Fangs (Ivory bone with translucent gloss)
  const fangMaterial = new THREE.MeshStandardMaterial({
    color: 0xf1f5f9, // Ivory white bone
    roughness: 0.20,
    metalness: 0.15,
  });

  // Real Glowing Reactive Reptilian Iris (Striated fiery amber/gold predator eye)
  const eyeIrisMaterial = new THREE.MeshStandardMaterial({
    map: irisTexture,
    emissiveMap: irisEmissiveTexture,
    emissive: new THREE.Color(0xf59e0b),
    emissiveIntensity: 1.5,
    roughness: 0.15,
    metalness: 0.12,
  });

  // Glowing Eye Corona Lens Flare (Additive luminous aura)
  const eyeCoronaMaterial = new THREE.MeshBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  // Pupil Slit Material (Deep obsidian black)
  const pupilMaterial = new THREE.MeshBasicMaterial({
    color: 0x020304,
  });

  // Infrared Loreal Pit Sensor Disc (Thermal nerve membrane with glowing core)
  const thermalPitMaterial = new THREE.MeshStandardMaterial({
    color: 0x2a0808,
    emissive: 0xef4444,
    emissiveIntensity: 0.45,
    roughness: 0.30,
    metalness: 0.20,
  });

  // Horn & Armored Crown Material (Keratin armor plates)
  const hornMaterial = new THREE.MeshStandardMaterial({
    color: 0x080a0e,
    bumpMap: headBumpTexture,
    bumpScale: 0.08,
    roughness: 0.35,
    metalness: 0.12,
  });

  // Armored Crown Crest Material (Conductive kinetic crest scutes)
  const crownMaterial = new THREE.MeshStandardMaterial({
    color: 0x161922,
    emissive: 0x38bdf8,
    emissiveIntensity: 0.0,
    bumpMap: headBumpTexture,
    bumpScale: 0.09,
    roughness: 0.32,
    metalness: 0.25,
  });

  // Chemosensory Forked Tongue (Deep obsidian-charcoal with gloss finish)
  const tongueMaterial = new THREE.MeshStandardMaterial({
    color: 0x0d0f12, // Pitch charcoal
    roughness: 0.28,
    metalness: 0.08,
  });

  // ---------------------------------------------------------------------------
  // 2. High-Resolution 24-Sided Smooth Tubular BufferGeometry (96 Rings)
  // ---------------------------------------------------------------------------
  const totalVerts = NUM_RINGS * VERTS_PER_RING + 1;
  const tailTipIndex = NUM_RINGS * VERTS_PER_RING;

  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const uvs = new Float32Array(totalVerts * 2);

  // Precompute static UVs
  for (let r = 0; r < NUM_RINGS; r++) {
    const u = r / (NUM_RINGS - 1);
    for (let k = 0; k < VERTS_PER_RING; k++) {
      const v = k / RADIAL_SEGMENTS;
      const vertIdx = r * VERTS_PER_RING + k;
      uvs[vertIdx * 2 + 0] = u;
      uvs[vertIdx * 2 + 1] = v;
    }
  }
  uvs[tailTipIndex * 2 + 0] = 1.0;
  uvs[tailTipIndex * 2 + 1] = 0.5;

  const ringTriangles = (NUM_RINGS - 1) * RADIAL_SEGMENTS * 2;
  const tailTriangles = RADIAL_SEGMENTS;
  const totalTriangles = ringTriangles + tailTriangles;
  const indices = new Uint16Array(totalTriangles * 3);

  let indexOffset = 0;
  for (let r = 0; r < NUM_RINGS - 1; r++) {
    const rowA = r * VERTS_PER_RING;
    const rowB = (r + 1) * VERTS_PER_RING;

    for (let k = 0; k < RADIAL_SEGMENTS; k++) {
      const a0 = rowA + k;
      const a1 = rowA + k + 1;
      const b0 = rowB + k;
      const b1 = rowB + k + 1;

      // Outward-facing triangles (CCW seen from exterior)
      indices[indexOffset++] = a0;
      indices[indexOffset++] = a1;
      indices[indexOffset++] = b0;

      indices[indexOffset++] = a1;
      indices[indexOffset++] = b1;
      indices[indexOffset++] = b0;
    }
  }

  const lastRow = (NUM_RINGS - 1) * VERTS_PER_RING;
  for (let k = 0; k < RADIAL_SEGMENTS; k++) {
    indices[indexOffset++] = lastRow + k;
    indices[indexOffset++] = lastRow + k + 1;
    indices[indexOffset++] = tailTipIndex;
  }

  const bodyGeometry = new THREE.BufferGeometry();
  bodyGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  bodyGeometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  bodyGeometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  bodyGeometry.setIndex(new THREE.BufferAttribute(indices, 1));

  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  bodyMesh.castShadow = true;
  bodyMesh.receiveShadow = true;
  bodyMesh.frustumCulled = false;
  root.add(bodyMesh);

  // ---------------------------------------------------------------------------
  // 3. Anatomically Realistic Viper / Python Head
  // ---------------------------------------------------------------------------
  const headGroup = new THREE.Group();
  headGroup.name = 'real_world_snake_head';

  const headRadius = BASE_RADIUS * 0.72; // Seamlessly matches Ring 0 radius

  // 3a. Main Anatomical Skull / Cranium (Triangular Viper Spade Shape)
  // Sculpted with smooth organic contours, wide temporal venom arches, and tapered snout
  const skullLength = 1.35;
  const skullWidth = headRadius * 1.55;
  const skullHeight = headRadius * 0.92;

  // Upper Cranium Body (Curved organic shell with 32 radial segments)
  const skullGeo = new THREE.CylinderGeometry(
    skullWidth * 0.38, // Anterior snout width
    skullWidth * 0.95, // Posterior temporal lobes width
    skullLength,
    32,
    12,
    false
  );
  skullGeo.rotateX(Math.PI / 2);
  skullGeo.scale(1.0, 0.58, 1.0);
  skullGeo.translate(0, skullHeight * 0.12, skullLength * 0.42);

  const craniumMesh = new THREE.Mesh(skullGeo, headMaterial);
  craniumMesh.castShadow = true;
  craniumMesh.receiveShadow = true;
  headGroup.add(craniumMesh);

  // 3b. Posterior Venom Gland Bulges (Smooth organic temporal swells on each rear flank)
  const venomLobeGeo = new THREE.SphereGeometry(headRadius * 0.48, 28, 20);
  venomLobeGeo.scale(0.88, 0.62, 1.35);

  const leftVenomLobe = new THREE.Mesh(venomLobeGeo, headMaterial);
  leftVenomLobe.position.set(headRadius * 0.72, headRadius * 0.08, 0.18);
  leftVenomLobe.rotation.set(0.1, 0.25, -0.15);
  craniumMesh.add(leftVenomLobe);

  const rightVenomLobe = new THREE.Mesh(venomLobeGeo, headMaterial);
  rightVenomLobe.position.set(-headRadius * 0.72, headRadius * 0.08, 0.18);
  rightVenomLobe.rotation.set(0.1, -0.25, 0.15);
  craniumMesh.add(rightVenomLobe);

  // 3c. Anatomical Neck Mantle & Seamless Body Collar (32-segment smooth ring encasing Ring 0)
  const neckCollarGeo = new THREE.CylinderGeometry(
    headRadius * 1.06,
    headRadius * 0.98,
    0.46,
    32,
    4,
    false
  );
  neckCollarGeo.rotateX(Math.PI / 2);
  neckCollarGeo.scale(1.15, 0.75, 1.0);
  neckCollarGeo.translate(0, headRadius * 0.05, -0.16);

  const neckCollarMesh = new THREE.Mesh(neckCollarGeo, headMaterial);
  neckCollarMesh.castShadow = true;
  neckCollarMesh.receiveShadow = true;
  headGroup.add(neckCollarMesh);

  // 3d. Realistic Head Scute Plates (Dorsal Parietal Diamond, Frontal Shield, and Occipital Collars)
  // Parietal Crown Diamond Plate (Matches body diamondback motif in high-relief keratin)
  const parietalDiamondGeo = new THREE.ConeGeometry(headRadius * 0.38, 0.45, 4);
  parietalDiamondGeo.rotateY(Math.PI / 4);
  parietalDiamondGeo.rotateX(Math.PI / 2);
  parietalDiamondGeo.scale(1.0, 0.22, 1.25);
  parietalDiamondGeo.translate(0, headRadius * 0.36, 0.34);
  const parietalDiamond = new THREE.Mesh(parietalDiamondGeo, headDorsalMaterial);
  craniumMesh.add(parietalDiamond);

  // Frontal Cephalic Shield (Sculpted plate shielding mid-snout bridge)
  const frontalPlateGeo = new THREE.CylinderGeometry(
    headRadius * 0.28,
    headRadius * 0.36,
    0.48,
    28,
    4,
    false,
    0,
    Math.PI
  );
  frontalPlateGeo.rotateZ(Math.PI / 2);
  frontalPlateGeo.scale(0.85, 0.24, 1.0);
  frontalPlateGeo.translate(0, headRadius * 0.30, 0.58);
  const frontalPlate = new THREE.Mesh(frontalPlateGeo, headDorsalMaterial);
  craniumMesh.add(frontalPlate);

  // Canthal Ridge Keels (Smooth tapered snout bridge ridges along nasal bones)
  const canthalBridgeGeo = new THREE.ConeGeometry(headRadius * 0.34, 0.65, 24);
  canthalBridgeGeo.rotateX(Math.PI / 2);
  canthalBridgeGeo.scale(1.0, 0.38, 1.0);
  canthalBridgeGeo.translate(0, headRadius * 0.26, 0.80);
  const canthalBridge = new THREE.Mesh(canthalBridgeGeo, headDorsalMaterial);
  craniumMesh.add(canthalBridge);

  // Bilateral Canthal Ridge Spines (Left & Right keeled lateral ridges)
  const keelGeo = new THREE.CylinderGeometry(0.016, 0.024, 0.52, 16);
  keelGeo.rotateX(Math.PI / 2);

  const leftCanthalKeel = new THREE.Mesh(keelGeo, headDorsalMaterial);
  leftCanthalKeel.position.set(headRadius * 0.36, headRadius * 0.22, 0.72);
  leftCanthalKeel.rotation.set(-0.12, 0.22, -0.25);
  craniumMesh.add(leftCanthalKeel);

  const rightCanthalKeel = new THREE.Mesh(keelGeo, headDorsalMaterial);
  rightCanthalKeel.position.set(-headRadius * 0.36, headRadius * 0.22, 0.72);
  rightCanthalKeel.rotation.set(-0.12, -0.22, 0.25);
  craniumMesh.add(rightCanthalKeel);

  // 3e. Supraocular Brow Ridges & Subocular Eye Socket Rims (Deep Hooded Predator Eyes)
  const browGeo = new THREE.SphereGeometry(headRadius * 0.18, 22, 16);
  browGeo.scale(1.42, 0.44, 2.3);

  const leftBrow = new THREE.Mesh(browGeo, headDorsalMaterial);
  leftBrow.position.set(headRadius * 0.68, headRadius * 0.30, 0.50);
  leftBrow.rotation.set(0.12, 0.22, -0.32);
  craniumMesh.add(leftBrow);

  const rightBrow = new THREE.Mesh(browGeo, headDorsalMaterial);
  rightBrow.position.set(-headRadius * 0.68, headRadius * 0.30, 0.50);
  rightBrow.rotation.set(0.12, -0.22, 0.32);
  craniumMesh.add(rightBrow);

  // Subocular Orbital Scale Rims (Form-fitting lower orbital boundary)
  const subocularGeo = new THREE.TorusGeometry(0.11, 0.022, 12, 24, Math.PI * 0.85);
  subocularGeo.rotateZ(Math.PI * 0.08);

  const leftSubocular = new THREE.Mesh(subocularGeo, headMaterial);
  leftSubocular.position.set(headRadius * 0.74, headRadius * 0.12, 0.52);
  leftSubocular.rotation.set(0, Math.PI / 2, -0.20);
  craniumMesh.add(leftSubocular);

  const rightSubocular = new THREE.Mesh(subocularGeo, headMaterial);
  rightSubocular.position.set(-headRadius * 0.74, headRadius * 0.12, 0.52);
  rightSubocular.rotation.set(0, -Math.PI / 2, 0.20);
  craniumMesh.add(rightSubocular);

  // Postocular Stepped Temporal Scutes (Overlapping scales on venom lobes)
  const tempPlateGeo = new THREE.CylinderGeometry(headRadius * 0.22, headRadius * 0.28, 0.42, 20, 2, false, 0, Math.PI);
  tempPlateGeo.rotateZ(Math.PI / 2);
  tempPlateGeo.scale(1.0, 0.28, 1.0);

  const leftTempPlate = new THREE.Mesh(tempPlateGeo, headMaterial);
  leftTempPlate.position.set(headRadius * 0.64, headRadius * 0.14, 0.18);
  leftTempPlate.rotation.set(0.15, 0.35, -0.22);
  craniumMesh.add(leftTempPlate);

  const rightTempPlate = new THREE.Mesh(tempPlateGeo, headMaterial);
  rightTempPlate.position.set(-headRadius * 0.64, headRadius * 0.14, 0.18);
  rightTempPlate.rotation.set(0.15, -0.35, 0.22);
  craniumMesh.add(rightTempPlate);

  // 3f. Rounded Rostral Snout Apex & Sub-Rostral Lingual Notch (Smooth 24-segment spheroid)
  const rostralGeo = new THREE.SphereGeometry(headRadius * 0.24, 24, 18);
  rostralGeo.scale(1.1, 0.65, 0.9);
  const rostralMesh = new THREE.Mesh(rostralGeo, headDorsalMaterial);
  rostralMesh.position.set(0, headRadius * 0.06, 1.08);
  craniumMesh.add(rostralMesh);

  // Supralabial Lip Scale Rows (Upper lip margin with 24 segments)
  const labialGeo = new THREE.CylinderGeometry(headRadius * 0.09, headRadius * 0.11, 0.85, 24);
  labialGeo.rotateX(Math.PI / 2);

  const leftLabial = new THREE.Mesh(labialGeo, headMaterial);
  leftLabial.position.set(headRadius * 0.56, -headRadius * 0.08, 0.58);
  leftLabial.rotation.set(0, 0.18, 0);
  craniumMesh.add(leftLabial);

  const rightLabial = new THREE.Mesh(labialGeo, headMaterial);
  rightLabial.position.set(-headRadius * 0.56, -headRadius * 0.08, 0.58);
  rightLabial.rotation.set(0, -0.18, 0);
  craniumMesh.add(rightLabial);

  // 3g. Nostril Cavities (Nares) & Infrared Loreal Heat-Sensing Pits
  // Nares Cavity & Elevated Scale Rims
  const nostrilGeo = new THREE.SphereGeometry(0.024, 16, 12);
  nostrilGeo.scale(0.8, 1.2, 1.0);

  const leftNostril = new THREE.Mesh(nostrilGeo, pupilMaterial);
  leftNostril.position.set(0.11, headRadius * 0.16, 0.98);
  craniumMesh.add(leftNostril);

  const rightNostril = new THREE.Mesh(nostrilGeo, pupilMaterial);
  rightNostril.position.set(-0.11, headRadius * 0.16, 0.98);
  craniumMesh.add(rightNostril);

  const nostrilRimGeo = new THREE.TorusGeometry(0.026, 0.008, 10, 16);
  const leftNostrilRim = new THREE.Mesh(nostrilRimGeo, headDorsalMaterial);
  leftNostrilRim.position.set(0.11, headRadius * 0.16, 0.98);
  leftNostrilRim.rotation.set(0.2, 0.4, 0);
  craniumMesh.add(leftNostrilRim);

  const rightNostrilRim = new THREE.Mesh(nostrilRimGeo, headDorsalMaterial);
  rightNostrilRim.position.set(-0.11, headRadius * 0.16, 0.98);
  rightNostrilRim.rotation.set(0.2, -0.4, 0);
  craniumMesh.add(rightNostrilRim);

  // Infrared Loreal Heat Pits (Deep dark cavity + inner glowing thermal sensor disc)
  const thermalPitsGroup = new THREE.Group();
  thermalPitsGroup.name = 'infrared_loreal_heat_pits';

  const pitGeo = new THREE.SphereGeometry(0.032, 16, 12);
  pitGeo.scale(0.7, 1.3, 1.0);

  const leftPit = new THREE.Mesh(pitGeo, pupilMaterial);
  leftPit.position.set(headRadius * 0.44, headRadius * 0.10, 0.78);
  thermalPitsGroup.add(leftPit);

  const rightPit = new THREE.Mesh(pitGeo, pupilMaterial);
  rightPit.position.set(-headRadius * 0.44, headRadius * 0.10, 0.78);
  thermalPitsGroup.add(rightPit);

  // Glowing Thermal Sensor Nerve Disc inside the pit
  const sensorDiscGeo = new THREE.CircleGeometry(0.016, 12);
  const leftSensorDisc = new THREE.Mesh(sensorDiscGeo, thermalPitMaterial);
  leftSensorDisc.position.set(headRadius * 0.44 + 0.005, headRadius * 0.10, 0.78);
  leftSensorDisc.rotation.set(0, Math.PI * 0.45, 0);
  thermalPitsGroup.add(leftSensorDisc);

  const rightSensorDisc = new THREE.Mesh(sensorDiscGeo, thermalPitMaterial);
  rightSensorDisc.position.set(-headRadius * 0.44 - 0.005, headRadius * 0.10, 0.78);
  rightSensorDisc.rotation.set(0, -Math.PI * 0.45, 0);
  thermalPitsGroup.add(rightSensorDisc);

  craniumMesh.add(thermalPitsGroup);

  // 3h. Articulated Lower Mandible (Hinged Lower Jaw & Elastic Throat Skin - 28 segments)
  const jawHingeGroup = new THREE.Group();
  jawHingeGroup.name = 'articulated_mandible_hinge';
  jawHingeGroup.position.set(0, -headRadius * 0.14, 0.10);

  // Main Mandibular Body (Split dental arc with smoothly rounded chin apex)
  const jawGeo = new THREE.CylinderGeometry(
    skullWidth * 0.30, // Chin width
    skullWidth * 0.88, // Posterior jaw angle width
    skullLength * 0.90,
    28,
    8,
    false
  );
  jawGeo.rotateX(Math.PI / 2);
  jawGeo.scale(1.0, 0.34, 1.0);
  jawGeo.translate(0, -headRadius * 0.12, skullLength * 0.38);

  const jawMesh = new THREE.Mesh(jawGeo, headVentralMaterial);
  jawMesh.castShadow = true;
  jawHingeGroup.add(jawMesh);

  // Mandibular Mental Groove (Longitudinal median elasticity groove along chin underside)
  const grooveGeo = new THREE.CylinderGeometry(0.014, 0.018, 0.65, 12);
  grooveGeo.rotateX(Math.PI / 2);
  const mentalGroove = new THREE.Mesh(grooveGeo, headDorsalMaterial);
  mentalGroove.position.set(0, -headRadius * 0.26, skullLength * 0.46);
  jawMesh.add(mentalGroove);

  // Infralabial Lower Lip Scale Rims (Smooth 24-segment rims)
  const leftInfraLabial = new THREE.Mesh(labialGeo.clone().scale(0.9, 0.9, 0.9), headVentralMaterial);
  leftInfraLabial.position.set(headRadius * 0.50, 0.02, 0.54);
  leftInfraLabial.rotation.set(0, 0.18, 0);
  jawMesh.add(leftInfraLabial);

  const rightInfraLabial = new THREE.Mesh(labialGeo.clone().scale(0.9, 0.9, 0.9), headVentralMaterial);
  rightInfraLabial.position.set(-headRadius * 0.50, 0.02, 0.54);
  rightInfraLabial.rotation.set(0, -0.18, 0);
  jawMesh.add(rightInfraLabial);

  // Mouth Interior / Mouth Floor Epithelium (Smooth curved half-cylinder)
  const mouthFloorGeo = new THREE.CylinderGeometry(
    headRadius * 0.30,
    headRadius * 0.40,
    0.75,
    24,
    2,
    false,
    0,
    Math.PI
  );
  mouthFloorGeo.rotateZ(Math.PI / 2);
  mouthFloorGeo.scale(0.85, 0.10, 1.0);
  mouthFloorGeo.translate(0, 0.01, 0.42);
  const mouthFloor = new THREE.Mesh(mouthFloorGeo, mouthLiningMaterial);
  jawMesh.add(mouthFloor);

  // Glottis Tube (Breathing tube on floor of snake mouth with 16 segments)
  const glottisGeo = new THREE.CylinderGeometry(0.025, 0.03, 0.14, 16);
  glottisGeo.rotateX(Math.PI / 2);
  const glottis = new THREE.Mesh(glottisGeo, mouthLiningMaterial);
  glottis.position.set(0, 0.04, 0.38);
  jawMesh.add(glottis);

  // Lower Mandibular Recurved Micro-Teeth (Smooth 12-segment recurved dental rows)
  for (let t = 0; t < 6; t++) {
    const microToothGeo = new THREE.ConeGeometry(0.012, 0.055, 12);
    microToothGeo.rotateX(-0.35); // Backward recurved

    const lt = new THREE.Mesh(microToothGeo, fangMaterial);
    lt.position.set(0.12 + (t % 3) * 0.05, 0.04, 0.28 + t * 0.10);
    jawMesh.add(lt);

    const rt = new THREE.Mesh(microToothGeo, fangMaterial);
    rt.position.set(-0.12 - (t % 3) * 0.05, 0.04, 0.28 + t * 0.10);
    jawMesh.add(rt);
  }

  // Elastic Gular Throat Skin (Curved organic throat envelope - No sharp corners)
  const throatGeo = new THREE.CylinderGeometry(
    headRadius * 0.42,
    headRadius * 0.46,
    0.38,
    24,
    4,
    false,
    Math.PI,
    Math.PI
  );
  throatGeo.rotateZ(Math.PI / 2);
  throatGeo.scale(0.92, 0.38, 1.0);
  throatGeo.translate(0, -headRadius * 0.15, -0.06);
  const throatMesh = new THREE.Mesh(throatGeo, headVentralMaterial);
  jawMesh.add(throatMesh);

  headGroup.add(jawHingeGroup);

  // 3i. Retractable Maxillary Needling Fangs with Translucent Tissue Sheaths
  const fangCurveGeo = new THREE.ConeGeometry(0.032, 0.32, 20);
  fangCurveGeo.rotateX(-0.45); // Natural backward curved resting angle
  fangCurveGeo.scale(1.0, 1.0, 0.85);

  const leftFang = new THREE.Mesh(fangCurveGeo, fangMaterial);
  leftFang.position.set(0.18, -headRadius * 0.06, 0.72);
  headGroup.add(leftFang);

  const rightFang = new THREE.Mesh(fangCurveGeo, fangMaterial);
  rightFang.position.set(-0.18, -headRadius * 0.06, 0.72);
  headGroup.add(rightFang);

  // Translucent Fang Sheaths at the gumline
  const sheathGeo = new THREE.TorusGeometry(0.036, 0.012, 10, 16);
  const leftSheath = new THREE.Mesh(sheathGeo, mouthLiningMaterial);
  leftSheath.position.set(0.18, -headRadius * 0.04, 0.72);
  leftSheath.rotation.x = Math.PI / 2;
  headGroup.add(leftSheath);

  const rightSheath = new THREE.Mesh(sheathGeo, mouthLiningMaterial);
  rightSheath.position.set(-0.18, -headRadius * 0.04, 0.72);
  rightSheath.rotation.x = Math.PI / 2;
  headGroup.add(rightSheath);

  // Palatal Micro-Teeth Rows (Roof of upper maxilla with 12 segments)
  for (let pt = 0; pt < 4; pt++) {
    const pToothGeo = new THREE.ConeGeometry(0.012, 0.06, 12);
    pToothGeo.rotateX(-0.30);

    const lpt = new THREE.Mesh(pToothGeo, fangMaterial);
    lpt.position.set(0.08, -headRadius * 0.04, 0.44 + pt * 0.09);
    headGroup.add(lpt);

    const rpt = new THREE.Mesh(pToothGeo, fangMaterial);
    rpt.position.set(-0.08, -headRadius * 0.04, 0.44 + pt * 0.09);
    headGroup.add(rpt);
  }

  // 3j. Glowing Reactive Reptilian Spherical Eyes with Clear Spectacle (Brille) & Smooth Slit Pupils (24x20)
  const eyeRadius = 0.098;
  const eyeGlobeGeo = new THREE.SphereGeometry(eyeRadius, 24, 20);
  const pupilGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.13, 16);
  pupilGeo.scale(0.35, 1.0, 1.0);

  // Clear Spectacle Cornea Scale (Outer transparent high-gloss glass lens)
  const spectacleMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.04,
    metalness: 0.0,
    transparent: true,
    opacity: 0.35,
  });
  const spectacleGeo = new THREE.SphereGeometry(eyeRadius * 1.08, 24, 20);
  const coronaGeo = new THREE.RingGeometry(0.016, 0.054, 24);

  // Left Eye Orbit Assembly
  const leftEyeGroup = new THREE.Group();
  leftEyeGroup.name = 'left_reptile_eye_orbit';
  leftEyeGroup.position.set(headRadius * 0.74, headRadius * 0.18, 0.52);

  const leftEyeGlobe = new THREE.Mesh(eyeGlobeGeo, eyeIrisMaterial);
  leftEyeGlobe.rotation.y = Math.PI / 2;
  leftEyeGroup.add(leftEyeGlobe);

  const leftSpectacle = new THREE.Mesh(spectacleGeo, spectacleMat);
  leftEyeGroup.add(leftSpectacle);

  const leftPupil = new THREE.Mesh(pupilGeo, pupilMaterial);
  leftPupil.position.set(eyeRadius * 0.88, 0, 0);
  leftEyeGroup.add(leftPupil);

  const leftCorona = new THREE.Mesh(coronaGeo, eyeCoronaMaterial);
  leftCorona.rotation.y = Math.PI / 2;
  leftCorona.position.set(eyeRadius * 0.90, 0, 0);
  leftEyeGroup.add(leftCorona);

  const leftEyeLight = new THREE.PointLight(0xf59e0b, 1.4, 4.0);
  leftEyeLight.position.set(eyeRadius * 1.15, 0, 0);
  leftEyeGroup.add(leftEyeLight);

  headGroup.add(leftEyeGroup);

  // Right Eye Orbit Assembly
  const rightEyeGroup = new THREE.Group();
  rightEyeGroup.name = 'right_reptile_eye_orbit';
  rightEyeGroup.position.set(-headRadius * 0.74, headRadius * 0.18, 0.52);

  const rightEyeGlobe = new THREE.Mesh(eyeGlobeGeo, eyeIrisMaterial);
  rightEyeGlobe.rotation.y = -Math.PI / 2;
  rightEyeGroup.add(rightEyeGlobe);

  const rightSpectacle = new THREE.Mesh(spectacleGeo, spectacleMat);
  rightEyeGroup.add(rightSpectacle);

  const rightPupil = new THREE.Mesh(pupilGeo, pupilMaterial);
  rightPupil.position.set(-eyeRadius * 0.88, 0, 0);
  rightEyeGroup.add(rightPupil);

  const rightCorona = new THREE.Mesh(coronaGeo, eyeCoronaMaterial);
  rightCorona.rotation.y = -Math.PI / 2;
  rightCorona.position.set(-eyeRadius * 0.90, 0, 0);
  rightEyeGroup.add(rightCorona);

  const rightEyeLight = new THREE.PointLight(0xf59e0b, 1.4, 4.0);
  rightEyeLight.position.set(-eyeRadius * 1.15, 0, 0);
  rightEyeGroup.add(rightEyeLight);

  headGroup.add(rightEyeGroup);

  // 3k. Chemosensory Bifurcated Tongue (Exits smoothly through the sub-rostral notch - 16 segments)
  const tongueGroup = new THREE.Group();
  tongueGroup.name = 'chemosensory_forked_tongue';
  tongueGroup.position.set(0, -headRadius * 0.08, 0.78);

  const tongueShaftGeo = new THREE.CylinderGeometry(0.016, 0.018, 0.38, 16);
  tongueShaftGeo.rotateX(Math.PI / 2);
  tongueShaftGeo.translate(0, 0, 0.19);
  const tongueShaft = new THREE.Mesh(tongueShaftGeo, tongueMaterial);
  tongueGroup.add(tongueShaft);

  // Left Fork Tine
  const forkTineGeo = new THREE.ConeGeometry(0.012, 0.22, 12);
  forkTineGeo.rotateX(Math.PI / 2);
  forkTineGeo.rotateY(0.35);

  const leftFork = new THREE.Mesh(forkTineGeo, tongueMaterial);
  leftFork.position.set(0.032, 0, 0.44);
  tongueGroup.add(leftFork);

  // Right Fork Tine
  const rightFork = new THREE.Mesh(forkTineGeo.clone().rotateY(-0.70), tongueMaterial);
  rightFork.position.set(-0.032, 0, 0.44);
  tongueGroup.add(rightFork);

  headGroup.add(tongueGroup);

  // 3l. Custom Cephalic Adornments (Swept Cerastes Horns, Sensory Whiskers, Armored Crown Crest)
  // Horned Viper (Cerastes) Swept Horns (Supraocular keratin horns curving upward and rearward)
  const hornGeo = new THREE.ConeGeometry(0.038, 0.34, 16);
  hornGeo.rotateX(-0.55); // Swept back
  hornGeo.translate(0, 0.14, -0.04);

  const leftHorn = new THREE.Mesh(hornGeo, hornMaterial);
  leftHorn.position.set(headRadius * 0.62, headRadius * 0.36, 0.46);
  leftHorn.rotation.set(0.10, 0.25, -0.35);
  leftHorn.visible = false;
  craniumMesh.add(leftHorn);

  const rightHorn = new THREE.Mesh(hornGeo, hornMaterial);
  rightHorn.position.set(-headRadius * 0.62, headRadius * 0.36, 0.46);
  rightHorn.rotation.set(0.10, -0.25, 0.35);
  rightHorn.visible = false;
  craniumMesh.add(rightHorn);

  // Tactile Sensory Whisker Barbels (Sinuous sub-rostral sensory barbels)
  const tactileWhiskers = new THREE.Group();
  tactileWhiskers.name = 'tactile_sensory_whiskers';
  tactileWhiskers.visible = false;

  const whiskerGeo = new THREE.CylinderGeometry(0.008, 0.003, 0.42, 12);
  whiskerGeo.rotateZ(Math.PI / 2);
  whiskerGeo.translate(0.20, 0, 0);

  const leftWhisker = new THREE.Mesh(whiskerGeo, tongueMaterial);
  leftWhisker.position.set(0.08, -headRadius * 0.04, 0.95);
  leftWhisker.rotation.set(0.15, 0.55, -0.20);
  tactileWhiskers.add(leftWhisker);

  const rightWhisker = new THREE.Mesh(whiskerGeo.clone().scale(-1, 1, 1), tongueMaterial);
  rightWhisker.position.set(-0.08, -headRadius * 0.04, 0.95);
  rightWhisker.rotation.set(0.15, -0.55, 0.20);
  tactileWhiskers.add(rightWhisker);

  craniumMesh.add(tactileWhiskers);

  // Armored Dorsal Crown Crest (3-tiered raised scute crest along parietal-occipital axis)
  const crownCrestGroup = new THREE.Group();
  crownCrestGroup.name = 'armored_crown_crest';
  crownCrestGroup.visible = false;

  for (let c = 0; c < 3; c++) {
    const cSize = 0.26 - c * 0.05;
    const cGeo = new THREE.ConeGeometry(headRadius * cSize, 0.32, 4);
    cGeo.rotateY(Math.PI / 4);
    cGeo.rotateX(Math.PI / 2 - 0.25);
    cGeo.scale(0.85, 0.30, 1.2);

    const cMesh = new THREE.Mesh(cGeo, crownMaterial);
    cMesh.position.set(0, headRadius * 0.38 + (2 - c) * 0.04, 0.15 + c * 0.22);
    crownCrestGroup.add(cMesh);
  }
  craniumMesh.add(crownCrestGroup);

  root.add(headGroup);

  // ---------------------------------------------------------------------------
  // 4. Visual Debug Architecture (Rings, Sampled Spine, Frames)
  // ---------------------------------------------------------------------------
  const debugGroup = new THREE.Group();
  debugGroup.name = 'snake_debug_group';
  debugGroup.visible = false;

  const ringSegmentsPerRing = RADIAL_SEGMENTS;
  const totalRingLineVerts = NUM_RINGS * ringSegmentsPerRing * 2;
  const ringPositions = new Float32Array(totalRingLineVerts * 3);
  const ringsGeo = new THREE.BufferGeometry();
  ringsGeo.setAttribute('position', new THREE.BufferAttribute(ringPositions, 3));
  const ringsMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.85,
  });
  const ringsMesh = new THREE.LineSegments(ringsGeo, ringsMat);
  debugGroup.add(ringsMesh);

  const spinePointPositions = new Float32Array(NUM_RINGS * 3);
  const spinePointsGeo = new THREE.BufferGeometry();
  spinePointsGeo.setAttribute('position', new THREE.BufferAttribute(spinePointPositions, 3));
  const spinePointsMat = new THREE.PointsMaterial({
    color: 0xfacc15,
    size: 0.35,
    sizeAttenuation: true,
  });
  const spinePointsMesh = new THREE.Points(spinePointsGeo, spinePointsMat);
  debugGroup.add(spinePointsMesh);

  const framePositions = new Float32Array(NUM_RINGS * 6 * 3);
  const frameColors = new Float32Array(NUM_RINGS * 6 * 3);

  for (let r = 0; r < NUM_RINGS; r++) {
    const base = r * 18;
    // Tangent (Blue)
    frameColors[base + 0] = 0.2; frameColors[base + 1] = 0.6; frameColors[base + 2] = 1.0;
    frameColors[base + 3] = 0.2; frameColors[base + 4] = 0.6; frameColors[base + 5] = 1.0;
    // Normal / Up (Green)
    frameColors[base + 6] = 0.2; frameColors[base + 7] = 1.0; frameColors[base + 8] = 0.3;
    frameColors[base + 9] = 0.2; frameColors[base + 10] = 1.0; frameColors[base + 11] = 0.3;
    // Binormal / Right (Red)
    frameColors[base + 12] = 1.0; frameColors[base + 13] = 0.2; frameColors[base + 14] = 0.3;
    frameColors[base + 15] = 1.0; frameColors[base + 16] = 0.2; frameColors[base + 17] = 0.3;
  }

  const framesGeo = new THREE.BufferGeometry();
  framesGeo.setAttribute('position', new THREE.BufferAttribute(framePositions, 3));
  framesGeo.setAttribute('color', new THREE.BufferAttribute(frameColors, 3));
  const framesMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
  });
  const framesMesh = new THREE.LineSegments(framesGeo, framesMat);
  debugGroup.add(framesMesh);

  root.add(debugGroup);

  // ---------------------------------------------------------------------------
  // 5. Pre-allocated Zero-GC Runtime Spline Buffers
  // ---------------------------------------------------------------------------
  const spineCenters = new Float32Array(NUM_RINGS * 3);
  const spineTangents = new Float32Array(NUM_RINGS * 3);
  const spineNormals = new Float32Array(NUM_RINGS * 3);
  const spineBinormals = new Float32Array(NUM_RINGS * 3);
  const arcLengths = new Float32Array(1024);

  return {
    root,
    headGroup,
    craniumMesh,
    jawHingeGroup,
    jawMesh,
    leftEyeGroup,
    rightEyeGroup,
    leftEyeGlobe,
    rightEyeGlobe,
    leftPupil,
    rightPupil,
    leftEyeCorona: leftCorona,
    rightEyeCorona: rightCorona,
    leftEyeLight,
    rightEyeLight,
    leftFang,
    rightFang,
    leftHorn,
    rightHorn,
    tactileWhiskers,
    crownCrestMesh: crownCrestGroup,
    thermalPits: thermalPitsGroup,
    tongueGroup,
    tongueMesh: tongueShaft,
    bodyMesh,
    bodyGeometry,
    bodyMaterial,
    bodyTexture,
    headTexture,
    headBumpTexture,
    headRoughnessTexture,
    debug: {
      group: debugGroup,
      enabled: false,
      ringsMesh,
      spinePointsMesh,
      framesMesh,
      buffers: {
        ringPositions,
        spinePointPositions,
        framePositions,
        frameColors,
      },
    },
    materials: {
      head: headMaterial,
      headDorsal: headDorsalMaterial,
      headVentral: headVentralMaterial,
      body: bodyMaterial,
      eye: eyeIrisMaterial,
      eyeCorona: eyeCoronaMaterial,
      pupil: pupilMaterial,
      fang: fangMaterial,
      tongue: tongueMaterial,
      horn: hornMaterial,
      crown: crownMaterial,
      thermalPit: thermalPitMaterial,
    },
    buffers: {
      positions,
      normals,
      spineCenters,
      spineTangents,
      spineNormals,
      spineBinormals,
      arcLengths,
    },
    telemetry: {
      totalArcLength: 0,
      headScale: 1.0,
      bodyThickness: 1.0,
      growthFactor: 1.0,
      jawAngleDeg: 0,
      eyeGazeAngle: 0,
      crownCharge: 0,
    },
  };
}

/**
 * Toggles the interactive visual debug display of cross-section rings,
 * sampled spine centerline points, and coordinate frames.
 */
export function setSnakeDebugMode(rig: SnakeRig, enabled: boolean) {
  if (!rig || !rig.debug) return;
  rig.debug.enabled = enabled;
  rig.debug.group.visible = enabled;
}

/**
 * Updates the realistic snake mesh, anatomical predator head, and kinematics:
 * - 3D yaw, pitch, and organic serpentine micro-sway
 * - Realistic articulated mandible opening during strikes and lunges
 * - Retractable needle fangs erection during bites
 * - 3D ocular saccades & pupil tracking toward nearest targets
 * - Dynamic bifurcated tongue flicking through the sub-rostral notch
 */
export function updateSnakeRig(
  rig: SnakeRig,
  headPos: Vector3D | undefined,
  facingAngle: number = 0,
  pitch: number = 0,
  segments: SnakeSegment[] = [],
  isDashing: boolean = false,
  damageFlashTimer: number = 0,
  growthPulseTimer: number = 0,
  time: number = 0,
  growthFactor: number = 1.0,
  headState: SnakeHeadState = 'idle_cruise',
  jawAngle: number = 0,
  kineticCrownCharge: number = 0,
  eyeTargetPos: Vector3D | null = null,
  headConfig?: {
    jawStyle?: string;
    hornStyle?: string;
    eyeTracking?: boolean;
  }
) {
  if (!rig || !headPos) return;

  const hx = typeof headPos.x === 'number' ? headPos.x : 0;
  const hy = typeof headPos.y === 'number' ? headPos.y : 0.55;
  const hz = typeof headPos.z === 'number' ? headPos.z : 0;

  // 1. Natural Growth Scaling
  const safeGrowth = Math.max(1.0, growthFactor);
  const thicknessGrowth = Math.pow(safeGrowth, 0.18);
  const headScaleGrowth = Math.pow(safeGrowth, 0.14);

  rig.telemetry.growthFactor = safeGrowth;
  rig.telemetry.bodyThickness = thicknessGrowth;
  rig.telemetry.headScale = headScaleGrowth;

  // 2. Position and Orient Head with 3D Yaw, Pitch & Serpentine Micro-Sway
  rig.headGroup.position.set(hx, hy, hz);
  rig.headGroup.scale.set(headScaleGrowth, headScaleGrowth, headScaleGrowth);

  const headSway = Math.sin(time * 3.2) * 0.012;
  const headRoll = Math.sin(time * 2.4) * 0.015;
  rig.headGroup.rotation.order = 'YXZ';
  rig.headGroup.rotation.set(-pitch, facingAngle + headSway, headRoll, 'YXZ');
  rig.headGroup.updateMatrixWorld(true);

  // ---------------------------------------------------------------------------
  // 3. Articulated Mandible Mechanics (Lower Jaw)
  // ---------------------------------------------------------------------------
  let targetJawRot = Math.max(0, Math.min(0.75, jawAngle));

  // State-driven jaw posture
  if (headState === 'strike_lunge' || isDashing) {
    targetJawRot = Math.max(targetJawRot, 0.52); // Wide strike lunge
  } else if (headState === 'kinetic_roar') {
    targetJawRot = Math.max(targetJawRot, 0.72); // Roar / strike wide
  } else if (headState === 'hunting_track') {
    targetJawRot = Math.max(targetJawRot, 0.08); // Slight opening during stalking
  } else if (headState === 'damage_recoil') {
    targetJawRot = Math.max(targetJawRot, 0.35);
  }

  // Smooth jaw interpolation
  const currentJawRot = rig.jawHingeGroup.rotation.x;
  rig.jawHingeGroup.rotation.x = THREE.MathUtils.lerp(currentJawRot, targetJawRot, 0.25);
  rig.telemetry.jawAngleDeg = (rig.jawHingeGroup.rotation.x * 180) / Math.PI;

  // ---------------------------------------------------------------------------
  // 4. Retractable Fangs Erection Mechanics
  // ---------------------------------------------------------------------------
  const fangErect = Math.min(1.0, rig.jawHingeGroup.rotation.x / 0.40);
  const fangBaseAngle = -0.45; // Resting tucked position
  const fangErectAngle = 0.38; // Fully extended striking position
  const activeFangRot = THREE.MathUtils.lerp(fangBaseAngle, fangErectAngle, fangErect);

  rig.leftFang.rotation.x = activeFangRot;
  rig.rightFang.rotation.x = activeFangRot;

  // ---------------------------------------------------------------------------
  // 5. 3D Glowing Reactive Reptilian Eye Gaze Tracking & Vertical Slit Pupils
  // ---------------------------------------------------------------------------
  const enableTracking = headConfig?.eyeTracking !== false;
  let distToPrey = 999.0;
  let isTrackingBoid = false;

  if (enableTracking && eyeTargetPos && typeof eyeTargetPos.x === 'number') {
    vGazeTarget.set(eyeTargetPos.x, eyeTargetPos.y, eyeTargetPos.z);
    vHeadWorld.set(hx, hy, hz);
    vTargetDir.subVectors(vGazeTarget, vHeadWorld);
    distToPrey = vTargetDir.length();

    if (distToPrey > 0.05 && distToPrey < 48.0) {
      isTrackingBoid = true;

      // Transform target position to head local coordinate space
      mHeadWorldInv.copy(rig.headGroup.matrixWorld).invert();
      vLocalTarget.copy(vGazeTarget).applyMatrix4(mHeadWorldInv);

      vLeftEyePosLocal.copy(rig.leftEyeGroup.position);
      vRightEyePosLocal.copy(rig.rightEyeGroup.position);

      // Left eye gaze direction vector & stereoscopic convergence
      vLeftDir.subVectors(vLocalTarget, vLeftEyePosLocal);
      vLeftDir.normalize();

      // Right eye gaze direction vector & stereoscopic convergence
      vRightDir.subVectors(vLocalTarget, vRightEyePosLocal);
      vRightDir.normalize();

      const leftTargetYaw = -Math.atan2(vLeftDir.z, vLeftDir.x);
      const leftTargetPitch = Math.asin(Math.max(-1, Math.min(1, vLeftDir.y)));
      const leftYawClamped = Math.max(-1.50, Math.min(0.25, leftTargetYaw));
      const leftPitchClamped = Math.max(-0.50, Math.min(0.50, leftTargetPitch));

      const rightTargetYaw = Math.atan2(vRightDir.z, -vRightDir.x);
      const rightTargetPitch = Math.asin(Math.max(-1, Math.min(1, vRightDir.y)));
      const rightYawClamped = Math.max(-0.25, Math.min(1.50, rightTargetYaw));
      const rightPitchClamped = Math.max(-0.50, Math.min(0.50, rightTargetPitch));

      // Biological micro-saccadic tremor when locking onto agile prey
      const saccadeYaw = Math.sin(time * 16.0) * 0.015;
      const saccadePitch = Math.cos(time * 14.0) * 0.012;

      rig.leftEyeGroup.rotation.y = THREE.MathUtils.lerp(rig.leftEyeGroup.rotation.y, leftYawClamped + saccadeYaw, 0.25);
      rig.leftEyeGroup.rotation.z = THREE.MathUtils.lerp(rig.leftEyeGroup.rotation.z, leftPitchClamped + saccadePitch, 0.25);

      rig.rightEyeGroup.rotation.y = THREE.MathUtils.lerp(rig.rightEyeGroup.rotation.y, rightYawClamped + saccadeYaw, 0.25);
      rig.rightEyeGroup.rotation.z = THREE.MathUtils.lerp(rig.rightEyeGroup.rotation.z, -rightPitchClamped - saccadePitch, 0.25);

      rig.telemetry.eyeGazeAngle = ((leftYawClamped + rightYawClamped) * 0.5 * 180) / Math.PI;

      // Reactive vertical slit pupil contraction: narrows to a thin razor slit as boid approaches
      const pupilSlitWidth = Math.max(0.12, Math.min(1.0, distToPrey / 16.0));
      const pupilHeight = Math.max(1.0, 1.20 - pupilSlitWidth * 0.20);
      rig.leftPupil.scale.set(pupilSlitWidth, pupilHeight, 1.0);
      rig.rightPupil.scale.set(pupilSlitWidth, pupilHeight, 1.0);
    }
  }

  if (!isTrackingBoid) {
    // Resting natural forward-lateral predator eye alignment
    const restSway = Math.sin(time * 1.5) * 0.04;
    rig.leftEyeGroup.rotation.set(0, -0.65 + restSway, 0);
    rig.rightEyeGroup.rotation.set(0, 0.65 + restSway, 0);
    rig.leftPupil.scale.set(1.0, 1.0, 1.0);
    rig.rightPupil.scale.set(1.0, 1.0, 1.0);
  }

  // ---------------------------------------------------------------------------
  // 5b. Dynamic Reactive Glowing Eyes Emission & Corona Flare Modulation
  // ---------------------------------------------------------------------------
  let targetEmissiveColor = cPredatorAmber;
  let targetEmissiveIntensity = 1.3;
  let targetLightIntensity = 1.2;
  let targetLightDistance = 3.5;
  let targetCoronaOpacity = 0.70;
  let targetCoronaScale = 1.0;

  if (headState === 'strike_lunge' || isDashing) {
    // Blazing crimson-amber plasma flare during strike / lunge
    targetEmissiveColor = cStrikeCrimson;
    targetEmissiveIntensity = 3.4 + Math.sin(time * 20.0) * 0.4;
    targetLightIntensity = 3.2;
    targetLightDistance = 6.0;
    targetCoronaOpacity = 0.96;
    targetCoronaScale = 1.25;
  } else if (headState === 'damage_recoil' || (damageFlashTimer && damageFlashTimer > 0)) {
    // Defensive electric cyan flare during damage
    targetEmissiveColor = cDamageCyan;
    targetEmissiveIntensity = 3.0 + Math.sin(time * 32.0) * 1.4;
    targetLightIntensity = 2.8;
    targetLightDistance = 5.0;
    targetCoronaOpacity = 0.90;
    targetCoronaScale = 1.20;
  } else if (isTrackingBoid || headState === 'hunting_track') {
    // Intense pulsating electric golden-amber focus when tracking nearest boid
    const huntingPulse = Math.sin(time * 8.0) * 0.5;
    targetEmissiveColor = cPredatorAmber;
    targetEmissiveIntensity = 2.2 + huntingPulse;
    targetLightIntensity = 2.2 + huntingPulse * 0.4;
    targetLightDistance = 4.8;
    targetCoronaOpacity = 0.88;
    targetCoronaScale = 1.10;
  } else {
    // Steady, gentle breathing predator glow
    const cruiseGlow = Math.sin(time * 2.2) * 0.25;
    targetEmissiveColor = cPredatorAmber;
    targetEmissiveIntensity = 1.3 + cruiseGlow;
    targetLightIntensity = 1.2 + cruiseGlow * 0.2;
    targetLightDistance = 3.5;
    targetCoronaOpacity = 0.72;
    targetCoronaScale = 1.0;
  }

  // Smoothly update iris emissive color and intensity
  if (rig.materials.eye) {
    cEyeLerp.copy(rig.materials.eye.emissive).lerp(targetEmissiveColor, 0.25);
    rig.materials.eye.emissive.copy(cEyeLerp);
    rig.materials.eye.emissiveIntensity = THREE.MathUtils.lerp(rig.materials.eye.emissiveIntensity, targetEmissiveIntensity, 0.25);
  }

  // Update corona lens flare materials and scale
  if (rig.materials.eyeCorona) {
    rig.materials.eyeCorona.color.copy(cEyeLerp);
    rig.materials.eyeCorona.opacity = THREE.MathUtils.lerp(rig.materials.eyeCorona.opacity, targetCoronaOpacity, 0.25);
  }
  if (rig.leftEyeCorona && rig.rightEyeCorona) {
    rig.leftEyeCorona.scale.set(targetCoronaScale, targetCoronaScale, targetCoronaScale);
    rig.rightEyeCorona.scale.set(targetCoronaScale, targetCoronaScale, targetCoronaScale);
  }

  // Update eye point lights
  if (rig.leftEyeLight && rig.rightEyeLight) {
    rig.leftEyeLight.color.copy(cEyeLerp);
    rig.leftEyeLight.intensity = THREE.MathUtils.lerp(rig.leftEyeLight.intensity, targetLightIntensity, 0.25);
    rig.leftEyeLight.distance = targetLightDistance;

    rig.rightEyeLight.color.copy(cEyeLerp);
    rig.rightEyeLight.intensity = THREE.MathUtils.lerp(rig.rightEyeLight.intensity, targetLightIntensity, 0.25);
    rig.rightEyeLight.distance = targetLightDistance;
  }

  // ---------------------------------------------------------------------------
  // 6. Dynamic Chemosensory Forked Tongue Flickering (Through Sub-Rostral Notch)
  // ---------------------------------------------------------------------------
  const isHunting = headState === 'hunting_track' || headState === 'strike_lunge';
  const tongueCycle = Math.sin(time * (isHunting ? 4.0 : 2.0));
  const isFlicking = tongueCycle > (isHunting ? -0.1 : 0.45);
  const flickOscillation = Math.sin(time * (isHunting ? 24.0 : 16.0));

  const tongueExtension = isFlicking ? 1.0 + flickOscillation * 0.35 : 0.05;
  rig.tongueGroup.scale.set(1.0, 1.0, tongueExtension);
  rig.tongueGroup.rotation.x = isFlicking ? flickOscillation * 0.18 : 0;
  rig.tongueGroup.rotation.y = isHunting && eyeTargetPos ? Math.sin(time * 6.0) * 0.14 : 0;

  // ---------------------------------------------------------------------------
  // 6b. Dynamic Head Adornments & Sensory Organ Kinetics
  // ---------------------------------------------------------------------------
  const hornStyle = headConfig?.hornStyle || 'swept_horns';
  const showHorns = hornStyle === 'swept_horns';
  const showWhiskers = hornStyle === 'tactile_whiskers';
  const showCrown = hornStyle === 'crown_crest';

  if (rig.leftHorn && rig.rightHorn) {
    rig.leftHorn.visible = showHorns;
    rig.rightHorn.visible = showHorns;
    if (showHorns) {
      const hornFlex = Math.sin(time * 3.5) * 0.03;
      rig.leftHorn.rotation.x = 0.10 + hornFlex;
      rig.rightHorn.rotation.x = 0.10 + hornFlex;
    }
  }

  if (rig.tactileWhiskers) {
    rig.tactileWhiskers.visible = showWhiskers;
    if (showWhiskers) {
      const whiskerWobble = Math.sin(time * 8.0) * 0.12;
      rig.tactileWhiskers.rotation.y = whiskerWobble;
      rig.tactileWhiskers.rotation.x = Math.cos(time * 6.0) * 0.08;
    }
  }

  if (rig.crownCrestMesh) {
    rig.crownCrestMesh.visible = showCrown;
  }

  // Thermal Pit Sensor Emissive Pulsing during hunting or tracking
  if (rig.materials.thermalPit) {
    const pitPulse = isHunting ? 0.65 + Math.sin(time * 8.0) * 0.25 : 0.40 + Math.sin(time * 2.0) * 0.10;
    rig.materials.thermalPit.emissiveIntensity = pitPulse;
  }

  // Crown Crest Kinetic Energy Glow
  if (rig.materials.crown) {
    const crownGlow = Math.min(1.0, kineticCrownCharge) * 0.85;
    rig.materials.crown.emissiveIntensity = crownGlow;
  }

  rig.telemetry.crownCharge = kineticCrownCharge;

  // ---------------------------------------------------------------------------
  // 7. Build Continuous Kinematic Spine & Sample Arc Lengths
  // ---------------------------------------------------------------------------
  const segs = Array.isArray(segments) ? segments : [];
  const rawCount = segs.length + 1;

  const { positions, normals, spineCenters, spineTangents, spineNormals, spineBinormals, arcLengths } = rig.buffers;
  const maxRaw = Math.min(rawCount, arcLengths.length - 1);

  arcLengths[0] = 0;
  let totalLength = 0;
  vPrevPos.set(hx, hy, hz);

  for (let j = 0; j < maxRaw - 1; j++) {
    const s = segs[j];
    const sx = s && typeof s.x === 'number' ? s.x : vPrevPos.x;
    const sy = s && typeof s.y === 'number' ? s.y : vPrevPos.y;
    const sz = s && typeof s.z === 'number' ? s.z : vPrevPos.z;
    vNextPos.set(sx, sy, sz);

    const dist = vPrevPos.distanceTo(vNextPos);
    totalLength += dist;
    arcLengths[j + 1] = totalLength;
    vPrevPos.copy(vNextPos);
  }

  rig.telemetry.totalArcLength = totalLength;

  if (totalLength < 0.01) {
    totalLength = 2.0;
    for (let j = 1; j < maxRaw; j++) {
      arcLengths[j] = (j / (maxRaw - 1)) * totalLength;
    }
  }

  const getRawPoint = (idx: number, out: THREE.Vector3) => {
    const clamped = Math.max(0, Math.min(maxRaw - 1, idx));
    if (clamped === 0) {
      out.set(hx, hy, hz);
    } else {
      const s = segs[clamped - 1];
      if (s) {
        out.set(s.x, s.y, s.z);
      } else {
        out.set(hx, hy, hz);
      }
    }
  };

  // ---------------------------------------------------------------------------
  // 8. Resample NUM_RINGS Visual Spine Centers using Catmull-Rom Smoothing
  // ---------------------------------------------------------------------------
  let currentSegIdx = 0;

  for (let r = 0; r < NUM_RINGS; r++) {
    const targetDist = (r / (NUM_RINGS - 1)) * totalLength;

    while (currentSegIdx < maxRaw - 2 && arcLengths[currentSegIdx + 1] < targetDist) {
      currentSegIdx++;
    }

    const d0 = arcLengths[currentSegIdx];
    const d1 = arcLengths[currentSegIdx + 1] || d0 + 0.001;
    const segSpan = Math.max(0.0001, d1 - d0);
    const alpha = Math.max(0, Math.min(1, (targetDist - d0) / segSpan));

    getRawPoint(currentSegIdx - 1, vP0);
    getRawPoint(currentSegIdx, vP1);
    getRawPoint(currentSegIdx + 1, vP2);
    getRawPoint(currentSegIdx + 2, vP3);

    const u = alpha;
    const u2 = u * u;
    const u3 = u2 * u;

    const cx = 0.5 * (
      (2 * vP1.x) +
      (-vP0.x + vP2.x) * u +
      (2 * vP0.x - 5 * vP1.x + 4 * vP2.x - vP3.x) * u2 +
      (-vP0.x + 3 * vP1.x - 3 * vP2.x + vP3.x) * u3
    );

    const cy = 0.5 * (
      (2 * vP1.y) +
      (-vP0.y + vP2.y) * u +
      (2 * vP0.y - 5 * vP1.y + 4 * vP2.y - vP3.y) * u2 +
      (-vP0.y + 3 * vP1.y - 3 * vP2.y + vP3.y) * u3
    );

    const cz = 0.5 * (
      (2 * vP1.z) +
      (-vP0.z + vP2.z) * u +
      (2 * vP0.z - 5 * vP1.z + 4 * vP2.z - vP3.z) * u2 +
      (-vP0.z + 3 * vP1.z - 3 * vP2.z + vP3.z) * u3
    );

    const r3 = r * 3;
    spineCenters[r3 + 0] = cx;
    spineCenters[r3 + 1] = cy;
    spineCenters[r3 + 2] = cz;
  }

  // ---------------------------------------------------------------------------
  // 9. Compute Tangents along Sampled Spine Centers (Anchored to Skull Exit)
  // ---------------------------------------------------------------------------
  for (let r = 0; r < NUM_RINGS; r++) {
    const r3 = r * 3;
    if (r === 0) {
      const headBackX = -Math.sin(facingAngle) * Math.cos(pitch);
      const headBackY = -Math.sin(pitch);
      const headBackZ = -Math.cos(facingAngle) * Math.cos(pitch);

      const splineDirX = spineCenters[3] - spineCenters[0];
      const splineDirY = spineCenters[4] - spineCenters[1];
      const splineDirZ = spineCenters[5] - spineCenters[2];

      vTangent.set(
        headBackX * 0.70 + splineDirX * 0.30,
        headBackY * 0.70 + splineDirY * 0.30,
        headBackZ * 0.70 + splineDirZ * 0.30
      );
    } else if (r === NUM_RINGS - 1) {
      const prev3 = (NUM_RINGS - 2) * 3;
      vTangent.set(
        spineCenters[r3 + 0] - spineCenters[prev3 + 0],
        spineCenters[r3 + 1] - spineCenters[prev3 + 1],
        spineCenters[r3 + 2] - spineCenters[prev3 + 2]
      );
    } else {
      const next3 = (r + 1) * 3;
      const prev3 = (r - 1) * 3;
      vTangent.set(
        spineCenters[next3 + 0] - spineCenters[prev3 + 0],
        spineCenters[next3 + 1] - spineCenters[prev3 + 1],
        spineCenters[next3 + 2] - spineCenters[prev3 + 2]
      );
    }

    if (vTangent.lengthSq() < 0.000001) {
      vTangent.set(0, 0, 1);
    } else {
      vTangent.normalize();
    }

    spineTangents[r3 + 0] = vTangent.x;
    spineTangents[r3 + 1] = vTangent.y;
    spineTangents[r3 + 2] = vTangent.z;
  }

  // ---------------------------------------------------------------------------
  // 10. Rotation-Minimizing Parallel-Transport Frames
  // ---------------------------------------------------------------------------
  vTangent.set(spineTangents[0], spineTangents[1], spineTangents[2]);

  vUp.set(
    -Math.sin(facingAngle) * Math.sin(pitch),
    Math.cos(pitch),
    -Math.cos(facingAngle) * Math.sin(pitch)
  );

  vUp.addScaledVector(vTangent, -vUp.dot(vTangent));
  if (vUp.lengthSq() < 0.001) {
    vUp.set(0, 1, 0).addScaledVector(vTangent, -vTangent.y);
  }
  vUp.normalize();
  vRight.crossVectors(vTangent, vUp).normalize();

  spineNormals[0] = vUp.x;
  spineNormals[1] = vUp.y;
  spineNormals[2] = vUp.z;

  spineBinormals[0] = vRight.x;
  spineBinormals[1] = vRight.y;
  spineBinormals[2] = vRight.z;

  for (let r = 1; r < NUM_RINGS; r++) {
    const prev3 = (r - 1) * 3;
    const cur3 = r * 3;

    vPrevTangent.set(spineTangents[prev3 + 0], spineTangents[prev3 + 1], spineTangents[prev3 + 2]);
    vTangent.set(spineTangents[cur3 + 0], spineTangents[cur3 + 1], spineTangents[cur3 + 2]);
    vPrevUp.set(spineNormals[prev3 + 0], spineNormals[prev3 + 1], spineNormals[prev3 + 2]);

    vAxis.crossVectors(vPrevTangent, vTangent);
    const axisLen = vAxis.length();

    if (axisLen > 0.0001) {
      vAxis.divideScalar(axisLen);
      const dotVal = Math.max(-1, Math.min(1, vPrevTangent.dot(vTangent)));
      const angle = Math.acos(dotVal);
      qRot.setFromAxisAngle(vAxis, angle);
      vRotatedUp.copy(vPrevUp).applyQuaternion(qRot);
    } else {
      vRotatedUp.copy(vPrevUp);
    }

    vRotatedUp.addScaledVector(vTangent, -vRotatedUp.dot(vTangent));
    if (vRotatedUp.lengthSq() < 0.0001) {
      vRotatedUp.set(0, 1, 0).addScaledVector(vTangent, -vTangent.y);
    }
    vRotatedUp.normalize();

    // Prevent accumulation of roll / inversion flips along the spine
    // Blend gently towards world +Y when pitch is shallow
    const worldUpProj = new THREE.Vector3(0, 1, 0).addScaledVector(vTangent, -vTangent.y);
    if (worldUpProj.lengthSq() > 0.1) {
      worldUpProj.normalize();
      if (vRotatedUp.dot(worldUpProj) < -0.2) {
        vRotatedUp.negate();
      } else {
        vRotatedUp.lerp(worldUpProj, 0.04).normalize();
      }
    }

    vRight.crossVectors(vTangent, vRotatedUp).normalize();

    spineNormals[cur3 + 0] = vRotatedUp.x;
    spineNormals[cur3 + 1] = vRotatedUp.y;
    spineNormals[cur3 + 2] = vRotatedUp.z;

    spineBinormals[cur3 + 0] = vRight.x;
    spineBinormals[cur3 + 1] = vRight.y;
    spineBinormals[cur3 + 2] = vRight.z;
  }

  // ---------------------------------------------------------------------------
  // 11. Generate 24-Sided Anatomical Vertices & Smooth Normals In-Place
  // ---------------------------------------------------------------------------
  const angleStep = (Math.PI * 2) / RADIAL_SEGMENTS;

  for (let r = 0; r < NUM_RINGS; r++) {
    const t = r / (NUM_RINGS - 1);
    const r3 = r * 3;

    const cx = spineCenters[r3 + 0];
    const cy = spineCenters[r3 + 1];
    const cz = spineCenters[r3 + 2];

    const upX = spineNormals[r3 + 0];
    const upY = spineNormals[r3 + 1];
    const upZ = spineNormals[r3 + 2];

    const rightX = spineBinormals[r3 + 0];
    const rightY = spineBinormals[r3 + 1];
    const rightZ = spineBinormals[r3 + 2];

    const baseTaperR = getCatmullRomRadiusAtT(t);
    const organicBreathing = 1.0 + Math.sin(t * 12.0 - time * 2.5) * 0.012;
    const currentRadius = baseTaperR * BASE_RADIUS * thicknessGrowth * organicBreathing;

    const rHoriz = currentRadius;
    const rVertDorsal = currentRadius * DORSAL_FLATTEN;
    const rVertVentral = currentRadius * VENTRAL_FLATTEN;

    const rowOffset = r * VERTS_PER_RING;

    for (let k = 0; k < VERTS_PER_RING; k++) {
      const angle = k * angleStep;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      // Latitudinal (right/left) and vertical (dorsal/ventral) factors:
      // k = 0, 24: angle = 0, 2PI -> latFactor = 0, vertFactor = -1.0 (Ventral Belly / Underneath)
      // k = 6: angle = PI/2 -> latFactor = +1.0, vertFactor = 0.0 (Right Flank)
      // k = 12: angle = PI -> latFactor = 0.0, vertFactor = +1.0 (Dorsal Spine / TOP OF SNAKE)
      // k = 18: angle = 3PI/2 -> latFactor = -1.0, vertFactor = 0.0 (Left Flank)
      const latFactor = sinA;
      const vertFactor = -cosA;

      // Subtle keeled dorsal ridge elevation along the top spine for realistic reptilian anatomy
      const dorsalKeelFactor = vertFactor > 0.70 ? 1.0 + (vertFactor - 0.70) * 0.12 : 1.0;
      const rVert = (vertFactor >= 0 ? rVertDorsal : rVertVentral) * dorsalKeelFactor;

      const dx = rightX * (latFactor * rHoriz) + upX * (vertFactor * rVert);
      const dy = rightY * (latFactor * rHoriz) + upY * (vertFactor * rVert);
      const dz = rightZ * (latFactor * rHoriz) + upZ * (vertFactor * rVert);

      const vertIdx = rowOffset + k;
      const v3 = vertIdx * 3;

      positions[v3 + 0] = cx + dx;
      positions[v3 + 1] = cy + dy;
      positions[v3 + 2] = cz + dz;

      const nx = rightX * (latFactor / rHoriz) + upX * (vertFactor / rVert);
      const ny = rightY * (latFactor / rHoriz) + upY * (vertFactor / rVert);
      const nz = rightZ * (latFactor / rHoriz) + upZ * (vertFactor / rVert);
      const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1.0;

      normals[v3 + 0] = nx / nLen;
      normals[v3 + 1] = ny / nLen;
      normals[v3 + 2] = nz / nLen;
    }
  }

  // Tail Tip Apex Vertex
  const tailTipIdx = NUM_RINGS * VERTS_PER_RING;
  const lastR3 = (NUM_RINGS - 1) * 3;
  const tailExtrude = 0.030;

  positions[tailTipIdx * 3 + 0] = spineCenters[lastR3 + 0] + spineTangents[lastR3 + 0] * tailExtrude;
  positions[tailTipIdx * 3 + 1] = spineCenters[lastR3 + 1] + spineTangents[lastR3 + 1] * tailExtrude;
  positions[tailTipIdx * 3 + 2] = spineCenters[lastR3 + 2] + spineTangents[lastR3 + 2] * tailExtrude;

  normals[tailTipIdx * 3 + 0] = spineTangents[lastR3 + 0];
  normals[tailTipIdx * 3 + 1] = spineTangents[lastR3 + 1];
  normals[tailTipIdx * 3 + 2] = spineTangents[lastR3 + 2];

  rig.bodyGeometry.attributes.position.needsUpdate = true;
  rig.bodyGeometry.attributes.normal.needsUpdate = true;

  // ---------------------------------------------------------------------------
  // 12. Update Debug Visual Buffers if Enabled
  // ---------------------------------------------------------------------------
  if (rig.debug.enabled) {
    const { ringPositions, spinePointPositions, framePositions } = rig.debug.buffers;

    for (let r = 0; r < NUM_RINGS; r++) {
      const r3 = r * 3;
      spinePointPositions[r3 + 0] = spineCenters[r3 + 0];
      spinePointPositions[r3 + 1] = spineCenters[r3 + 1];
      spinePointPositions[r3 + 2] = spineCenters[r3 + 2];
    }
    rig.debug.spinePointsMesh.geometry.attributes.position.needsUpdate = true;

    let ringLineIdx = 0;
    for (let r = 0; r < NUM_RINGS; r++) {
      const rowOffset = r * VERTS_PER_RING;
      for (let k = 0; k < RADIAL_SEGMENTS; k++) {
        const v0 = (rowOffset + k) * 3;
        const v1 = (rowOffset + k + 1) * 3;

        ringPositions[ringLineIdx++] = positions[v0 + 0];
        ringPositions[ringLineIdx++] = positions[v0 + 1];
        ringPositions[ringLineIdx++] = positions[v0 + 2];

        ringPositions[ringLineIdx++] = positions[v1 + 0];
        ringPositions[ringLineIdx++] = positions[v1 + 1];
        ringPositions[ringLineIdx++] = positions[v1 + 2];
      }
    }
    rig.debug.ringsMesh.geometry.attributes.position.needsUpdate = true;

    const frameScale = 0.45;
    let frameVertIdx = 0;

    for (let r = 0; r < NUM_RINGS; r++) {
      const r3 = r * 3;
      const cx = spineCenters[r3 + 0];
      const cy = spineCenters[r3 + 1];
      const cz = spineCenters[r3 + 2];

      const tx = spineTangents[r3 + 0];
      const ty = spineTangents[r3 + 1];
      const tz = spineTangents[r3 + 2];

      const nx = spineNormals[r3 + 0];
      const ny = spineNormals[r3 + 1];
      const nz = spineNormals[r3 + 2];

      const bx = spineBinormals[r3 + 0];
      const by = spineBinormals[r3 + 1];
      const bz = spineBinormals[r3 + 2];

      framePositions[frameVertIdx++] = cx;
      framePositions[frameVertIdx++] = cy;
      framePositions[frameVertIdx++] = cz;
      framePositions[frameVertIdx++] = cx + tx * frameScale;
      framePositions[frameVertIdx++] = cy + ty * frameScale;
      framePositions[frameVertIdx++] = cz + tz * frameScale;

      framePositions[frameVertIdx++] = cx;
      framePositions[frameVertIdx++] = cy;
      framePositions[frameVertIdx++] = cz;
      framePositions[frameVertIdx++] = cx + nx * frameScale;
      framePositions[frameVertIdx++] = cy + ny * frameScale;
      framePositions[frameVertIdx++] = cz + nz * frameScale;

      framePositions[frameVertIdx++] = cx;
      framePositions[frameVertIdx++] = cy;
      framePositions[frameVertIdx++] = cz;
      framePositions[frameVertIdx++] = cx + bx * frameScale;
      framePositions[frameVertIdx++] = cy + by * frameScale;
      framePositions[frameVertIdx++] = cz + bz * frameScale;
    }
    rig.debug.framesMesh.geometry.attributes.position.needsUpdate = true;
  }
}
