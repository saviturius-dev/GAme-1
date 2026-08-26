import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { UrsinaRepelSimulation } from '../simulation/simulationEngine';
import { SimulationConfig } from '../types';
import {
  createSnakeRig,
  updateSnakeRig,
  setSnakeDebugMode,
  TAPER_PROFILE,
  RADIAL_SEGMENTS,
  NUM_RINGS,
  SnakeRig,
} from '../rendering/snakeFactory';
import { SnakeTailTrailSystem } from '../rendering/snakeTailTrail';
import { createProceduralSkyDome } from '../rendering/skyShader';
import {
  Shield,
  Flame,
  Brain,
  User,
  Heart,
  Timer,
  Trophy,
  Zap,
  Activity,
  ChevronDown,
  ChevronUp,
  Sliders,
  Sparkles,
  Camera,
  Layers,
  Dna,
  Cpu,
  Compass,
  Navigation,
  Eye,
  EyeOff,
  HelpCircle,
  RotateCcw,
  Code2,
  X,
  Box,
} from 'lucide-react';

interface SimulationViewportProps {
  simulation: UrsinaRepelSimulation;
  config: SimulationConfig;
  isRunning: boolean;
  isSidebarOpen?: boolean;
  isZenMode?: boolean;
  onToggleSidebar?: () => void;
  onToggleZenMode?: () => void;
  onUpdateConfig: (updates: Partial<SimulationConfig>) => void;
  onManualRepel: () => void;
  onManualDash: () => void;
  onManualAscend?: () => void;
  onManualDescend?: () => void;
  onCarveTrench: () => void;
  onManualSpawn: () => void;
  onReset: () => void;
  onToggleControlMode: () => void;
}

export const SimulationViewport: React.FC<SimulationViewportProps> = ({
  simulation,
  config,
  isRunning,
  isSidebarOpen = true,
  isZenMode = false,
  onToggleSidebar,
  onToggleZenMode,
  onUpdateConfig,
  onManualRepel,
  onManualDash,
  onManualAscend,
  onManualDescend,
  onCarveTrench,
  onManualSpawn,
  onReset,
  onToggleControlMode,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  // Red Snake 3D Mesh Rig (Head + Eyes + Instanced Body Segments)
  const snakeRigRef = useRef<SnakeRig | null>(null);
  const repelCircleRef = useRef<THREE.Mesh | null>(null);

  // Swarm Particle Point Cloud References
  const swarmPointsRef = useRef<THREE.Points | null>(null);
  const swarmGeoRef = useRef<THREE.BufferGeometry | null>(null);

  // Decaying Creature Explosion Particle Point Cloud References
  const decayPointsRef = useRef<THREE.Points | null>(null);
  const decayGeoRef = useRef<THREE.BufferGeometry | null>(null);

  // Dynamic Snake Tail Trailing Particle System
  const tailTrailRef = useRef<SnakeTailTrailSystem | null>(null);

  // Terrain & Barrier Mesh References
  const cubeMeshesMapRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const cubeGroupRef = useRef<THREE.Group | null>(null);

  // Shockwave Rings
  const shockwaveGroupRef = useRef<THREE.Group | null>(null);

  const [cameraPreset, setCameraPreset] = useState<'overhead_follow_red' | 'free_orbit' | 'top_down'>('overhead_follow_red');
  const [isRLHudExpanded, setIsRLHudExpanded] = useState(true);
  const [showControlsGuide, setShowControlsGuide] = useState(false);
  const [isDebugMeshEnabled, setIsDebugMeshEnabled] = useState(false);
  const debugMeshRef = useRef(false);
  debugMeshRef.current = isDebugMeshEnabled;

  // 1. Procedural 64x64 Particle Glow Texture
  const getParticleGlowTexture = (): THREE.Texture => {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)'); // White core
    grad.addColorStop(0.2, 'rgba(254, 240, 138, 0.95)'); // Inner Ember
    grad.addColorStop(0.5, 'rgba(245, 158, 11, 0.70)'); // Mid Flame
    grad.addColorStop(0.8, 'rgba(239, 68, 68, 0.30)'); // Outer Halo
    grad.addColorStop(1.0, 'rgba(239, 68, 68, 0.0)'); // Edge feathering

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    return texture;
  };

  // 2. Procedural 64x64 Decaying Ash / Soot Radial Gradient Particle Texture
  const getDecayingAshTexture = (): THREE.Texture => {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d')!;

    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)'); // Solid core
    grad.addColorStop(0.35, 'rgba(240, 240, 245, 0.90)'); // Soft body
    grad.addColorStop(0.70, 'rgba(180, 180, 190, 0.45)'); // Feathery smoke edge
    grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)'); // Transparent edge

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    return texture;
  };

  // Initialize Three.js scene
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x7a8b99);
    scene.fog = new THREE.FogExp2(0x7a8b99, 0.008);
    sceneRef.current = scene;

    // Procedural Animated Scrolling Noise Cloud Sky Dome (#7A8B99 mist to #2E3440 storm)
    const skyDome = createProceduralSkyDome(450);
    scene.add(skyDome.mesh);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 26, 20);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, canvasRef.current);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.03;
    controls.minDistance = 4;
    controls.maxDistance = 80;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xf1f5f9, 1.4);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.4);
    dirLight.position.set(12, 32, 15);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    // Accent Diamond/Cyan Point Light on Snake Head
    const playerPointLight = new THREE.PointLight(0xf8fafc, 3.2, 16);
    playerPointLight.position.set(0, 2, 0);
    scene.add(playerPointLight);

    // Arena Ground Plane
    const groundGeo = new THREE.PlaneGeometry(60, 60, 60, 60);
    groundGeo.rotateX(-Math.PI / 2);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x1e2633,
      roughness: 0.45,
      metalness: 0.55,
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.receiveShadow = true;
    scene.add(groundMesh);

    // Grid Overlay
    const gridHelper = new THREE.GridHelper(60, 60, 0x64748b, 0x334155);
    gridHelper.position.y = 0.01;
    scene.add(gridHelper);

    const majorGrid = new THREE.GridHelper(60, 12, 0x0284c7, 0x475569);
    majorGrid.position.y = 0.02;
    scene.add(majorGrid);

    // Arena Perimeter Bounds Line
    const perimeterPoints = [
      new THREE.Vector3(-28, 0.03, -28),
      new THREE.Vector3(28, 0.03, -28),
      new THREE.Vector3(28, 0.03, 28),
      new THREE.Vector3(-28, 0.03, 28),
      new THREE.Vector3(-28, 0.03, -28),
    ];
    const perimeterGeo = new THREE.BufferGeometry().setFromPoints(perimeterPoints);
    const perimeterMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2 });
    const perimeterLine = new THREE.Line(perimeterGeo, perimeterMat);
    scene.add(perimeterLine);

    // White Celestial Snake Rig (Head + Instanced Segments for up to 1000 units)
    const snakeRig = createSnakeRig(1000);
    scene.add(snakeRig.root);
    snakeRigRef.current = snakeRig;

    // Repel Radius Indicator Ring
    const repelGeo = new THREE.RingGeometry(5.8, 6.0, 48);
    repelGeo.rotateX(-Math.PI / 2);
    const repelMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
    });
    const repelCircle = new THREE.Mesh(repelGeo, repelMat);
    repelCircle.position.y = 0.04;
    scene.add(repelCircle);
    repelCircleRef.current = repelCircle;

    // Luminous 3D Particle Point Cloud
    const maxBoids = simulation.swarmField.maxCapacity;
    const swarmGeo = new THREE.BufferGeometry();
    swarmGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(simulation.swarmField.positions, 3)
    );
    swarmGeo.setAttribute(
      'color',
      new THREE.BufferAttribute(simulation.swarmField.colors, 3)
    );
    swarmGeoRef.current = swarmGeo;

    const swarmMat = new THREE.PointsMaterial({
      size: 0.75,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      map: getParticleGlowTexture(),
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const swarmPoints = new THREE.Points(swarmGeo, swarmMat);
    scene.add(swarmPoints);
    swarmPointsRef.current = swarmPoints;

    // Decaying Creature Explosion Particle Point Cloud (Zero-GC BufferGeometry)
    const MAX_DECAY_PARTICLES = 3000;
    const decayPositions = new Float32Array(MAX_DECAY_PARTICLES * 3);
    const decayColors = new Float32Array(MAX_DECAY_PARTICLES * 3);

    const decayGeo = new THREE.BufferGeometry();
    decayGeo.setAttribute('position', new THREE.BufferAttribute(decayPositions, 3));
    decayGeo.setAttribute('color', new THREE.BufferAttribute(decayColors, 3));
    decayGeo.setDrawRange(0, 0);
    decayGeoRef.current = decayGeo;

    const decayMat = new THREE.PointsMaterial({
      size: 0.95,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      map: getDecayingAshTexture(),
      blending: THREE.NormalBlending,
      depthWrite: false,
    });

    const decayPoints = new THREE.Points(decayGeo, decayMat);
    scene.add(decayPoints);
    decayPointsRef.current = decayPoints;

    // Dynamic Snake Tail Trailing Particle System (Luminous Ethereal Wake & Sparks)
    const tailTrailSystem = new SnakeTailTrailSystem({
      maxParticles: 1800,
      baseSize: 0.85,
    });
    scene.add(tailTrailSystem.group);
    tailTrailRef.current = tailTrailSystem;

    // Groups for Cubes and Shockwaves
    const cubeGroup = new THREE.Group();
    scene.add(cubeGroup);
    cubeGroupRef.current = cubeGroup;

    const shockwaveGroup = new THREE.Group();
    scene.add(shockwaveGroup);
    shockwaveGroupRef.current = shockwaveGroup;

    // Pre-allocated shockwave pool for zero-GC rendering
    const MAX_SHOCKWAVES = 8;
    const shockwavePool: THREE.Mesh[] = [];
    const ringBaseGeo = new THREE.RingGeometry(0.92, 1.0, 48);
    ringBaseGeo.rotateX(-Math.PI / 2);

    for (let i = 0; i < MAX_SHOCKWAVES; i++) {
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(ringBaseGeo, ringMat);
      mesh.visible = false;
      mesh.position.y = 0.08;
      shockwaveGroup.add(mesh);
      shockwavePool.push(mesh);
    }

    // Resize Handler & ResizeObserver for dynamic container adjustments (sidebar toggles, zen mode, window resize)
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      if (w <= 0 || h <= 0) return;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h, false);
      rendererRef.current.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    };

    window.addEventListener('resize', handleResize);

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    // Animation Loop
    let animationFrameId: number;
    let lastTime = performance.now();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      const now = performance.now();
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      // Update simulation if running
      if (isRunning) {
        simulation.update(dt);
      }

      // Update Red Snake Model & Kinematics in Full 3D
      if (snakeRigRef.current) {
        const p = simulation.player;
        const growthFactor = Math.max(1.0, (p.snakeLength || 10) / 10);
        updateSnakeRig(
          snakeRigRef.current,
          p.position,
          p.facingAngle,
          p.pitch,
          p.segments,
          p.isDashing,
          p.damageFlashTimer,
          p.growthPulseTimer,
          now * 0.001,
          growthFactor,
          p.headState,
          p.jawAngle,
          p.kineticCrownCharge,
          p.eyeTargetPos,
          {
            jawStyle: config.snakeHeadJawStyle,
            hornStyle: config.snakeHeadHornStyle,
            eyeTracking: config.snakeHeadEyeTracking,
          }
        );

        playerPointLight.position.set(p.position.x, p.position.y + 1.8, p.position.z);

        if (repelCircleRef.current) {
          repelCircleRef.current.position.set(p.position.x, Math.max(0.04, p.position.y - 0.2), p.position.z);
          const scale = p.repelRadius / 6.0;
          repelCircleRef.current.scale.set(scale, scale, scale);
        }

        // Update Dynamic Snake Tail Trailing Particle Effect along 3D tail wake
        if (tailTrailRef.current) {
          const showTrail = config.showSnakeTailTrail !== false;
          tailTrailRef.current.setVisible(showTrail);
          if (showTrail) {
            const r3 = (NUM_RINGS - 1) * 3;
            const spineCenters = snakeRigRef.current.buffers.spineCenters;
            const spineTangents = snakeRigRef.current.buffers.spineTangents;

            let tailX = spineCenters[r3 + 0];
            let tailY = spineCenters[r3 + 1];
            let tailZ = spineCenters[r3 + 2];

            // Fallback to last segment if spine centers not initialized
            if (isNaN(tailX) || (tailX === 0 && tailY === 0 && tailZ === 0 && p.segments.length > 0)) {
              const lastSeg = p.segments[p.segments.length - 1];
              if (lastSeg) {
                tailX = lastSeg.x;
                tailY = lastSeg.y ?? 0.55;
                tailZ = lastSeg.z;
              }
            }

            const tailPos = { x: tailX, y: tailY, z: tailZ };
            const tailTangent = {
              x: spineTangents[r3 + 0] || 0,
              y: spineTangents[r3 + 1] || 0,
              z: spineTangents[r3 + 2] || 1,
            };

            tailTrailRef.current.update(
              tailPos,
              tailTangent,
              dt,
              p.isDashing,
              p.speed || 5.0,
              now * 0.001
            );
          }
        }
      }

      // Update Swarm Point Cloud Geometry Buffers
      if (swarmGeoRef.current && swarmPointsRef.current) {
        const activeCount = simulation.swarmField.count;
        swarmGeoRef.current.setDrawRange(0, activeCount);
        swarmGeoRef.current.attributes.position.needsUpdate = true;
        swarmGeoRef.current.attributes.color.needsUpdate = true;
      }

      // Update Decaying Creature Explosion Particle Point Cloud (Zero-GC Buffer Streaming)
      if (decayGeoRef.current && decayPointsRef.current) {
        const particles = simulation.particles;
        const activeCount = Math.min(particles.length, 3000);
        const posAttr = decayGeoRef.current.attributes.position as THREE.BufferAttribute;
        const colAttr = decayGeoRef.current.attributes.color as THREE.BufferAttribute;
        const posArr = posAttr.array as Float32Array;
        const colArr = colAttr.array as Float32Array;

        for (let i = 0; i < activeCount; i++) {
          const p = particles[i];
          const p3 = i * 3;
          posArr[p3 + 0] = p.position.x;
          posArr[p3 + 1] = p.position.y;
          posArr[p3 + 2] = p.position.z;

          if (p.type === 'decay_creature_explosion') {
            const progress = Math.max(0, Math.min(1, p.life / p.maxLife));
            let r = 0.72, g = 0.74, b = 0.78;
            if (progress > 0.6) {
              const n = (progress - 0.6) / 0.4;
              r = 0.42 + n * (0.75 - 0.42);
              g = 0.45 + n * (0.77 - 0.45);
              b = 0.50 + n * (0.82 - 0.50);
            } else if (progress > 0.25) {
              const n = (progress - 0.25) / 0.35;
              r = 0.15 + n * (0.42 - 0.15);
              g = 0.15 + n * (0.45 - 0.15);
              b = 0.16 + n * (0.50 - 0.16);
            } else {
              const n = progress / 0.25;
              r = 0.035 + n * (0.15 - 0.035);
              g = 0.035 + n * (0.15 - 0.035);
              b = 0.045 + n * (0.16 - 0.045);
            }
            colArr[p3 + 0] = r;
            colArr[p3 + 1] = g;
            colArr[p3 + 2] = b;
          } else {
            // Default particles (e.g. dash crimson red #dc2626)
            colArr[p3 + 0] = 0.86;
            colArr[p3 + 1] = 0.15;
            colArr[p3 + 2] = 0.15;
          }
        }

        decayGeoRef.current.setDrawRange(0, activeCount);
        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;
      }

      // Sync Tactical Terrain Cubes
      if (cubeGroupRef.current) {
        const deformations = simulation.terrainDeformations;
        const currentMap = cubeMeshesMapRef.current;
        const activeIds = new Set<string>();

        deformations.forEach((deform) => {
          activeIds.add(deform.id);
          if (!currentMap.has(deform.id)) {
            const isCarved = deform.heightChange < 0;
            const h = deform.height ?? Math.abs(deform.heightChange);
            const geo = new THREE.BoxGeometry(
              deform.width ?? deform.cubeSize,
              h,
              deform.depth ?? deform.cubeSize
            );
            const mat = new THREE.MeshStandardMaterial({
              color: isCarved ? 0x0f172a : 0x38bdf8,
              roughness: 0.3,
              metalness: 0.7,
              emissive: isCarved ? 0x020617 : 0x0284c7,
              emissiveIntensity: isCarved ? 0.1 : 0.4,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(deform.point.x, isCarved ? -h * 0.5 : h * 0.5, deform.point.z);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            cubeGroupRef.current?.add(mesh);
            currentMap.set(deform.id, mesh);
          }
        });

        // Remove deleted cubes
        currentMap.forEach((mesh, id) => {
          if (!activeIds.has(id)) {
            cubeGroupRef.current?.remove(mesh);
            mesh.geometry.dispose();
            (mesh.material as THREE.Material).dispose();
            currentMap.delete(id);
          }
        });
      }

      // Update Shockwaves from Pre-allocated Pool (Zero Allocations)
      const shockwaves = simulation.shockwaves;
      for (let i = 0; i < shockwavePool.length; i++) {
        const mesh = shockwavePool[i];
        if (i < shockwaves.length) {
          const sw = shockwaves[i];
          const scale = Math.max(0.1, sw.radius);
          mesh.scale.set(scale, scale, scale);
          mesh.position.set(sw.position.x, 0.08, sw.position.z);
          const mat = mesh.material as THREE.MeshBasicMaterial;
          mat.opacity = Math.max(0, sw.life / sw.maxLife) * 0.85;
          mesh.visible = true;
        } else {
          mesh.visible = false;
        }
      }

      // Camera Follow Tracking in Full 3D
      if (cameraPreset === 'overhead_follow_red' && cameraRef.current && controlsRef.current) {
        const px = simulation.player.position.x;
        const py = simulation.player.position.y;
        const pz = simulation.player.position.z;

        const targetCamX = px;
        const targetCamZ = pz + 18;
        const targetCamY = py + 16;

        cameraRef.current.position.lerp(
          new THREE.Vector3(targetCamX, targetCamY, targetCamZ),
          0.05
        );
        controlsRef.current.target.lerp(new THREE.Vector3(px, py, pz), 0.08);
      }

      // Update Animated Scrolling Noise Cloud Sky Dome
      skyDome.update(now * 0.001, cameraRef.current?.position);

      controlsRef.current?.update();
      rendererRef.current?.render(sceneRef.current!, cameraRef.current!);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      skyDome.dispose();
      tailTrailSystem.dispose();
      renderer.dispose();
    };
  }, [simulation, isRunning, cameraPreset]);

  // Immediately respond to layout, sidebar, and Zen Mode toggles
  useEffect(() => {
    const handleLayoutResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      if (w <= 0 || h <= 0) return;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h, false);
    };

    handleLayoutResize();
    const frameId = requestAnimationFrame(handleLayoutResize);
    const timeoutId = setTimeout(handleLayoutResize, 50);

    return () => {
      cancelAnimationFrame(frameId);
      clearTimeout(timeoutId);
    };
  }, [isSidebarOpen, isZenMode]);

  // Handle Keyboard Shortcuts for Camera, Debug Mode & Clean View
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      // Key 1 / 2 / 3 -> Quick Camera Views
      if (e.code === 'Digit1' || e.code === 'Numpad1') {
        setCameraView('overhead_follow_red');
      } else if (e.code === 'Digit2' || e.code === 'Numpad2') {
        setCameraView('free_orbit');
      } else if (e.code === 'Digit3' || e.code === 'Numpad3') {
        setCameraView('top_down');
      }

      // KeyG -> Toggle Mesh Debug Mode & Coordinate Frames
      if (e.code === 'KeyG') {
        setIsDebugMeshEnabled((prev) => {
          const next = !prev;
          if (snakeRigRef.current) {
            setSnakeDebugMode(snakeRigRef.current, next);
          }
          return next;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Handle Camera Presets
  const setCameraView = (view: 'overhead_follow_red' | 'free_orbit' | 'top_down') => {
    setCameraPreset(view);
    if (!cameraRef.current || !controlsRef.current) return;

    if (view === 'overhead_follow_red') {
      const px = simulation.player.position.x;
      const pz = simulation.player.position.z;
      cameraRef.current.position.set(px, 22, pz + 18);
      controlsRef.current.target.set(px, 0.55, pz);
    } else if (view === 'top_down') {
      cameraRef.current.position.set(0, 38, 0.1);
      controlsRef.current.target.set(0, 0, 0);
    } else {
      cameraRef.current.position.set(0, 26, 22);
      controlsRef.current.target.set(0, 0, 0);
    }
  };

  const rlStats = simulation.stats.rlAgent;
  const snakeLen = simulation.player.snakeLength;
  const maxCap = simulation.player.snakeMaxCap;
  const lenPercent = Math.min(100, (snakeLen / maxCap) * 100);
  const nextGrow = Math.max(0, simulation.player.elongateInterval - simulation.player.elongateTimer);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-[#7a8b99] overflow-hidden select-none">
      <canvas ref={canvasRef} className="w-full h-full block cursor-grab active:cursor-grabbing" />

      {/* Clean View: Camera Modes Only (Follow Snake & Free Orbit) */}
      {isZenMode && (
        <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-slate-950/80 hover:bg-slate-950/95 backdrop-blur-md p-1.5 rounded-2xl border border-slate-800/80 shadow-2xl text-xs font-mono transition-all">
          <button
            onClick={() => setCameraView('overhead_follow_red')}
            title="Follow Snake Camera (Press 1)"
            className={`px-3 py-1.5 rounded-xl font-medium transition-all ${
              cameraPreset === 'overhead_follow_red'
                ? 'bg-red-600 text-white font-bold shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
            }`}
          >
            Follow Snake
          </button>
          <button
            onClick={() => setCameraView('free_orbit')}
            title="Free Orbit Camera (Press 2)"
            className={`px-3 py-1.5 rounded-xl font-medium transition-all ${
              cameraPreset === 'free_orbit'
                ? 'bg-red-600 text-white font-bold shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
            }`}
          >
            Free Orbit
          </button>
          <button
            onClick={() => setCameraView('top_down')}
            title="Top Down Map Camera (Press 3)"
            className={`px-3 py-1.5 rounded-xl font-medium transition-all ${
              cameraPreset === 'top_down'
                ? 'bg-red-600 text-white font-bold shadow-md'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
            }`}
          >
            Top Down
          </button>
        </div>
      )}

      {/* Floating Edge Tab to Re-open Parameters Panel (when hidden and not in Zen mode) */}
      {!isSidebarOpen && !isZenMode && onToggleSidebar && (
        <button
          onClick={onToggleSidebar}
          title="Show Simulation Parameters Panel"
          className="absolute top-1/2 -translate-y-1/2 right-0 z-20 px-2 py-4 rounded-l-2xl bg-slate-950/90 hover:bg-slate-900 text-slate-200 hover:text-white border-y border-l border-red-500/50 backdrop-blur-md text-xs font-bold flex flex-col items-center gap-2 shadow-2xl transition-all active:scale-95 group hover:border-red-400 cursor-pointer"
        >
          <Sliders className="w-4 h-4 text-red-400 group-hover:scale-110 transition-transform" />
          <span className="text-[10px] tracking-wider uppercase [writing-mode:vertical-rl] rotate-180 font-mono text-slate-300 font-semibold">
            Show Panel
          </span>
        </button>
      )}

      {/* TOP LEFT: Live RL Decision HUD & Snake Metrics */}
      {!isZenMode && (
        <div className="absolute top-3 left-3 z-10 space-y-2 max-w-sm">
          {/* Snake Length Status Card */}
          <div className="p-3 bg-slate-950/85 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
                  <Dna className="w-4 h-4 text-cyan-300" />
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider">Coral Snake Length</span>
                  <div className="flex items-baseline gap-1.5">
                    <span className="text-base font-black text-white font-mono">{snakeLen}</span>
                    <span className="text-[10px] text-slate-400 font-mono">/ {maxCap} max</span>
                  </div>
                </div>
              </div>
              <div className="text-right font-mono">
                <span className="text-[10px] text-slate-400 block">+1 grow in</span>
                <span className="text-xs font-bold text-emerald-400">{nextGrow.toFixed(1)}s</span>
              </div>
            </div>

            {/* Snake Growth Progress Bar */}
            <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-slate-200 via-cyan-400 to-emerald-400 transition-all duration-300 rounded-full"
                style={{ width: `${lenPercent}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 pt-0.5">
              <span>Peak: {simulation.player.peakLength}</span>
              <span>Bites Deflected: {simulation.player.shortenCount}</span>
              <span>Survival: {simulation.player.survivalTime.toFixed(1)}s</span>
            </div>
          </div>

          {/* Main RL Telemetry Card (SAC Continuous vs Q-Learning) */}
          <div className="p-3 bg-slate-950/85 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl space-y-2.5">
            <div
              onClick={() => setIsRLHudExpanded(!isRLHudExpanded)}
              className="flex items-center justify-between cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <div className={`w-6 h-6 rounded-lg flex items-center justify-center ${
                  config.sacEnabled
                    ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400'
                    : 'bg-rose-500/20 border border-rose-500/40 text-rose-400'
                }`}>
                  {config.sacEnabled ? <Cpu className="w-3.5 h-3.5" /> : <Brain className="w-3.5 h-3.5" />}
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
                    <span>{config.sacEnabled ? 'SAC Continuous 3D' : 'Tabular Q-Brain'}</span>
                    <span
                      className={`text-[9px] px-1.5 py-0.2 rounded font-mono ${
                        config.sacEnabled
                          ? (config.sacIsEvaluation
                              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30')
                          : (rlStats.isExploring
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30')
                      }`}
                    >
                      {config.sacEnabled
                        ? (config.sacIsEvaluation ? 'EVALUATION' : 'TRAINING')
                        : (rlStats.isExploring ? 'EXPLORING' : 'EXPLOITING')}
                    </span>
                  </h3>
                </div>
              </div>
              {isRLHudExpanded ? (
                <ChevronUp className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              )}
            </div>

            {isRLHudExpanded && (
              <div className="space-y-2 pt-1 border-t border-slate-800/80 text-xs">
                {config.sacEnabled ? (
                  <>
                    {/* Continuous 7D Action Vector [ax, ay, az, a_repel, a_dash, a_barrier, a_trench] */}
                    <div className="bg-slate-900/90 p-2 rounded-xl border border-slate-800 space-y-1.5">
                      <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                        <span>7D Action Space $a_t \in [-1, 1]^7$</span>
                        <span className="text-emerald-400 font-bold">
                          Hold: {config.sacActionPersistence ?? 4} ticks
                        </span>
                      </div>
                      
                      {/* 3D Flight Controls */}
                      <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-[11px]">
                        <div className="p-1 rounded bg-slate-950 border border-slate-800">
                          <div className="text-[9px] text-slate-400">a_x (Steer)</div>
                          <span className={`font-bold ${(simulation.sacAgent.currentAction[0] ?? 0) >= 0 ? 'text-cyan-400' : 'text-rose-400'}`}>
                            {(simulation.sacAgent.currentAction[0] ?? 0).toFixed(2)}
                          </span>
                        </div>
                        <div className="p-1 rounded bg-slate-950 border border-slate-800">
                          <div className="text-[9px] text-slate-400">a_y (Altitude)</div>
                          <span className={`font-bold ${(simulation.sacAgent.currentAction[1] ?? 0) >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                            {(simulation.sacAgent.currentAction[1] ?? 0).toFixed(2)}
                          </span>
                        </div>
                        <div className="p-1 rounded bg-slate-950 border border-slate-800">
                          <div className="text-[9px] text-slate-400">a_z (Depth)</div>
                          <span className={`font-bold ${(simulation.sacAgent.currentAction[2] ?? 0) >= 0 ? 'text-indigo-400' : 'text-purple-400'}`}>
                            {(simulation.sacAgent.currentAction[2] ?? 0).toFixed(2)}
                          </span>
                        </div>
                      </div>

                      {/* Tactical Ability Intent Gauges */}
                      <div className="pt-1 border-t border-slate-800/60">
                        <div className="text-[9px] text-slate-400 font-mono mb-1">Tactical Ability Intents & Executions:</div>
                        <div className="grid grid-cols-4 gap-1 text-center font-mono text-[10px]">
                          <div className={`p-1 rounded border ${(simulation.sacAgent.currentAction[3] ?? 0) > 0.2 ? 'bg-red-950/80 border-red-500 text-red-300 font-bold animate-pulse' : 'bg-slate-950 border-slate-800 text-slate-400'}`}>
                            <div>⚡ Repel</div>
                            <div className="text-[9px]">{(simulation.sacAgent.currentAction[3] ?? 0).toFixed(2)}</div>
                            <div className="text-[8px] text-red-400 font-bold">x{simulation.sacAgent.abilityExecutions.repels}</div>
                          </div>
                          <div className={`p-1 rounded border ${(simulation.sacAgent.currentAction[4] ?? 0) > 0.25 ? 'bg-sky-950/80 border-sky-500 text-sky-300 font-bold animate-pulse' : 'bg-slate-950 border-slate-800 text-slate-400'}`}>
                            <div>🚀 Dash</div>
                            <div className="text-[9px]">{(simulation.sacAgent.currentAction[4] ?? 0).toFixed(2)}</div>
                            <div className="text-[8px] text-sky-400 font-bold">x{simulation.sacAgent.abilityExecutions.dashes}</div>
                          </div>
                          <div className={`p-1 rounded border ${(simulation.sacAgent.currentAction[5] ?? 0) > 0.3 ? 'bg-cyan-950/80 border-cyan-500 text-cyan-300 font-bold animate-pulse' : 'bg-slate-950 border-slate-800 text-slate-400'}`}>
                            <div>🛡️ Barrier</div>
                            <div className="text-[9px]">{(simulation.sacAgent.currentAction[5] ?? 0).toFixed(2)}</div>
                            <div className="text-[8px] text-cyan-400 font-bold">x{simulation.sacAgent.abilityExecutions.barriers}</div>
                          </div>
                          <div className={`p-1 rounded border ${(simulation.sacAgent.currentAction[6] ?? 0) > 0.35 ? 'bg-purple-950/80 border-purple-500 text-purple-300 font-bold animate-pulse' : 'bg-slate-950 border-slate-800 text-slate-400'}`}>
                            <div>⛏️ Trench</div>
                            <div className="text-[9px]">{(simulation.sacAgent.currentAction[6] ?? 0).toFixed(2)}</div>
                            <div className="text-[8px] text-purple-400 font-bold">x{simulation.sacAgent.abilityExecutions.trenches}</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Reward & Telemetry */}
                    <div className="grid grid-cols-4 gap-1.5 text-center font-mono text-[11px]">
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[8px] text-slate-400">Step Reward</div>
                        <span className={`font-bold text-[10px] ${simulation.stats.lastStepReward >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {simulation.stats.lastStepReward.toFixed(2)}
                        </span>
                      </div>
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[8px] text-slate-400">Mean Length</div>
                        <span className="font-bold text-[10px] text-amber-300">
                          {(simulation.stats.sacMetrics?.meanLength ?? simulation.player.snakeLength).toFixed(1)}
                        </span>
                      </div>
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[8px] text-slate-400">PER SumTree</div>
                        <span className="font-bold text-[10px] text-amber-400">
                          {simulation.stats.sacMetrics?.usePER ? `α=${(simulation.stats.sacMetrics?.perAlpha ?? 0.6).toFixed(1)}` : 'OFF'}
                        </span>
                      </div>
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[8px] text-slate-400">N-Step Ret</div>
                        <span className="font-bold text-[10px] text-cyan-300">
                          {simulation.stats.sacMetrics?.useNStep ? `${simulation.stats.sacMetrics?.nStep ?? 3} steps` : '1 step'}
                        </span>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    {/* State & Action Row */}
                    <div className="flex items-center justify-between bg-slate-900/90 p-2 rounded-xl border border-slate-800 font-mono text-[11px]">
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase">State Vector</div>
                        <span className="text-cyan-300 font-semibold">{rlStats.lastStateKey}</span>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-slate-400 uppercase">Chosen Action</div>
                        <span className="text-rose-400 font-bold uppercase">
                          {rlStats.lastAction ?? 'evade_kite'}
                        </span>
                      </div>
                    </div>

                    {/* Reward & Exploration Metrics */}
                    <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-[11px]">
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[9px] text-slate-400">Total Reward</div>
                        <span
                          className={`font-bold ${
                            rlStats.totalReward >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          {rlStats.totalReward.toFixed(1)}
                        </span>
                      </div>
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[9px] text-slate-400">Epsilon (ε)</div>
                        <span className="font-bold text-amber-300">
                          {(rlStats.epsilon * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="p-1.5 bg-slate-900/60 rounded-lg border border-slate-800">
                        <div className="text-[9px] text-slate-400">Episode</div>
                        <span className="font-bold text-slate-200">#{rlStats.episodesCount}</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TOP RIGHT: Camera Controls & Help Guide */}
      {!isZenMode && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5 bg-slate-950/85 backdrop-blur-md p-1 rounded-xl border border-slate-800 shadow-lg text-xs">
          <button
            onClick={() => setCameraView('overhead_follow_red')}
            className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
              cameraPreset === 'overhead_follow_red'
                ? 'bg-red-600 text-white font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Follow Snake
          </button>
          <button
            onClick={() => setCameraView('free_orbit')}
            className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
              cameraPreset === 'free_orbit'
                ? 'bg-red-600 text-white font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Free Orbit
          </button>
          <button
            onClick={() => setCameraView('top_down')}
            className={`px-2.5 py-1 rounded-lg font-medium transition-all ${
              cameraPreset === 'top_down'
                ? 'bg-red-600 text-white font-bold shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Top Down
          </button>

          {/* Procedural Snake Mesh & Frame Debug Toggle */}
          <button
            onClick={() => {
              const next = !isDebugMeshEnabled;
              setIsDebugMeshEnabled(next);
              if (snakeRigRef.current) {
                setSnakeDebugMode(snakeRigRef.current, next);
              }
            }}
            title="Toggle Procedural Snake 12-Sided Wireframe Rings, Spine Sample Points, and Bishop Coordinate Frames"
            className={`px-2.5 py-1 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
              isDebugMeshEnabled
                ? 'bg-cyan-600 text-white font-bold shadow shadow-cyan-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>Debug Mode</span>
          </button>

          {/* Parameters Panel Toggle */}
          {onToggleSidebar && (
            <button
              onClick={onToggleSidebar}
              title={isSidebarOpen ? 'Hide Parameters Panel' : 'Show Parameters Panel'}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                isSidebarOpen
                  ? 'bg-red-600 text-white font-bold shadow-sm shadow-red-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sliders className="w-3.5 h-3.5 text-red-400" />
              <span>{isSidebarOpen ? 'Hide Panel' : 'Show Panel'}</span>
            </button>
          )}

          <button
            onClick={() => setShowControlsGuide(true)}
            title="View Keybindings & Controls Guide"
            className="p-1.5 rounded-lg text-slate-400 hover:text-cyan-300 hover:bg-slate-800 transition-colors"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* TOP RIGHT: Floating Procedural Snake Mesh Debug HUD */}
      {!isZenMode && isDebugMeshEnabled && (
        <div className="absolute top-14 right-3 z-20 w-80 bg-slate-950/90 backdrop-blur-md p-3.5 rounded-2xl border border-cyan-500/40 shadow-2xl space-y-2.5 text-xs animate-fade-in">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                <Code2 className="w-3.5 h-3.5" />
              </div>
              <div>
                <h4 className="text-xs font-bold text-white">Snake Mesh Inspector</h4>
                <p className="text-[10px] text-slate-400 font-mono">12-Sided Smooth Bishop Tube</p>
              </div>
            </div>
            <button
              onClick={() => {
                setIsDebugMeshEnabled(false);
                if (snakeRigRef.current) setSnakeDebugMode(snakeRigRef.current, false);
              }}
              className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Mesh Geometry Metrics */}
          <div className="grid grid-cols-2 gap-1.5 font-mono text-[11px]">
            <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[9px] text-slate-400 block">Radial Sides</span>
              <span className="font-bold text-cyan-400">{RADIAL_SEGMENTS} sides (Smooth Normals)</span>
            </div>
            <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[9px] text-slate-400 block">Cross Rings</span>
              <span className="font-bold text-emerald-400">{NUM_RINGS} Rings (Zero GC)</span>
            </div>
            <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[9px] text-slate-400 block">Total Arc Length</span>
              <span className="font-bold text-amber-400">
                {snakeRigRef.current?.telemetry.totalArcLength.toFixed(2) ?? '0.00'}m
              </span>
            </div>
            <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800">
              <span className="text-[9px] text-slate-400 block">Thickness Scale</span>
              <span className="font-bold text-purple-400">
                {snakeRigRef.current?.telemetry.bodyThickness.toFixed(2) ?? '1.00'}x (Mild)
              </span>
            </div>
          </div>

          {/* Coordinate Frame Vector Legend */}
          <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1 font-mono text-[10px]">
            <div className="text-slate-400 font-bold mb-1">Parallel-Transport Axes (No Twist):</div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-blue-400 font-bold">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Tangent (T)
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Normal / Up (N)
              </span>
              <span className="flex items-center gap-1.5 text-rose-400 font-bold">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500" /> Binormal (B)
              </span>
            </div>
          </div>

          {/* Catmull-Rom Radius Taper Profile */}
          <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
              <span>Catmull-Rom Taper Profile</span>
              <span className="text-cyan-400 font-bold">Head → Tail</span>
            </div>
            {/* Visual Mini Graph of Taper Profile */}
            <div className="h-10 flex items-end gap-1 px-1 bg-slate-950 rounded-lg border border-slate-800/80 pt-1">
              {TAPER_PROFILE.map((pt, idx) => (
                <div
                  key={idx}
                  className="flex-1 flex flex-col items-center justify-end h-full"
                  title={`t: ${pt.t.toFixed(2)} | radius: ${pt.radius.toFixed(2)}`}
                >
                  <div
                    className="w-full bg-gradient-to-t from-cyan-600 to-emerald-400 rounded-t-sm"
                    style={{ height: `${Math.max(8, pt.radius * 90)}%` }}
                  />
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between text-[9px] font-mono text-slate-400">
              <span>Cranium (0.82)</span>
              <span>Thorax (1.05)</span>
              <span>Mid (1.00)</span>
              <span>Tip (0.015)</span>
            </div>
          </div>
        </div>
      )}

      {/* BOTTOM CENTER: Tactical Survival Action Deck & Keybinds */}
      {!isZenMode && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 bg-slate-950/90 backdrop-blur-md p-2 rounded-2xl border border-slate-800 shadow-2xl max-w-[calc(100vw-24px)] overflow-x-auto">
          {/* Kinetic Repel Blast Button */}
          <button
            onClick={onManualRepel}
            disabled={simulation.player.repelCooldown > 0}
            title="Trigger Kinetic Shockwave Repel Blast (Space / E)"
            className="relative px-3.5 py-2 rounded-xl bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 disabled:opacity-40 text-white font-bold text-xs flex items-center gap-2 shadow-lg shadow-red-600/30 transition-all active:scale-95 whitespace-nowrap"
          >
            <Shield className="w-4 h-4" />
            <span>Blast Repel</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/30 border border-white/20">
              {simulation.player.repelCooldown > 0
                ? `${simulation.player.repelCooldown.toFixed(1)}s`
                : 'SPACE'}
            </span>
          </button>

          {/* Tactical Dash Button */}
          <button
            onClick={onManualDash}
            disabled={simulation.player.dashCooldown > 0}
            title="Rapid Snake Surge Dash (Shift)"
            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 font-bold text-xs flex items-center gap-2 border border-slate-700 transition-all active:scale-95 whitespace-nowrap"
          >
            <Zap className="w-4 h-4 text-amber-400" />
            <span>Surge Dash</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/30 text-slate-300">
              {simulation.player.dashCooldown > 0
                ? `${simulation.player.dashCooldown.toFixed(1)}s`
                : 'SHIFT'}
            </span>
          </button>

          {/* 3D Vertical Flight: Ascend */}
          <button
            onClick={onManualAscend}
            title="Ascend Sky / Pitch Up into 3D Altitude (R / Hold Space)"
            className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-emerald-300 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition-all active:scale-95 whitespace-nowrap"
          >
            <ChevronUp className="w-4 h-4 text-emerald-400" />
            <span>Climb</span>
            <span className="text-[10px] font-mono px-1 py-0.5 rounded bg-black/30 text-emerald-400">
              R
            </span>
          </button>

          {/* 3D Vertical Flight: Descend */}
          <button
            onClick={onManualDescend}
            title="Descend Dive / Pitch Down to Ground (F / Hold Shift)"
            className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-sky-300 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition-all active:scale-95 whitespace-nowrap"
          >
            <ChevronDown className="w-4 h-4 text-sky-400" />
            <span>Dive</span>
            <span className="text-[10px] font-mono px-1 py-0.5 rounded bg-black/30 text-sky-400">
              F
            </span>
          </button>

          {/* Altitude Telemetry Indicator */}
          <div className="px-2.5 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col items-center justify-center whitespace-nowrap">
            <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider">3D Alt</span>
            <span className="text-[11px] font-mono font-bold text-emerald-400">
              {simulation.player.position.y.toFixed(1)}m
            </span>
          </div>

          {/* Carve Trench Button */}
          <button
            onClick={onCarveTrench}
            title="Carve Terrain Trench Channel (T / C)"
            className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition-all active:scale-95 whitespace-nowrap"
          >
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span>Trench</span>
            <span className="text-[10px] font-mono px-1 py-0.5 rounded bg-black/30 text-slate-400">
              T
            </span>
          </button>

          {/* Spawn Swarm Injection */}
          <button
            onClick={onManualSpawn}
            title="Inject +20 Swarm Boids (S)"
            className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
          >
            <Flame className="w-4 h-4 text-orange-400" />
          </button>
        </div>
      )}

      {/* Game Over / Defeat Overlay */}
      {simulation.player.isDead && (
        <div className="absolute inset-0 z-40 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-red-500/50 rounded-3xl p-6 max-w-md w-full shadow-2xl text-center space-y-4 animate-fade-in">
            <div className="w-12 h-12 rounded-2xl bg-red-600/20 border border-red-500/40 flex items-center justify-center mx-auto text-red-500 shadow-lg shadow-red-950/50">
              <Flame className="w-6 h-6 animate-pulse" />
            </div>
            <div className="space-y-1">
              <h2 className="text-xl font-black text-white uppercase tracking-wider">Snake Perished</h2>
              <p className="text-xs text-slate-400">The swarm overwhelmed the continuous serpentine body.</p>
            </div>

            <div className="grid grid-cols-3 gap-2 bg-slate-900/80 p-3 rounded-2xl border border-slate-800 font-mono text-xs">
              <div>
                <span className="text-[10px] text-slate-400 block">Peak Length</span>
                <span className="text-sm font-black text-white">{simulation.player.peakLength}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-400 block">Survived</span>
                <span className="text-sm font-black text-cyan-300">{simulation.player.survivalTime.toFixed(1)}s</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-400 block">Repelled</span>
                <span className="text-sm font-black text-emerald-400">{simulation.stats.totalRepelled}</span>
              </div>
            </div>

            <button
              onClick={onReset}
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-red-600/30 transition-all active:scale-95"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Respawn Episode (Press X)</span>
            </button>
          </div>
        </div>
      )}

      {/* Controls & Keybindings Quick Modal */}
      {showControlsGuide && (
        <div
          className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setShowControlsGuide(false)}
        >
          <div
            className="bg-slate-950 border border-slate-800 rounded-3xl p-6 max-w-lg w-full shadow-2xl space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-cyan-400" />
                <h3 className="font-bold text-sm text-slate-100">Controls & Keybindings</h3>
              </div>
              <button
                onClick={() => setShowControlsGuide(false)}
                className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Steering</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-300 font-bold">W A S D</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">3D Climb / Dive</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-emerald-300 font-bold">R / F</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Repel Blast</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-red-300 font-bold">SPACE</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Surge Dash</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-amber-300 font-bold">SHIFT</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Carve Trench</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-indigo-300 font-bold">T / C</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Toggle RL / Manual</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-rose-300 font-bold">M</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Camera Presets</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-sky-300 font-bold">1 / 2 / 3</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Clean / Zen View</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-amber-300 font-bold">H</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 flex justify-between items-center">
                <span className="text-slate-400">Mesh Rig Debug</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-300 font-bold">G</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
