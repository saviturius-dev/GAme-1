import * as THREE from 'three';

export type MannequinType = 'partyA' | 'partyB' | 'minion' | 'redPlayer';

export interface MannequinRig {
  root: THREE.Group;
  type: MannequinType;
  pelvis: THREE.Mesh;
  chest: THREE.Mesh;
  head: THREE.Mesh;
  visor: THREE.Mesh;
  core: THREE.Mesh;
  leftShoulder: THREE.Group;
  rightShoulder: THREE.Group;
  leftElbow: THREE.Group;
  rightElbow: THREE.Group;
  leftHip: THREE.Group;
  rightHip: THREE.Group;
  leftKnee: THREE.Group;
  rightKnee: THREE.Group;
  leftWing?: THREE.Group;
  rightWing?: THREE.Group;
  auraRing?: THREE.Mesh;
  materials: {
    primary: THREE.MeshStandardMaterial;
    secondary: THREE.MeshStandardMaterial;
    accent: THREE.MeshStandardMaterial;
    joint: THREE.MeshStandardMaterial;
    glow: THREE.MeshStandardMaterial;
    wing?: THREE.MeshStandardMaterial;
  };
}

/**
 * Creates a procedural articulated humanoid mannequin mesh hierarchy
 */
export function createHumanoidMannequin(type: MannequinType): MannequinRig {
  const root = new THREE.Group();
  root.name = `mannequin_${type}`;

  // Dimensions & Colors per character type
  let primaryColor = 0xdc2626; // Red for Survivor Player
  let secondaryColor = 0x18181b; // Dark graphite
  let glowColor = 0xf87171; // Radiant red
  let heightScale = 1.15;

  if (type === 'redPlayer' || type === 'partyB') {
    primaryColor = 0xdc2626; // Crimson Survivor / Red Player
    secondaryColor = 0x0f172a;
    glowColor = 0xf87171; // Radiant fiery red
    heightScale = 1.18;
  } else if (type === 'partyA') {
    primaryColor = 0x2563eb;
    secondaryColor = 0x0f172a;
    glowColor = 0x22d3ee;
    heightScale = 1.15;
  } else {
    // Minion
    primaryColor = 0xf97316;
    secondaryColor = 0x27272a;
    glowColor = 0xfde047;
    heightScale = 0.55;
  }

  // Shared reusable materials
  const primaryMat = new THREE.MeshStandardMaterial({
    color: primaryColor,
    roughness: 0.3,
    metalness: 0.35,
    emissive: primaryColor,
    emissiveIntensity: 0.2,
  });

  const secondaryMat = new THREE.MeshStandardMaterial({
    color: secondaryColor,
    roughness: 0.4,
    metalness: 0.6,
  });

  const jointMat = new THREE.MeshStandardMaterial({
    color: 0x475569,
    roughness: 0.2,
    metalness: 0.8,
  });

  const accentMat = new THREE.MeshStandardMaterial({
    color: glowColor,
    roughness: 0.2,
    metalness: 0.4,
    emissive: glowColor,
    emissiveIntensity: 0.6,
  });

  const glowMat = new THREE.MeshStandardMaterial({
    color: glowColor,
    roughness: 0.1,
    metalness: 0.1,
    emissive: glowColor,
    emissiveIntensity: 1.2,
  });

  // Reusable Geometries
  const jointSphereGeo = new THREE.SphereGeometry(0.12 * heightScale, 16, 12);
  const limbGeo = new THREE.CylinderGeometry(
    0.09 * heightScale,
    0.075 * heightScale,
    0.35 * heightScale,
    14
  );
  const forearmGeo = new THREE.CylinderGeometry(
    0.075 * heightScale,
    0.06 * heightScale,
    0.32 * heightScale,
    14
  );
  const thighGeo = new THREE.CylinderGeometry(
    0.11 * heightScale,
    0.09 * heightScale,
    0.42 * heightScale,
    14
  );
  const shinGeo = new THREE.CylinderGeometry(
    0.09 * heightScale,
    0.07 * heightScale,
    0.4 * heightScale,
    14
  );

  // 1. Pelvis / Hips (Root of body hierarchy)
  const pelvisGeo = new THREE.CylinderGeometry(
    0.22 * heightScale,
    0.18 * heightScale,
    0.18 * heightScale,
    16
  );
  const pelvis = new THREE.Mesh(pelvisGeo, secondaryMat);
  pelvis.position.y = 0.85 * heightScale;
  pelvis.castShadow = true;
  pelvis.receiveShadow = true;
  root.add(pelvis);

  // Spine Joint
  const spineBall = new THREE.Mesh(jointSphereGeo, jointMat);
  spineBall.position.y = 0.12 * heightScale;
  pelvis.add(spineBall);

  // 2. Torso / Chest
  const chestGeo = new THREE.BoxGeometry(
    0.42 * heightScale,
    0.38 * heightScale,
    0.26 * heightScale
  );
  const chest = new THREE.Mesh(chestGeo, primaryMat);
  chest.position.y = 0.3 * heightScale;
  chest.castShadow = true;
  chest.receiveShadow = true;
  pelvis.add(chest);

  // Glowing Core Reactor in chest center
  const coreGeo = new THREE.SphereGeometry(0.09 * heightScale, 16, 16);
  const core = new THREE.Mesh(coreGeo, glowMat);
  core.position.set(0, 0.04 * heightScale, 0.13 * heightScale);
  chest.add(core);

  // Neck
  const neckBall = new THREE.Mesh(jointSphereGeo, jointMat);
  neckBall.position.y = 0.24 * heightScale;
  neckBall.scale.set(0.7, 0.7, 0.7);
  chest.add(neckBall);

  // 3. Head & Visor
  const headGeo = new THREE.SphereGeometry(0.18 * heightScale, 20, 16);
  headGeo.scale(1, 1.15, 1.05);
  const head = new THREE.Mesh(headGeo, secondaryMat);
  head.position.y = 0.22 * heightScale;
  head.castShadow = true;
  chest.add(head);

  // Glowing Visor / Face Plate
  const visorGeo = new THREE.BoxGeometry(
    0.26 * heightScale,
    0.09 * heightScale,
    0.12 * heightScale
  );
  const visor = new THREE.Mesh(visorGeo, glowMat);
  visor.position.set(0, 0.02 * heightScale, 0.14 * heightScale);
  head.add(visor);

  // Special Crown / Horns for Summoner (Party B)
  if (type === 'partyB') {
    const hornGeo = new THREE.ConeGeometry(0.04 * heightScale, 0.22 * heightScale, 8);
    const leftHorn = new THREE.Mesh(hornGeo, accentMat);
    leftHorn.position.set(0.12 * heightScale, 0.18 * heightScale, 0);
    leftHorn.rotation.z = -0.35;
    leftHorn.rotation.x = -0.15;

    const rightHorn = new THREE.Mesh(hornGeo, accentMat);
    rightHorn.position.set(-0.12 * heightScale, 0.18 * heightScale, 0);
    rightHorn.rotation.z = 0.35;
    rightHorn.rotation.x = -0.15;

    head.add(leftHorn, rightHorn);
  }

  // Special Energy Wings for Minions
  let leftWing: THREE.Group | undefined;
  let rightWing: THREE.Group | undefined;
  let wingMat: THREE.MeshStandardMaterial | undefined;

  if (type === 'minion') {
    wingMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xd97706,
      emissiveIntensity: 1.3,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
      roughness: 0.2,
      metalness: 0.7,
    });

    const wingShape = new THREE.Shape();
    wingShape.moveTo(0, 0);
    wingShape.lineTo(0.35, 0.12);
    wingShape.lineTo(0.58, 0.04);
    wingShape.lineTo(0.5, -0.16);
    wingShape.lineTo(0.2, -0.09);
    wingShape.closePath();

    const wingGeo = new THREE.ShapeGeometry(wingShape);

    leftWing = new THREE.Group();
    leftWing.position.set(0.08 * heightScale, 0.12 * heightScale, -0.1 * heightScale);
    const leftWingMesh = new THREE.Mesh(wingGeo, wingMat);
    leftWing.add(leftWingMesh);
    chest.add(leftWing);

    rightWing = new THREE.Group();
    rightWing.position.set(-0.08 * heightScale, 0.12 * heightScale, -0.1 * heightScale);
    const rightWingMesh = new THREE.Mesh(wingGeo, wingMat);
    rightWingMesh.scale.set(-1, 1, 1);
    rightWing.add(rightWingMesh);
    chest.add(rightWing);
  }

  // Special Guardian Pauldrons for Repeller (Party A)
  if (type === 'partyA') {
    const pauldronGeo = new THREE.BoxGeometry(
      0.16 * heightScale,
      0.08 * heightScale,
      0.24 * heightScale
    );
    const leftPauldron = new THREE.Mesh(pauldronGeo, accentMat);
    leftPauldron.position.set(0.28 * heightScale, 0.15 * heightScale, 0);
    leftPauldron.rotation.z = -0.2;

    const rightPauldron = new THREE.Mesh(pauldronGeo, accentMat);
    rightPauldron.position.set(-0.28 * heightScale, 0.15 * heightScale, 0);
    rightPauldron.rotation.z = 0.2;

    chest.add(leftPauldron, rightPauldron);
  }

  // ============================================
  // 4. Arms (Left and Right)
  // ============================================
  // Left Shoulder
  const leftShoulder = new THREE.Group();
  leftShoulder.position.set(0.26 * heightScale, 0.14 * heightScale, 0);
  chest.add(leftShoulder);

  const leftShoulderBall = new THREE.Mesh(jointSphereGeo, jointMat);
  leftShoulder.add(leftShoulderBall);

  const leftUpperArm = new THREE.Mesh(limbGeo, primaryMat);
  leftUpperArm.position.y = -0.18 * heightScale;
  leftUpperArm.castShadow = true;
  leftShoulder.add(leftUpperArm);

  const leftElbow = new THREE.Group();
  leftElbow.position.y = -0.36 * heightScale;
  leftShoulder.add(leftElbow);

  const leftElbowBall = new THREE.Mesh(jointSphereGeo, jointMat);
  leftElbowBall.scale.set(0.8, 0.8, 0.8);
  leftElbow.add(leftElbowBall);

  const leftForearm = new THREE.Mesh(forearmGeo, secondaryMat);
  leftForearm.position.y = -0.16 * heightScale;
  leftForearm.castShadow = true;
  leftElbow.add(leftForearm);

  const leftHand = new THREE.Mesh(
    new THREE.BoxGeometry(0.08 * heightScale, 0.09 * heightScale, 0.06 * heightScale),
    accentMat
  );
  leftHand.position.y = -0.34 * heightScale;
  leftElbow.add(leftHand);

  // Right Shoulder
  const rightShoulder = new THREE.Group();
  rightShoulder.position.set(-0.26 * heightScale, 0.14 * heightScale, 0);
  chest.add(rightShoulder);

  const rightShoulderBall = new THREE.Mesh(jointSphereGeo, jointMat);
  rightShoulder.add(rightShoulderBall);

  const rightUpperArm = new THREE.Mesh(limbGeo, primaryMat);
  rightUpperArm.position.y = -0.18 * heightScale;
  rightUpperArm.castShadow = true;
  rightShoulder.add(rightUpperArm);

  const rightElbow = new THREE.Group();
  rightElbow.position.y = -0.36 * heightScale;
  rightShoulder.add(rightElbow);

  const rightElbowBall = new THREE.Mesh(jointSphereGeo, jointMat);
  rightElbowBall.scale.set(0.8, 0.8, 0.8);
  rightElbow.add(rightElbowBall);

  const rightForearm = new THREE.Mesh(forearmGeo, secondaryMat);
  rightForearm.position.y = -0.16 * heightScale;
  rightForearm.castShadow = true;
  rightElbow.add(rightForearm);

  const rightHand = new THREE.Mesh(
    new THREE.BoxGeometry(0.08 * heightScale, 0.09 * heightScale, 0.06 * heightScale),
    accentMat
  );
  rightHand.position.y = -0.34 * heightScale;
  rightElbow.add(rightHand);

  // ============================================
  // 5. Legs (Left and Right)
  // ============================================
  // Left Hip
  const leftHip = new THREE.Group();
  leftHip.position.set(0.13 * heightScale, -0.08 * heightScale, 0);
  pelvis.add(leftHip);

  const leftHipBall = new THREE.Mesh(jointSphereGeo, jointMat);
  leftHip.add(leftHipBall);

  const leftThigh = new THREE.Mesh(thighGeo, primaryMat);
  leftThigh.position.y = -0.22 * heightScale;
  leftThigh.castShadow = true;
  leftHip.add(leftThigh);

  const leftKnee = new THREE.Group();
  leftKnee.position.y = -0.44 * heightScale;
  leftHip.add(leftKnee);

  const leftKneeBall = new THREE.Mesh(jointSphereGeo, jointMat);
  leftKneeBall.scale.set(0.85, 0.85, 0.85);
  leftKnee.add(leftKneeBall);

  const leftShin = new THREE.Mesh(shinGeo, secondaryMat);
  leftShin.position.y = -0.2 * heightScale;
  leftShin.castShadow = true;
  leftKnee.add(leftShin);

  const leftFoot = new THREE.Mesh(
    new THREE.BoxGeometry(0.1 * heightScale, 0.07 * heightScale, 0.22 * heightScale),
    jointMat
  );
  leftFoot.position.set(0, -0.42 * heightScale, 0.06 * heightScale);
  leftFoot.castShadow = true;
  leftKnee.add(leftFoot);

  // Right Hip
  const rightHip = new THREE.Group();
  rightHip.position.set(-0.13 * heightScale, -0.08 * heightScale, 0);
  pelvis.add(rightHip);

  const rightHipBall = new THREE.Mesh(jointSphereGeo, jointMat);
  rightHip.add(rightHipBall);

  const rightThigh = new THREE.Mesh(thighGeo, primaryMat);
  rightThigh.position.y = -0.22 * heightScale;
  rightThigh.castShadow = true;
  rightHip.add(rightThigh);

  const rightKnee = new THREE.Group();
  rightKnee.position.y = -0.44 * heightScale;
  rightHip.add(rightKnee);

  const rightKneeBall = new THREE.Mesh(jointSphereGeo, jointMat);
  rightKneeBall.scale.set(0.85, 0.85, 0.85);
  rightKnee.add(rightKneeBall);

  const rightShin = new THREE.Mesh(shinGeo, secondaryMat);
  rightShin.position.y = -0.2 * heightScale;
  rightShin.castShadow = true;
  rightKnee.add(rightShin);

  const rightFoot = new THREE.Mesh(
    new THREE.BoxGeometry(0.1 * heightScale, 0.07 * heightScale, 0.22 * heightScale),
    jointMat
  );
  rightFoot.position.set(0, -0.42 * heightScale, 0.06 * heightScale);
  rightFoot.castShadow = true;
  rightKnee.add(rightFoot);

  return {
    root,
    type,
    pelvis,
    chest,
    head,
    visor,
    core,
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftWing,
    rightWing,
    materials: {
      primary: primaryMat,
      secondary: secondaryMat,
      accent: accentMat,
      joint: jointMat,
      glow: glowMat,
      wing: wingMat,
    },
  };
}

/**
 * Procedural running / flight animation for minion mannequin with Role & Evasion Dynamics
 */
export function animateMinionMannequin(
  rig: MannequinRig,
  walkCycle: number,
  leanAngle: number,
  animScale: number,
  priority: 'MULTIPLICATION' | 'ATTACK' = 'MULTIPLICATION',
  isAirborne: boolean = true,
  pitchAngle: number = 0,
  rollAngle: number = 0,
  wingPhase: number = 0,
  role: 'SPAWNER' | 'KILLER' = 'KILLER',
  isEvading: boolean = false
) {
  const isSpawner = role === 'SPAWNER';
  const isAttack = role === 'KILLER' || priority === 'ATTACK';

  if (isAirborne) {
    // ==========================================
    // 3D AERIAL FLIGHT LOCOMOTION & POSTURE
    // ==========================================
    // 1. Dynamic Aerodynamic Wing Beats
    if (rig.leftWing && rig.rightWing) {
      const beatIntensity = isEvading ? 0.75 : isAttack ? 0.55 : 0.4;
      const wingBeat = Math.sin(wingPhase * (isEvading ? 1.4 : 1.0)) * beatIntensity;
      const wingFlapZ = Math.cos(wingPhase) * 0.25;

      rig.leftWing.rotation.y = 0.25 + wingBeat;
      rig.rightWing.rotation.y = -0.25 - wingBeat;
      rig.leftWing.rotation.z = 0.2 + wingFlapZ + (isEvading ? 0.15 : 0);
      rig.rightWing.rotation.z = -0.2 - wingFlapZ - (isEvading ? 0.15 : 0);
      rig.leftWing.visible = true;
      rig.rightWing.visible = true;
    }

    // 2. Torso Pitch & Aerodynamic Banking Roll
    const baseDiveLean = isAttack ? 0.75 : isEvading ? 0.35 : 0.45;
    rig.chest.rotation.x = baseDiveLean + pitchAngle * 0.7; // forward dive pitch
    rig.chest.rotation.z = -rollAngle * (isEvading ? 0.9 : 0.6); // banking into turn
    rig.head.rotation.x = -0.35 - pitchAngle * 0.4; // head looks ahead along flight trajectory
    rig.head.rotation.z = rollAngle * 0.25;

    // 3. Arms Streamlined for Aerodynamic Gliding / Steering
    const armFlutter = Math.sin(wingPhase * 0.5) * (isEvading ? 0.15 : 0.08);
    rig.leftShoulder.rotation.x = -0.65 + armFlutter;
    rig.rightShoulder.rotation.x = -0.65 + armFlutter;
    rig.leftShoulder.rotation.z = isEvading ? 0.55 : 0.35;
    rig.rightShoulder.rotation.z = isEvading ? -0.55 : -0.35;
    rig.leftElbow.rotation.x = -0.35;
    rig.rightElbow.rotation.x = -0.35;

    // 4. Legs Streamlined Trailing Behind
    const legFlutter = Math.cos(wingPhase * 0.5) * 0.06;
    rig.leftHip.rotation.x = -0.35 + legFlutter;
    rig.rightHip.rotation.x = -0.35 - legFlutter;
    rig.leftHip.rotation.z = 0.06;
    rig.rightHip.rotation.z = -0.06;
    rig.leftKnee.rotation.x = 0.45;
    rig.rightKnee.rotation.x = 0.45;
  } else {
    // ==========================================
    // GROUND WALKING / RUNNING POSE
    // ==========================================
    if (rig.leftWing && rig.rightWing) {
      rig.leftWing.rotation.set(0, 0.1, 0.15);
      rig.rightWing.rotation.set(0, -0.1, -0.15);
    }

    const stride = Math.sin(walkCycle);
    const kneeBendL = Math.max(0, Math.sin(walkCycle));
    const kneeBendR = Math.max(0, -Math.sin(walkCycle));
    const stridePower = isAttack ? 0.85 : isEvading ? 0.95 : 0.65;

    // Leg strides
    rig.leftHip.rotation.x = stride * stridePower;
    rig.rightHip.rotation.x = -stride * stridePower;
    rig.leftHip.rotation.z = 0;
    rig.rightHip.rotation.z = 0;

    // Knee bending during backstroke
    rig.leftKnee.rotation.x = kneeBendL * (isAttack ? 1.0 : 0.8);
    rig.rightKnee.rotation.x = kneeBendR * (isAttack ? 1.0 : 0.8);

    // Arm swings in opposition to legs
    rig.leftShoulder.rotation.x = -stride * (isAttack ? 0.9 : 0.65);
    rig.rightShoulder.rotation.x = stride * (isAttack ? 0.9 : 0.65);
    rig.leftShoulder.rotation.z = 0;
    rig.rightShoulder.rotation.z = 0;

    // Forearm flex
    rig.leftElbow.rotation.x = -0.5 - Math.abs(stride) * (isAttack ? 0.5 : 0.3);
    rig.rightElbow.rotation.x = -0.5 - Math.abs(stride) * (isAttack ? 0.5 : 0.3);

    // Torso forward sprint lean + curved banking tilt
    rig.chest.rotation.x = isAttack ? 0.3 : isEvading ? 0.22 : 0.14;
    rig.chest.rotation.z = -leanAngle * 0.4;
    rig.head.rotation.x = 0;
    rig.head.rotation.z = leanAngle * 0.2;
  }

  // Scale pop-in
  rig.root.scale.set(animScale, animScale, animScale);

  // Visual differentiation by Role (Spawner = Luminous Amber/Gold, Killer = Aggressive Fiery Crimson)
  if (isSpawner) {
    if (isEvading) {
      // Actively Evading: Radiant pulsing topaz gold with overdrive emission
      rig.materials.primary.color.setHex(0xfacc15);
      rig.materials.glow.color.setHex(0xfef08a);
      rig.materials.glow.emissive.setHex(0xeab308);
      rig.materials.glow.emissiveIntensity = 2.4;
      if (rig.materials.wing) {
        rig.materials.wing.color.setHex(0xfde047);
        rig.materials.wing.emissive.setHex(0xeab308);
        rig.materials.wing.emissiveIntensity = 2.2;
      }
    } else {
      // Peaceful Incubator Spawner: Warm golden aura
      rig.materials.primary.color.setHex(0xeab308);
      rig.materials.glow.color.setHex(0xfde047);
      rig.materials.glow.emissive.setHex(0xca8a04);
      rig.materials.glow.emissiveIntensity = 1.3;
      if (rig.materials.wing) {
        rig.materials.wing.color.setHex(0xfacc15);
        rig.materials.wing.emissive.setHex(0xca8a04);
        rig.materials.wing.emissiveIntensity = 1.4;
      }
    }
  } else {
    // Killer Minion: Aggressive blazing crimson & predatory red glow
    rig.materials.primary.color.setHex(0xdc2626);
    rig.materials.glow.color.setHex(0xf87171);
    rig.materials.glow.emissive.setHex(0xb91c1c);
    rig.materials.glow.emissiveIntensity = 1.7;
    if (rig.materials.wing) {
      rig.materials.wing.color.setHex(0xef4444);
      rig.materials.wing.emissive.setHex(0xdc2626);
      rig.materials.wing.emissiveIntensity = 1.9;
    }
  }
}

/**
 * Procedural guardian / repel / hunter animation for Party A
 */
export function animatePartyAMannequin(
  rig: MannequinRig,
  time: number,
  flashTimer: number,
  repelCooldown: number,
  isMoving: boolean = false,
  walkCycle: number = 0
) {
  const isRepelling = flashTimer > 0;
  const breathe = Math.sin(time * 2.5) * 0.05;

  if (isRepelling) {
    // Powerful blast posture: Chest raised, arms thrust outward
    rig.chest.rotation.x = -0.15;
    rig.leftShoulder.rotation.set(-0.3, 0.4, 0.9);
    rig.rightShoulder.rotation.set(-0.3, -0.4, -0.9);
    rig.leftElbow.rotation.x = -0.1;
    rig.rightElbow.rotation.x = -0.1;

    // Glowing intense cyan
    rig.materials.primary.color.setHex(0x06b6d4);
    rig.materials.primary.emissive.setHex(0x22d3ee);
    rig.materials.primary.emissiveIntensity = 1.0;
    rig.materials.glow.emissiveIntensity = 2.0;
  } else if (isMoving) {
    // Active Hunter Locomotion Stride: Tracking & pursuing spawners
    const stride = Math.sin(walkCycle);
    const kneeL = Math.max(0, Math.sin(walkCycle));
    const kneeR = Math.max(0, -Math.sin(walkCycle));

    rig.chest.rotation.x = 0.18; // forward hunt lean
    rig.chest.rotation.y = Math.sin(walkCycle * 0.5) * 0.08;

    // Arms pumping aggressively in hunt stride
    rig.leftShoulder.rotation.set(-stride * 0.8, 0.1, 0.2);
    rig.rightShoulder.rotation.set(stride * 0.8, -0.1, -0.2);
    rig.leftElbow.rotation.x = -0.8 - Math.abs(stride) * 0.4;
    rig.rightElbow.rotation.x = -0.8 - Math.abs(stride) * 0.4;

    // Dynamic legs
    rig.leftHip.rotation.x = stride * 0.75;
    rig.rightHip.rotation.x = -stride * 0.75;
    rig.leftKnee.rotation.x = kneeL * 0.85;
    rig.rightKnee.rotation.x = kneeR * 0.85;

    // Ready energized glow
    rig.materials.primary.color.setHex(0x2563eb);
    rig.materials.primary.emissive.setHex(0x3b82f6);
    rig.materials.primary.emissiveIntensity = 0.6;
    rig.materials.glow.emissiveIntensity = 1.6;
  } else {
    // Defensive Guardian floating stance
    rig.chest.rotation.x = 0.03 + breathe;
    rig.chest.rotation.y = Math.sin(time * 1.2) * 0.08;

    // Arms in ready defensive posture
    rig.leftShoulder.rotation.set(-0.4 + breathe, 0.2, 0.35);
    rig.rightShoulder.rotation.set(-0.4 + breathe, -0.2, -0.35);
    rig.leftElbow.rotation.x = -0.8;
    rig.rightElbow.rotation.x = -0.8;

    // Legs gently trailing in hover stance
    rig.leftHip.rotation.x = 0.15 + breathe * 0.5;
    rig.rightHip.rotation.x = 0.25 - breathe * 0.5;
    rig.leftKnee.rotation.x = 0.3;
    rig.rightKnee.rotation.x = 0.4;

    // Normal Cobalt Blue with ready glow
    rig.materials.primary.color.setHex(0x2563eb);
    rig.materials.primary.emissive.setHex(0x1d4ed8);
    rig.materials.primary.emissiveIntensity = repelCooldown <= 0 ? 0.45 : 0.2;
    rig.materials.glow.emissiveIntensity = repelCooldown <= 0 ? 1.4 : 0.8;
  }
}

/**
 * Procedural summoner ritual animation for Party B
 */
export function animatePartyBMannequin(
  rig: MannequinRig,
  time: number,
  summonTimer: number,
  isSummoning: boolean
) {
  const breathe = Math.sin(time * 3.0) * 0.06;
  const summonProgress = Math.sin(time * 6.0);

  if (isSummoning) {
    // Summoning cast posture: Right hand raised invoking energy, left hand channeling downward
    rig.chest.rotation.x = -0.05;
    rig.chest.rotation.y = -0.15;

    // Right arm high invocation
    rig.rightShoulder.rotation.set(-1.4 + breathe, -0.3, -0.4 + summonProgress * 0.1);
    rig.rightElbow.rotation.x = -0.6;

    // Left arm lower stabilization
    rig.leftShoulder.rotation.set(-0.3 + breathe, 0.4, 0.5);
    rig.leftElbow.rotation.x = -0.9;

    // Head tilted up towards sky/portal
    rig.head.rotation.x = -0.2;
    rig.head.rotation.y = 0.1;

    // Legs planted firmly in ritual stance
    rig.leftHip.rotation.set(0.1, 0, 0.2);
    rig.rightHip.rotation.set(-0.15, 0, -0.2);
    rig.leftKnee.rotation.x = 0.25;
    rig.rightKnee.rotation.x = 0.35;

    // Pulsing fiery emissive
    const pulse = 0.6 + Math.sin(time * 8.0) * 0.4;
    rig.materials.primary.emissiveIntensity = 0.4 * pulse;
    rig.materials.glow.emissiveIntensity = 1.0 + pulse * 0.8;
  } else {
    // Idle dark summoner stance
    rig.chest.rotation.x = 0.05;
    rig.rightShoulder.rotation.set(-0.5, -0.2, -0.3);
    rig.rightElbow.rotation.x = -0.7;
    rig.leftShoulder.rotation.set(-0.5, 0.2, 0.3);
    rig.leftElbow.rotation.x = -0.7;

    rig.head.rotation.set(0.1, 0, 0);

    rig.materials.primary.emissiveIntensity = 0.25;
    rig.materials.glow.emissiveIntensity = 0.8;
  }
}

/**
 * Procedural animation for Red Survivor (Single Player Agent)
 */
export function animateRedSurvivorMannequin(
  rig: MannequinRig,
  time: number,
  flashTimer: number,
  damageFlashTimer: number,
  isDashing: boolean,
  isMoving: boolean = false,
  walkCycle: number = 0
) {
  const isRepelling = flashTimer > 0;
  const isDamaged = damageFlashTimer > 0;
  const breathe = Math.sin(time * 3.2) * 0.04;

  if (isDamaged) {
    // Flinch reaction
    rig.chest.rotation.x = -0.25;
    rig.head.rotation.x = 0.2;
    rig.leftShoulder.rotation.set(-0.6, 0.4, 0.4);
    rig.rightShoulder.rotation.set(-0.6, -0.4, -0.4);
    rig.materials.primary.color.setHex(0xffffff);
    rig.materials.primary.emissive.setHex(0xffffff);
    rig.materials.primary.emissiveIntensity = 1.5;
  } else if (isRepelling) {
    // Kinetic blast thrust: Chest flared, hands thrust outward
    rig.chest.rotation.x = -0.18;
    rig.leftShoulder.rotation.set(-0.4, 0.5, 0.9);
    rig.rightShoulder.rotation.set(-0.4, -0.5, -0.9);
    rig.leftElbow.rotation.x = -0.1;
    rig.rightElbow.rotation.x = -0.1;

    // Glowing hyper-red core & visor
    rig.materials.primary.color.setHex(0xef4444);
    rig.materials.primary.emissive.setHex(0xf87171);
    rig.materials.primary.emissiveIntensity = 1.2;
    rig.materials.glow.emissiveIntensity = 2.2;
  } else if (isDashing) {
    // Aerodynamic sprint thrust
    rig.chest.rotation.x = 0.45;
    rig.head.rotation.x = -0.2;
    rig.leftShoulder.rotation.set(0.8, 0.2, 0.2);
    rig.rightShoulder.rotation.set(0.8, -0.2, -0.2);
    rig.leftElbow.rotation.x = -1.2;
    rig.rightElbow.rotation.x = -1.2;

    rig.leftHip.rotation.x = 0.6;
    rig.rightHip.rotation.x = -0.6;
    rig.leftKnee.rotation.x = 0.8;
    rig.rightKnee.rotation.x = 0.8;

    rig.materials.primary.color.setHex(0xdc2626);
    rig.materials.primary.emissive.setHex(0xf87171);
    rig.materials.primary.emissiveIntensity = 0.9;
  } else if (isMoving) {
    // Tactical Survival Stride
    const stride = Math.sin(walkCycle);
    const kneeL = Math.max(0, Math.sin(walkCycle));
    const kneeR = Math.max(0, -Math.sin(walkCycle));

    rig.chest.rotation.x = 0.15;
    rig.chest.rotation.y = Math.sin(walkCycle * 0.5) * 0.08;

    rig.leftShoulder.rotation.set(-stride * 0.75, 0.1, 0.2);
    rig.rightShoulder.rotation.set(stride * 0.75, -0.1, -0.2);
    rig.leftElbow.rotation.x = -0.7 - Math.abs(stride) * 0.3;
    rig.rightElbow.rotation.x = -0.7 - Math.abs(stride) * 0.3;

    rig.leftHip.rotation.x = stride * 0.7;
    rig.rightHip.rotation.x = -stride * 0.7;
    rig.leftKnee.rotation.x = kneeL * 0.8;
    rig.rightKnee.rotation.x = kneeR * 0.8;

    rig.materials.primary.color.setHex(0xdc2626);
    rig.materials.primary.emissive.setHex(0xb91c1c);
    rig.materials.primary.emissiveIntensity = 0.45;
    rig.materials.glow.emissiveIntensity = 1.3;
  } else {
    // Alert Combat Stance
    rig.chest.rotation.x = 0.04 + breathe;
    rig.chest.rotation.y = Math.sin(time * 1.5) * 0.06;

    rig.leftShoulder.rotation.set(-0.35 + breathe, 0.2, 0.3);
    rig.rightShoulder.rotation.set(-0.35 + breathe, -0.2, -0.3);
    rig.leftElbow.rotation.x = -0.75;
    rig.rightElbow.rotation.x = -0.75;

    rig.leftHip.rotation.x = 0.1 + breathe * 0.4;
    rig.rightHip.rotation.x = 0.18 - breathe * 0.4;
    rig.leftKnee.rotation.x = 0.2;
    rig.rightKnee.rotation.x = 0.3;

    rig.materials.primary.color.setHex(0xdc2626);
    rig.materials.primary.emissive.setHex(0x991b1b);
    rig.materials.primary.emissiveIntensity = 0.3;
    rig.materials.glow.emissiveIntensity = 1.0;
  }
}

