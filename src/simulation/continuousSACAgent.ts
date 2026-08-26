import {
  CurriculumStage,
  EmergentBehaviorProfile,
  EmergentTactic,
  SACActionVector,
  SACMetrics,
  SACRewardVariant,
  SACTransition,
} from '../types';

/**
 * Fast, zero-GC Layer Normalization module for stabilizing deep Actor-Critic training
 */
export class LayerNorm {
  public dim: number;
  public gamma: Float32Array;
  public beta: Float32Array;
  public dGamma: Float32Array;
  public dBeta: Float32Array;
  public mGamma: Float32Array;
  public vGamma: Float32Array;
  public mBeta: Float32Array;
  public vBeta: Float32Array;
  public xHat: Float32Array;
  public mean: number = 0;
  public variance: number = 1;
  public invStd: number = 1;

  constructor(dim: number) {
    this.dim = dim;
    this.gamma = new Float32Array(dim);
    this.gamma.fill(1.0);
    this.beta = new Float32Array(dim);
    this.beta.fill(0.0);
    this.dGamma = new Float32Array(dim);
    this.dBeta = new Float32Array(dim);
    this.mGamma = new Float32Array(dim);
    this.vGamma = new Float32Array(dim);
    this.mBeta = new Float32Array(dim);
    this.vBeta = new Float32Array(dim);
    this.xHat = new Float32Array(dim);
  }

  public forward(input: Float32Array, output: Float32Array) {
    const d = this.dim;
    let sum = 0;
    for (let i = 0; i < d; i++) sum += input[i];
    const mean = sum / d;
    this.mean = mean;

    let varSum = 0;
    for (let i = 0; i < d; i++) {
      const diff = input[i] - mean;
      varSum += diff * diff;
    }
    const variance = varSum / d;
    this.variance = variance;
    const invStd = 1.0 / Math.sqrt(variance + 1e-5);
    this.invStd = invStd;

    for (let i = 0; i < d; i++) {
      const xh = (input[i] - mean) * invStd;
      this.xHat[i] = xh;
      output[i] = this.gamma[i] * xh + this.beta[i];
    }
  }

  public backward(dOutput: Float32Array, dInput: Float32Array) {
    const d = this.dim;
    const invStd = this.invStd;
    let sumDOut = 0;
    let sumDOutXHat = 0;

    for (let i = 0; i < d; i++) {
      const dOut = dOutput[i];
      this.dGamma[i] += dOut * this.xHat[i];
      this.dBeta[i] += dOut;

      const gammaDOut = this.gamma[i] * dOut;
      sumDOut += gammaDOut;
      sumDOutXHat += gammaDOut * this.xHat[i];
    }

    const invD = 1.0 / d;
    for (let i = 0; i < d; i++) {
      const gammaDOut = this.gamma[i] * dOutput[i];
      dInput[i] = invStd * (gammaDOut - invD * sumDOut - this.xHat[i] * invD * sumDOutXHat);
    }
  }

  public stepAdam(lr: number, t: number) {
    const fix1 = 1.0 - Math.pow(0.9, t);
    const fix2 = 1.0 - Math.pow(0.999, t);

    for (let i = 0; i < this.dim; i++) {
      const gGrad = Math.max(-5.0, Math.min(5.0, this.dGamma[i]));
      this.mGamma[i] = 0.9 * this.mGamma[i] + 0.1 * gGrad;
      this.vGamma[i] = 0.999 * this.vGamma[i] + 0.001 * (gGrad * gGrad);
      this.gamma[i] -= (lr * (this.mGamma[i] / fix1)) / (Math.sqrt(this.vGamma[i] / fix2) + 1e-8);
      this.dGamma[i] = 0;

      const bGrad = Math.max(-5.0, Math.min(5.0, this.dBeta[i]));
      this.mBeta[i] = 0.9 * this.mBeta[i] + 0.1 * bGrad;
      this.vBeta[i] = 0.999 * this.vBeta[i] + 0.001 * (bGrad * bGrad);
      this.beta[i] -= (lr * (this.mBeta[i] / fix1)) / (Math.sqrt(this.vBeta[i] / fix2) + 1e-8);
      this.dBeta[i] = 0;
    }
  }

  public copyFrom(source: LayerNorm) {
    this.gamma.set(source.gamma);
    this.beta.set(source.beta);
  }

  public polyakUpdate(source: LayerNorm, tau: number) {
    for (let i = 0; i < this.dim; i++) {
      this.gamma[i] = (1.0 - tau) * this.gamma[i] + tau * source.gamma[i];
      this.beta[i] = (1.0 - tau) * this.beta[i] + tau * source.beta[i];
    }
  }

  public toJSON() {
    return {
      dim: this.dim,
      gamma: Array.from(this.gamma),
      beta: Array.from(this.beta),
    };
  }

  public fromJSON(data: any) {
    if (data?.gamma && data.gamma.length === this.dim) this.gamma.set(data.gamma);
    if (data?.beta && data.beta.length === this.dim) this.beta.set(data.beta);
  }
}

/**
 * Lightweight, zero-GC matrix & neural layer utilities for real-time WebGL SAC training
 */
class DenseLayer {
  public inDim: number;
  public outDim: number;
  public weights: Float32Array; // outDim x inDim (flattened)
  public bias: Float32Array; // outDim
  // Gradients
  public dWeights: Float32Array;
  public dBias: Float32Array;
  // Adam optimizer state
  public mWeights: Float32Array;
  public vWeights: Float32Array;
  public mBias: Float32Array;
  public vBias: Float32Array;

  constructor(inDim: number, outDim: number, initScale: number = 0.1) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.weights = new Float32Array(outDim * inDim);
    this.bias = new Float32Array(outDim);
    this.dWeights = new Float32Array(outDim * inDim);
    this.dBias = new Float32Array(outDim);

    this.mWeights = new Float32Array(outDim * inDim);
    this.vWeights = new Float32Array(outDim * inDim);
    this.mBias = new Float32Array(outDim);
    this.vBias = new Float32Array(outDim);

    // He / Xavier Initialization
    const std = Math.sqrt(2.0 / (inDim + outDim)) * initScale;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() * 2 - 1) * std;
    }
    for (let i = 0; i < this.bias.length; i++) {
      this.bias[i] = 0.0;
    }
  }

  public forward(input: Float32Array, output: Float32Array) {
    const outD = this.outDim;
    const inD = this.inDim;
    const w = this.weights;
    const b = this.bias;

    for (let i = 0; i < outD; i++) {
      let sum = b[i];
      const rowOffset = i * inD;
      for (let j = 0; j < inD; j++) {
        sum += w[rowOffset + j] * input[j];
      }
      output[i] = sum;
    }
  }

  public backward(input: Float32Array, dOutput: Float32Array, dInput: Float32Array | null) {
    const outD = this.outDim;
    const inD = this.inDim;
    const w = this.weights;
    const dW = this.dWeights;
    const dB = this.dBias;

    for (let i = 0; i < outD; i++) {
      const gradOut = dOutput[i];
      dB[i] += gradOut;
      const rowOffset = i * inD;
      for (let j = 0; j < inD; j++) {
        dW[rowOffset + j] += gradOut * input[j];
      }
    }

    if (dInput) {
      for (let j = 0; j < inD; j++) {
        let sum = 0;
        for (let i = 0; i < outD; i++) {
          sum += w[i * inD + j] * dOutput[i];
        }
        dInput[j] = sum;
      }
    }
  }

  public stepAdam(lr: number, beta1: number = 0.9, beta2: number = 0.999, eps: number = 1e-8, t: number = 1) {
    const fix1 = 1.0 - Math.pow(beta1, t);
    const fix2 = 1.0 - Math.pow(beta2, t);

    // Weights
    for (let i = 0; i < this.weights.length; i++) {
      const grad = Math.max(-5.0, Math.min(5.0, this.dWeights[i]));
      this.mWeights[i] = beta1 * this.mWeights[i] + (1.0 - beta1) * grad;
      this.vWeights[i] = beta2 * this.vWeights[i] + (1.0 - beta2) * (grad * grad);

      const mHat = this.mWeights[i] / fix1;
      const vHat = this.vWeights[i] / fix2;
      this.weights[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      this.dWeights[i] = 0;
    }

    // Bias
    for (let i = 0; i < this.bias.length; i++) {
      const grad = Math.max(-5.0, Math.min(5.0, this.dBias[i]));
      this.mBias[i] = beta1 * this.mBias[i] + (1.0 - beta1) * grad;
      this.vBias[i] = beta2 * this.vBias[i] + (1.0 - beta2) * (grad * grad);

      const mHat = this.mBias[i] / fix1;
      const vHat = this.vBias[i] / fix2;
      this.bias[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      this.dBias[i] = 0;
    }
  }

  public copyWeightsFrom(source: DenseLayer) {
    this.weights.set(source.weights);
    this.bias.set(source.bias);
  }

  public polyakUpdate(source: DenseLayer, tau: number) {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (1.0 - tau) * this.weights[i] + tau * source.weights[i];
    }
    for (let i = 0; i < this.bias.length; i++) {
      this.bias[i] = (1.0 - tau) * this.bias[i] + tau * source.bias[i];
    }
  }

  public toJSON(): { inDim: number; outDim: number; weights: number[]; bias: number[] } {
    return {
      inDim: this.inDim,
      outDim: this.outDim,
      weights: Array.from(this.weights),
      bias: Array.from(this.bias),
    };
  }

  public fromJSON(data: { inDim?: number; outDim?: number; weights: number[]; bias: number[] }) {
    if (data.weights && data.weights.length === this.weights.length) {
      this.weights.set(data.weights);
    }
    if (data.bias && data.bias.length === this.bias.length) {
      this.bias.set(data.bias);
    }
  }
}

/**
 * Multi-Layer Perceptron (MLP) Network for SAC Actor & Critic with Layer Normalization & LeakyReLU
 */
class MLP {
  public l1: DenseLayer;
  public ln1: LayerNorm;
  public l2: DenseLayer;
  public ln2: LayerNorm;
  public lOut: DenseLayer;

  // Activation buffers for zero GC
  public z1: Float32Array; // pre-norm / pre-act
  public h1: Float32Array; // post-norm + act
  public z2: Float32Array;
  public h2: Float32Array;
  public out: Float32Array;

  // Gradient buffers
  public dh1: Float32Array;
  public dz1: Float32Array;
  public dh2: Float32Array;
  public dz2: Float32Array;
  public dOut: Float32Array;

  public inDim: number;
  public hiddenDim: number;
  public outDim: number;
  public useLayerNorm: boolean;

  constructor(
    inDim: number,
    hiddenDim: number,
    outDim: number,
    initScale: number = 1.0,
    useLayerNorm: boolean = true
  ) {
    this.inDim = inDim;
    this.hiddenDim = hiddenDim;
    this.outDim = outDim;
    this.useLayerNorm = useLayerNorm;

    this.l1 = new DenseLayer(inDim, hiddenDim, 1.0);
    this.ln1 = new LayerNorm(hiddenDim);
    this.l2 = new DenseLayer(hiddenDim, hiddenDim, 1.0);
    this.ln2 = new LayerNorm(hiddenDim);
    this.lOut = new DenseLayer(hiddenDim, outDim, initScale);

    this.z1 = new Float32Array(hiddenDim);
    this.h1 = new Float32Array(hiddenDim);
    this.z2 = new Float32Array(hiddenDim);
    this.h2 = new Float32Array(hiddenDim);
    this.out = new Float32Array(outDim);

    this.dh1 = new Float32Array(hiddenDim);
    this.dz1 = new Float32Array(hiddenDim);
    this.dh2 = new Float32Array(hiddenDim);
    this.dz2 = new Float32Array(hiddenDim);
    this.dOut = new Float32Array(outDim);
  }

  public forward(input: Float32Array): Float32Array {
    this.l1.forward(input, this.z1);
    if (this.useLayerNorm) {
      this.ln1.forward(this.z1, this.h1);
    } else {
      this.h1.set(this.z1);
    }
    // LeakyReLU (alpha = 0.02)
    for (let i = 0; i < this.h1.length; i++) {
      if (this.h1[i] < 0) this.h1[i] *= 0.02;
    }

    this.l2.forward(this.h1, this.z2);
    if (this.useLayerNorm) {
      this.ln2.forward(this.z2, this.h2);
    } else {
      this.h2.set(this.z2);
    }
    // LeakyReLU (alpha = 0.02)
    for (let i = 0; i < this.h2.length; i++) {
      if (this.h2[i] < 0) this.h2[i] *= 0.02;
    }

    this.lOut.forward(this.h2, this.out);
    return this.out;
  }

  public backward(input: Float32Array, dLoss: Float32Array, dInput: Float32Array | null) {
    this.dOut.set(dLoss);

    // Backward output layer
    this.lOut.backward(this.h2, this.dOut, this.dh2);

    // LeakyReLU gradient on h2
    for (let i = 0; i < this.h2.length; i++) {
      if (this.h2[i] <= 0) this.dh2[i] *= 0.02;
    }

    // Backward LayerNorm 2
    if (this.useLayerNorm) {
      this.ln2.backward(this.dh2, this.dz2);
    } else {
      this.dz2.set(this.dh2);
    }

    // Backward layer 2
    this.l2.backward(this.h1, this.dz2, this.dh1);

    // LeakyReLU gradient on h1
    for (let i = 0; i < this.h1.length; i++) {
      if (this.h1[i] <= 0) this.dh1[i] *= 0.02;
    }

    // Backward LayerNorm 1
    if (this.useLayerNorm) {
      this.ln1.backward(this.dh1, this.dz1);
    } else {
      this.dz1.set(this.dh1);
    }

    // Backward layer 1
    this.l1.backward(input, this.dz1, dInput);
  }

  public computeActionGradient(
    input: Float32Array,
    obsDim: number,
    actionDim: number,
    gradActionOut: Float32Array
  ) {
    const hiddenDim = this.h1.length;
    const inDim = this.l1.inDim;
    const wOut = this.lOut.weights;
    const w2 = this.l2.weights;
    const w1 = this.l1.weights;

    // Backprop to h2
    for (let k = 0; k < hiddenDim; k++) {
      const actGrad = this.h2[k] > 0 ? 1.0 : 0.02;
      this.dh2[k] = wOut[k] * actGrad;
    }

    if (this.useLayerNorm) {
      this.ln2.backward(this.dh2, this.dz2);
    } else {
      this.dz2.set(this.dh2);
    }

    // Backprop to h1
    for (let m = 0; m < hiddenDim; m++) {
      let sum = 0;
      for (let k = 0; k < hiddenDim; k++) {
        sum += w2[k * hiddenDim + m] * this.dz2[k];
      }
      const actGrad1 = this.h1[m] > 0 ? 1.0 : 0.02;
      this.dh1[m] = sum * actGrad1;
    }

    if (this.useLayerNorm) {
      this.ln1.backward(this.dh1, this.dz1);
    } else {
      this.dz1.set(this.dh1);
    }

    // Backprop to input action slice [obsDim .. obsDim + actionDim]
    for (let j = 0; j < actionDim; j++) {
      let sum = 0;
      const inputIdx = obsDim + j;
      for (let m = 0; m < hiddenDim; m++) {
        sum += w1[m * inDim + inputIdx] * this.dz1[m];
      }
      gradActionOut[j] = sum;
    }
  }

  public stepAdam(lr: number, t: number) {
    this.l1.stepAdam(lr, 0.9, 0.999, 1e-8, t);
    if (this.useLayerNorm) this.ln1.stepAdam(lr, t);
    this.l2.stepAdam(lr, 0.9, 0.999, 1e-8, t);
    if (this.useLayerNorm) this.ln2.stepAdam(lr, t);
    this.lOut.stepAdam(lr, 0.9, 0.999, 1e-8, t);
  }

  public copyFrom(source: MLP) {
    this.l1.copyWeightsFrom(source.l1);
    if (this.useLayerNorm && source.ln1) this.ln1.copyFrom(source.ln1);
    this.l2.copyWeightsFrom(source.l2);
    if (this.useLayerNorm && source.ln2) this.ln2.copyFrom(source.ln2);
    this.lOut.copyWeightsFrom(source.lOut);
  }

  public polyakUpdate(source: MLP, tau: number) {
    this.l1.polyakUpdate(source.l1, tau);
    if (this.useLayerNorm && source.ln1) this.ln1.polyakUpdate(source.ln1, tau);
    this.l2.polyakUpdate(source.l2, tau);
    if (this.useLayerNorm && source.ln2) this.ln2.polyakUpdate(source.ln2, tau);
    this.lOut.polyakUpdate(source.lOut, tau);
  }

  public toJSON() {
    return {
      inDim: this.inDim,
      hiddenDim: this.hiddenDim,
      outDim: this.outDim,
      useLayerNorm: this.useLayerNorm,
      l1: this.l1.toJSON(),
      ln1: this.ln1?.toJSON(),
      l2: this.l2.toJSON(),
      ln2: this.ln2?.toJSON(),
      lOut: this.lOut.toJSON(),
    };
  }

  public fromJSON(data: any) {
    if (!data) return;
    if (data.l1) this.l1.fromJSON(data.l1);
    if (data.ln1 && this.ln1) this.ln1.fromJSON(data.ln1);
    if (data.l2) this.l2.fromJSON(data.l2);
    if (data.ln2 && this.ln2) this.ln2.fromJSON(data.ln2);
    if (data.lOut) this.lOut.fromJSON(data.lOut);
  }
}

/**
 * Standard Normal Gaussian Random Sampler (Box-Muller Transform)
 */
function sampleStandardNormal(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Binary Sum-Tree Data Structure for O(log N) Proportional Prioritized Experience Replay
 */
export class SumTree {
  public capacity: number;
  public tree: Float32Array; // Size: 2 * capacity
  public dataPointer: number = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.tree = new Float32Array(2 * capacity);
  }

  public update(treeIndex: number, priority: number) {
    if (treeIndex < 0 || treeIndex >= this.tree.length) return;
    const change = priority - this.tree[treeIndex];
    this.tree[treeIndex] = priority;

    let parent = Math.floor((treeIndex - 1) / 2);
    while (parent >= 0) {
      this.tree[parent] += change;
      if (parent === 0) break;
      parent = Math.floor((parent - 1) / 2);
    }
  }

  public add(priority: number): number {
    const treeIndex = this.dataPointer + this.capacity - 1;
    this.update(treeIndex, priority);
    const currPointer = treeIndex;
    this.dataPointer = (this.dataPointer + 1) % this.capacity;
    return currPointer;
  }

  public get(value: number): { treeIndex: number; priority: number; dataIndex: number } {
    let parentIndex = 0;
    while (true) {
      const leftChildIndex = 2 * parentIndex + 1;
      const rightChildIndex = leftChildIndex + 1;

      if (leftChildIndex >= this.tree.length) {
        break;
      }

      if (value <= this.tree[leftChildIndex]) {
        parentIndex = leftChildIndex;
      } else {
        value -= this.tree[leftChildIndex];
        if (rightChildIndex < this.tree.length && this.tree[rightChildIndex] > 0) {
          parentIndex = rightChildIndex;
        } else {
          parentIndex = leftChildIndex;
        }
      }
    }

    const dataIndex = parentIndex - this.capacity + 1;
    return {
      treeIndex: parentIndex,
      priority: this.tree[parentIndex],
      dataIndex: Math.max(0, Math.min(this.capacity - 1, dataIndex)),
    };
  }

  public get totalPriority(): number {
    return Math.max(1e-5, this.tree[0]);
  }

  public clear() {
    this.tree.fill(0);
    this.dataPointer = 0;
  }
}

/**
 * Prioritized & Multi-Step Experience Replay Buffer (PER)
 */
export class ReplayBuffer {
  public capacity: number;
  public buffer: SACTransition[];
  public index: number = 0;
  public size: number = 0;
  public tree: SumTree;
  public maxPriority: number = 1.0;
  public alpha: number = 0.6; // Priority exponent alpha
  public eps: number = 1e-5;

  constructor(capacity: number = 50000, alpha: number = 0.6) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
    this.tree = new SumTree(capacity);
    this.alpha = alpha;
  }

  public add(
    state: Float32Array,
    action: Float32Array,
    reward: number,
    nextState: Float32Array,
    done: boolean,
    gammaN: number = 0.99,
    nStep: number = 1
  ): SACTransition {
    const priority = Math.pow(this.maxPriority, this.alpha);
    const treeIndex = this.tree.add(priority);
    const dataIndex = this.index;

    const transition: SACTransition = {
      state: new Float32Array(state),
      action: new Float32Array(action),
      reward,
      nextState: new Float32Array(nextState),
      done,
      gammaN,
      nStep,
      priority: this.maxPriority,
      treeIndex,
    };

    this.buffer[dataIndex] = transition;
    this.index = (this.index + 1) % this.capacity;
    if (this.size < this.capacity) {
      this.size++;
    }
    return transition;
  }

  public addTransition(trans: SACTransition) {
    const priority = Math.pow(this.maxPriority, this.alpha);
    const treeIndex = this.tree.add(priority);
    const dataIndex = this.index;

    trans.treeIndex = treeIndex;
    trans.priority = this.maxPriority;

    this.buffer[dataIndex] = trans;
    this.index = (this.index + 1) % this.capacity;
    if (this.size < this.capacity) {
      this.size++;
    }
  }

  public sampleUniform(batchSize: number): {
    batch: SACTransition[];
    isWeights: Float32Array;
    treeIndices: number[];
  } {
    const batch: SACTransition[] = [];
    const treeIndices: number[] = [];
    const count = Math.min(batchSize, this.size);
    const isWeights = new Float32Array(count);
    isWeights.fill(1.0);

    for (let i = 0; i < count; i++) {
      const randIdx = Math.floor(Math.random() * this.size);
      const item = this.buffer[randIdx];
      if (item) {
        batch.push(item);
        treeIndices.push(item.treeIndex ?? randIdx);
      }
    }
    return { batch, isWeights, treeIndices };
  }

  public samplePrioritized(
    batchSize: number,
    beta: number = 0.4
  ): { batch: SACTransition[]; isWeights: Float32Array; treeIndices: number[] } {
    if (this.size === 0) {
      return { batch: [], isWeights: new Float32Array(0), treeIndices: [] };
    }

    const batch: SACTransition[] = [];
    const treeIndices: number[] = [];
    const count = Math.min(batchSize, this.size);
    const rawWeights = new Float32Array(count);

    const totalP = this.tree.totalPriority;
    const segment = totalP / count;
    let maxWeight = 1e-8;

    for (let i = 0; i < count; i++) {
      const a = segment * i;
      const b = segment * (i + 1);
      const val = a + Math.random() * (b - a);

      const leaf = this.tree.get(val);
      const dataIdx = leaf.dataIndex;
      const trans = this.buffer[dataIdx] || this.buffer[Math.floor(Math.random() * this.size)];

      batch.push(trans);
      treeIndices.push(leaf.treeIndex);

      // Prob P(i) = p_i / totalP
      const p_i = Math.max(1e-6, leaf.priority) / totalP;
      // Importance Sampling weight: w_i = (N * P(i))^(-beta)
      const w_i = Math.pow(Math.max(1e-6, this.size * p_i), -beta);
      rawWeights[i] = w_i;
      if (w_i > maxWeight) maxWeight = w_i;
    }

    // Normalize weights by maxWeight so w_i in (0, 1]
    const isWeights = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      isWeights[i] = rawWeights[i] / maxWeight;
    }

    return { batch, isWeights, treeIndices };
  }

  public sample(
    batchSize: number,
    usePER: boolean = true,
    beta: number = 0.4
  ): {
    batch: SACTransition[];
    isWeights: Float32Array;
    treeIndices: number[];
  } {
    if (usePER) {
      return this.samplePrioritized(batchSize, beta);
    }
    return this.sampleUniform(batchSize);
  }

  public updatePriorities(treeIndices: number[], tdErrors: number[] | Float32Array) {
    for (let i = 0; i < treeIndices.length; i++) {
      const td = Math.abs(tdErrors[i]) + this.eps;
      const clippedTd = Math.min(10.0, td);
      const priority = Math.pow(clippedTd, this.alpha);
      this.maxPriority = Math.max(this.maxPriority, clippedTd);
      this.tree.update(treeIndices[i], priority);
    }
  }

  public clear() {
    this.buffer = new Array(this.capacity);
    this.tree.clear();
    this.index = 0;
    this.size = 0;
    this.maxPriority = 1.0;
  }
}

/**
 * Continuous 3D Soft Actor-Critic (SAC) Agent for Autonomous Snake Length Maximization
 */
export class ContinuousSACAgent {
  public obsDim: number;
  public actionDim: number = 7;
  public gamma: number = 0.99;
  public tau: number = 0.005;
  public lr: number = 3e-4;

  // Neural Networks
  public actor: MLP; // Input: obsDim -> Output: 14 (mean: 7, log_std: 7)
  public critic1: MLP; // Input: obsDim + 7 -> Output: 1 (Q1)
  public critic2: MLP; // Input: obsDim + 7 -> Output: 1 (Q2)
  public targetCritic1: MLP;
  public targetCritic2: MLP;

  // Automatic Entropy Tuning
  public logAlpha: number = 0.0; // alpha = exp(logAlpha)
  public mLogAlpha: number = 0.0;
  public vLogAlpha: number = 0.0;
  public targetEntropy: number = -7.0; // -dim(A)

  // Experience Replay Buffer
  public replay: ReplayBuffer;

  // Optimization step tracking
  public updateCount: number = 0;
  public totalStepCount: number = 0;
  public rolloutEpisode: number = 1;

  // Metrics & Diagnostics
  public lastActorLoss: number = 0;
  public lastCriticLoss: number = 0;
  public lastQMinMean: number = 0;
  public lastStepReward: number = 0;
  public totalAccumulatedReward: number = 0;
  public previousAction: Float32Array = new Float32Array([0, 0, 0, 0, 0, 0, 0]);
  public currentAction: number[] = [0, 0, 0, 0, 0, 0, 0];
  public currentVelocity: [number, number, number] = [0, 0, 0];

  // Prioritized Experience Replay (PER) & N-Step Returns Hyperparameters
  public usePER: boolean = true;
  public perAlpha: number = 0.6;
  public perBeta: number = 0.4;
  public useNStep: boolean = true;
  public nStep: number = 3;
  public nStepBuffer: SACTransition[] = [];
  public meanTDError: number = 0;
  public maxTDError: number = 0;

  // Curriculum Staging & Potential-Based Reward Shaping (PBRS)
  public curriculumStage: CurriculumStage = 'stage_2_tactical_harvesting';
  public usePBRS: boolean = true;
  public pbrsWeight: number = 0.45;
  public useLayerNorm: boolean = true;
  public domainRandomization: boolean = true;
  public encirclementBonusWeight: number = 0.8;
  public lastPBRSPotential: number = 0;
  public lastPBRSShapingReward: number = 0;
  public enclosedWindingArea: number = 0;
  public radarSectorDensities: number[] = new Array(8).fill(0);

  // Emergent Tactical Telemetry & Behavioral Profiling
  public behaviorProfile: EmergentBehaviorProfile = 'adaptive_predator';
  public currentEmergentTactic: EmergentTactic = 'free_adaptive_cruise';
  public emergentConfidence: number = 0.85;
  public tacticalIntentScores: Record<string, number> = {
    spear_evade: 0.15,
    orbital_flank: 0.20,
    bait_blast: 0.10,
    trench_choke: 0.05,
    barrier_shield: 0.05,
    surge_dash: 0.10,
    spiral_coil: 0.10,
    dive_bomb: 0.10,
    cruise: 0.15,
  };

  // Ability Executions Tracking
  public abilityExecutions: {
    repels: number;
    dashes: number;
    barriers: number;
    trenches: number;
  } = {
    repels: 0,
    dashes: 0,
    barriers: 0,
    trenches: 0,
  };

  // Statistics Logging
  public lengthHistory: number[] = [];
  public damageHistory: number[] = [];
  public totalSwarmHits: number = 0;
  public totalBiteDamage: number = 0;
  public totalGrowthSurges: number = 0;
  public maxLengthSeen: number = 24;
  public initialEpisodeLength: number = 24;
  public recentLossHistory: { step: number; actorLoss: number; criticLoss: number; length: number; alpha: number }[] = [];

  // Intermediate gradient buffers for Critic inputs
  private criticInputBuf: Float32Array;
  private nextCriticInputBuf: Float32Array;
  private actionGradBuf: Float32Array;

  constructor(
    obsDim: number = 126,
    actionDim: number = 7,
    lr: number = 3e-4,
    gamma: number = 0.99,
    tau: number = 0.005,
    replayCapacity: number = 50000
  ) {
    this.obsDim = obsDim;
    this.actionDim = actionDim;
    this.lr = lr;
    this.gamma = gamma;
    this.tau = tau;
    this.targetEntropy = -actionDim;

    // Networks architecture: 128-dim hidden layers with Layer Normalization & LeakyReLU
    this.actor = new MLP(obsDim, 128, actionDim * 2, 0.15, this.useLayerNorm);
    this.critic1 = new MLP(obsDim + actionDim, 128, 1, 0.15, this.useLayerNorm);
    this.critic2 = new MLP(obsDim + actionDim, 128, 1, 0.15, this.useLayerNorm);
    this.targetCritic1 = new MLP(obsDim + actionDim, 128, 1, 0.15, this.useLayerNorm);
    this.targetCritic2 = new MLP(obsDim + actionDim, 128, 1, 0.15, this.useLayerNorm);

    this.replay = new ReplayBuffer(replayCapacity);
    this.criticInputBuf = new Float32Array(obsDim + actionDim);
    this.nextCriticInputBuf = new Float32Array(obsDim + actionDim);
    this.actionGradBuf = new Float32Array(actionDim);

    // Bootstrap initial neural network weights with strong emergent behavioral priors
    this.initializeEmergentPriors(this.behaviorProfile);

    this.targetCritic1.copyFrom(this.critic1);
    this.targetCritic2.copyFrom(this.critic2);

    // Auto-load previously saved learned model from localStorage if exists
    this.loadFromLocalStorage();
  }

  /**
   * Initializes or bootstraps the Actor & Critic neural layers with structured behavioral priors.
   * This guarantees that from Step 0, the RL policy exhibits rich, intelligent emergent maneuvers
   * (soaring over spear charges, kinetic blast timing, orbital kiting, and terrain deflections)
   * while continuing to learn and optimize via live Bellman SGD updates.
   */
  public initializeEmergentPriors(profile: EmergentBehaviorProfile = 'adaptive_predator') {
    this.behaviorProfile = profile;

    // Layer 1 features:
    // obs[0..2] = pos, obs[3..5] = vel, obs[6] = len, obs[7..12] = bounds
    // obs[13..72] = nearest 6 boids (rel dx, dy, dz, dvx, dvy, dvz, dist, predDx, predDy, predDz)
    // obs[73..96] = 8-Sector Egocentric Radar (density[8], minDist[8], approachVel[8])
    // obs[97..102] = spine curvature & proprioception
    // obs[103..109] = swarm centroid kinematics & threat vector
    // obs[110] = local density, obs[120] = repelCd, obs[121] = dashCd, obs[122] = attackState (spear)
    // obs[123] = isShielded, obs[124] = enclosedWindingArea
    const w1 = this.actor.l1.weights;
    const inD = this.obsDim; // 126
    const hD = 128;

    // Zero out base weights and bias
    this.actor.l1.weights.fill(0);
    this.actor.l1.bias.fill(0);
    this.actor.l2.weights.fill(0);
    this.actor.l2.bias.fill(0);
    this.actor.lOut.weights.fill(0);
    this.actor.lOut.bias.fill(0);

    // Populate foundational heuristic pathways in Layer 1
    for (let h = 0; h < hD; h++) {
      // General subtle exploratory noise
      for (let j = 0; j < inD; j++) {
        w1[h * inD + j] = (Math.random() * 2 - 1) * 0.05;
      }

      // Feature extraction sub-banks:
      if (h < 24) {
        // [0..23] Evasion bank: strongly reacts to nearest boids & sectoral radar
        w1[h * inD + 13] = -1.2 - Math.random() * 0.8; // steer away from boid X
        w1[h * inD + 15] = -1.2 - Math.random() * 0.8; // steer away from boid Z
        w1[h * inD + 19] = -0.9; // scale with proximity
        if (inD > 75) {
          w1[h * inD + 73] = 0.8; // sector 0 density sensitivity
          w1[h * inD + 74] = 0.8; // sector 1 density
        }
        if (inD > 122) w1[h * inD + 122] = 1.4; // spear state sensitivity
        this.actor.l1.bias[h] = 0.3;
      } else if (h < 48) {
        // [24..47] 3D Altitude / Soaring bank: reacts to attackState & boid closeness
        if (inD > 122) w1[h * inD + 122] = 2.2 + (profile === 'aerial_spear_hunter' ? 1.5 : 0.0);
        w1[h * inD + 14] = -1.5; // climb if boid is below or level
        w1[h * inD + 9] = 1.2; // ground avoidance
        w1[h * inD + 10] = -1.5; // ceiling avoidance
        this.actor.l1.bias[h] = 0.4;
      } else if (h < 72) {
        // [48..71] Kinetic Repel Blast Trigger bank: reacts to local density & ready CD
        const densIdx = inD > 110 ? 110 : 93;
        const repelIdx = inD > 120 ? 120 : 103;
        w1[h * inD + densIdx] = 2.8 + (profile === 'bait_blast_specialist' ? 1.4 : 0.0);
        w1[h * inD + 19] = -2.2; // close proximity
        w1[h * inD + repelIdx] = -3.5; // only when repel is not cooling (cd low)
        this.actor.l1.bias[h] = -0.2;
      } else if (h < 96) {
        // [72..95] Dash & Agility bank: reacts to danger & high speed spear charges
        const attackIdx = inD > 122 ? 122 : 105;
        const dashIdx = inD > 121 ? 121 : 104;
        w1[h * inD + attackIdx] = 1.8; // spear attack thrust
        w1[h * inD + 19] = -1.8; // boid close
        w1[h * inD + dashIdx] = -2.5; // dash ready
        this.actor.l1.bias[h] = -0.4;
      } else if (h < 112) {
        // [96..111] Barrier & Trench Deployment bank: reacts to ground altitude & pursuers
        const densIdx = inD > 110 ? 110 : 93;
        const shieldIdx = inD > 123 ? 123 : 106;
        w1[h * inD + 1] = -1.5; // low altitude (ground)
        w1[h * inD + densIdx] = 1.8 + (profile === 'trench_architect' ? 1.6 : 0.0);
        w1[h * inD + shieldIdx] = -1.2; // deploy when not already shielded
        this.actor.l1.bias[h] = -0.3;
      } else {
        // [112..127] Orbital Flank / Corkscrew bank: tangential velocity shearing
        w1[h * inD + 13] = 1.1; // perpendicular shearing
        w1[h * inD + 15] = -1.1;
        w1[h * inD + 6] = 0.8; // length scaling
        if (inD > 124) w1[h * inD + 124] = 1.2; // encirclement loop bonus
        this.actor.l1.bias[h] = 0.2;
      }
    }

    // Layer 2 Identity / Feature Mixing Matrix
    const w2 = this.actor.l2.weights;
    for (let i = 0; i < hD; i++) {
      w2[i * hD + i] = 1.0;
      for (let j = 0; j < hD; j++) {
        if (i !== j) w2[i * hD + j] = (Math.random() * 2 - 1) * 0.04;
      }
    }

    // Layer Out mapping to 7 Actions + 7 Log_Std
    // Output: [ax(0), ay(1), az(2), a_repel(3), a_dash(4), a_barrier(5), a_trench(6), log_std[0..6]]
    const wOut = this.actor.lOut.weights;
    const outD = this.actionDim * 2; // 14

    for (let o = 0; o < outD; o++) {
      for (let h = 0; h < hD; h++) {
        wOut[o * hD + h] = (Math.random() * 2 - 1) * 0.03;
      }
    }

    // Connect specific feature banks to outputs:
    // a_x & a_z (Steering / Evasion)
    for (let h = 0; h < 24; h++) {
      wOut[0 * hD + h] = 0.45; // a_x
      wOut[2 * hD + h] = 0.45; // a_z
    }
    // a_y (Altitude Climb / Soar)
    for (let h = 24; h < 48; h++) {
      wOut[1 * hD + h] = 0.75; // a_y
    }
    // a_repel (Action 3)
    for (let h = 48; h < 72; h++) {
      wOut[3 * hD + h] = 1.1; // a_repel
    }
    // a_dash (Action 4)
    for (let h = 72; h < 96; h++) {
      wOut[4 * hD + h] = 0.95; // a_dash
    }
    // a_barrier & a_trench (Action 5 & 6)
    for (let h = 96; h < 112; h++) {
      wOut[5 * hD + h] = 0.85; // a_barrier
      wOut[6 * hD + h] = 0.85; // a_trench
    }
    // a_x & a_z orbital flank (Action 0 & 2)
    for (let h = 112; h < 128; h++) {
      wOut[0 * hD + h] += 0.35;
      wOut[2 * hD + h] += 0.35;
      if (profile === 'spiral_coil_tank') {
        wOut[1 * hD + h] += 0.50; // corkscrew climbing
      }
    }

    // Default log_std exploration initial floor (std ≈ 0.25)
    for (let i = 0; i < this.actionDim; i++) {
      this.actor.lOut.bias[this.actionDim + i] = -1.4;
    }
  }

  public exportModelJSON(): any {
    return {
      version: 2,
      modelType: 'SAC_CONTINUOUS_3D_LN',
      timestamp: Date.now(),
      savedAt: new Date().toISOString(),
      obsDim: this.obsDim,
      actionDim: this.actionDim,
      lr: this.lr,
      gamma: this.gamma,
      tau: this.tau,
      logAlpha: this.logAlpha,
      mLogAlpha: this.mLogAlpha,
      vLogAlpha: this.vLogAlpha,
      updateCount: this.updateCount,
      totalStepCount: this.totalStepCount,
      rolloutEpisode: this.rolloutEpisode,
      totalAccumulatedReward: this.totalAccumulatedReward,
      maxLengthSeen: this.maxLengthSeen,
      totalSwarmHits: this.totalSwarmHits,
      totalBiteDamage: this.totalBiteDamage,
      totalGrowthSurges: this.totalGrowthSurges,
      behaviorProfile: this.behaviorProfile,
      curriculumStage: this.curriculumStage,
      usePBRS: this.usePBRS,
      pbrsWeight: this.pbrsWeight,
      useLayerNorm: this.useLayerNorm,
      domainRandomization: this.domainRandomization,
      abilityExecutions: { ...this.abilityExecutions },
      networks: {
        actor: this.actor.toJSON(),
        critic1: this.critic1.toJSON(),
        critic2: this.critic2.toJSON(),
        targetCritic1: this.targetCritic1.toJSON(),
        targetCritic2: this.targetCritic2.toJSON(),
      },
      recentLossHistory: this.recentLossHistory.slice(-50),
    };
  }

  public importModelJSON(data: any): boolean {
    try {
      if (!data || typeof data !== 'object') return false;
      if (data.networks) {
        if (data.networks.actor) this.actor.fromJSON(data.networks.actor);
        if (data.networks.critic1) this.critic1.fromJSON(data.networks.critic1);
        if (data.networks.critic2) this.critic2.fromJSON(data.networks.critic2);
        if (data.networks.targetCritic1) this.targetCritic1.fromJSON(data.networks.targetCritic1);
        if (data.networks.targetCritic2) this.targetCritic2.fromJSON(data.networks.targetCritic2);
      }
      if (typeof data.logAlpha === 'number') this.logAlpha = data.logAlpha;
      if (typeof data.mLogAlpha === 'number') this.mLogAlpha = data.mLogAlpha;
      if (typeof data.vLogAlpha === 'number') this.vLogAlpha = data.vLogAlpha;
      if (typeof data.updateCount === 'number') this.updateCount = data.updateCount;
      if (typeof data.totalStepCount === 'number') this.totalStepCount = data.totalStepCount;
      if (typeof data.rolloutEpisode === 'number') this.rolloutEpisode = data.rolloutEpisode;
      if (typeof data.totalAccumulatedReward === 'number') this.totalAccumulatedReward = data.totalAccumulatedReward;
      if (typeof data.maxLengthSeen === 'number') this.maxLengthSeen = data.maxLengthSeen;
      if (typeof data.totalSwarmHits === 'number') this.totalSwarmHits = data.totalSwarmHits;
      if (typeof data.totalBiteDamage === 'number') this.totalBiteDamage = data.totalBiteDamage;
      if (typeof data.totalGrowthSurges === 'number') this.totalGrowthSurges = data.totalGrowthSurges;
      if (data.behaviorProfile) this.behaviorProfile = data.behaviorProfile;
      if (data.curriculumStage) this.curriculumStage = data.curriculumStage;
      if (typeof data.usePBRS === 'boolean') this.usePBRS = data.usePBRS;
      if (typeof data.pbrsWeight === 'number') this.pbrsWeight = data.pbrsWeight;
      if (typeof data.useLayerNorm === 'boolean') this.useLayerNorm = data.useLayerNorm;
      if (typeof data.domainRandomization === 'boolean') this.domainRandomization = data.domainRandomization;
      if (data.abilityExecutions) this.abilityExecutions = { ...data.abilityExecutions };
      if (Array.isArray(data.recentLossHistory)) this.recentLossHistory = data.recentLossHistory;
      return true;
    } catch (err) {
      console.warn('Failed to parse SAC model JSON:', err);
      return false;
    }
  }

  public saveToLocalStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return false;
      const json = JSON.stringify(this.exportModelJSON());
      window.localStorage.setItem('ursina_sac_continuous_model', json);
      return true;
    } catch (e) {
      return false;
    }
  }

  public loadFromLocalStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return false;
      const raw = window.localStorage.getItem('ursina_sac_continuous_model');
      if (!raw) return false;
      const parsed = JSON.parse(raw);
      return this.importModelJSON(parsed);
    } catch (e) {
      return false;
    }
  }

  public clearLocalStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.removeItem('ursina_sac_continuous_model');
      }
    } catch (e) {}
  }

  public get alpha(): number {
    return Math.exp(Math.max(-5.0, Math.min(2.0, this.logAlpha)));
  }

  /**
   * Evaluates the observation state and classifies current Emergent Tactic & Intent Scores
   */
  public evaluateEmergentTactics(state: Float32Array, action: Float32Array): { tactic: EmergentTactic; confidence: number } {
    const inD = this.obsDim;
    const boidDist = state[19] ?? 2.0; // closest boid dist
    const localDensity = inD > 110 ? state[110] : (state[93] ?? 0); // local cluster density
    const attackState = inD > 122 ? state[122] : (state[105] ?? 0); // 0=flock, 0.33=form spear, 0.66=thrust, 1.0=disperse
    const isShielded = (inD > 123 ? state[123] : (state[106] ?? 0)) > 0.5;
    const windingArea = inD > 124 ? state[124] : 0;

    // Cache radar sector densities for visualization telemetry
    if (inD >= 96) {
      for (let k = 0; k < 8; k++) {
        this.radarSectorDensities[k] = state[73 + k] ?? 0;
      }
    }
    this.enclosedWindingArea = windingArea;

    const ax = action[0];
    const ay = action[1];
    const az = action[2];
    const aRepel = action[3];
    const aDash = action[4];
    const aBarrier = action[5];
    const aTrench = action[6];

    const scores: Record<string, number> = {
      spear_evade: (attackState > 0.25 ? 0.6 : 0.1) + (ay > 0.3 ? 0.35 : 0.0),
      orbital_flank: (Math.hypot(ax, az) > 0.4 && Math.abs(ay) < 0.25) ? 0.75 : 0.2,
      bait_blast: (aRepel > 0.25 || localDensity > 0.4) ? 0.85 : 0.15,
      trench_choke: (aTrench > 0.3) ? 0.8 : 0.1,
      barrier_shield: (aBarrier > 0.3 || isShielded) ? 0.8 : 0.1,
      surge_dash: (aDash > 0.3) ? 0.85 : 0.15,
      spiral_coil: (windingArea > 0.15 || (Math.abs(ay) > 0.35 && Math.hypot(ax, az) > 0.35)) ? 0.85 : 0.2,
      dive_bomb: (ay < -0.35 && boidDist < 1.0) ? 0.8 : 0.15,
      cruise: (boidDist > 1.2 && attackState < 0.2) ? 0.7 : 0.2,
    };

    let highestScore = -1;
    let chosenTactic: EmergentTactic = 'free_adaptive_cruise';

    if (scores.spear_evade > 0.7 && attackState > 0.4) chosenTactic = 'soar_spear_evade';
    else if (scores.bait_blast > 0.7 && localDensity > 0.35) chosenTactic = 'bait_kinetic_blast';
    else if (scores.surge_dash > 0.7) chosenTactic = 'surge_dash_escape';
    else if (scores.barrier_shield > 0.65) chosenTactic = 'barrier_shield_deflect';
    else if (scores.trench_choke > 0.65) chosenTactic = 'trench_chokepoint_trap';
    else if (scores.spiral_coil > 0.7) chosenTactic = 'spiral_coil_defense';
    else if (scores.dive_bomb > 0.7) chosenTactic = 'volumetric_dive_bomb';
    else if (scores.orbital_flank > 0.6) chosenTactic = 'orbital_flank_peel';

    for (const key of Object.keys(scores)) {
      if (scores[key] > highestScore) highestScore = scores[key];
    }

    this.currentEmergentTactic = chosenTactic;
    this.emergentConfidence = Math.min(1.0, Math.max(0.4, highestScore));
    this.tacticalIntentScores = scores;

    return { tactic: chosenTactic, confidence: this.emergentConfidence };
  }

  /**
   * Potential Function for Potential-Based Reward Shaping (PBRS)
   * Guarantees policy invariance while accelerating convergence toward target behaviors
   */
  public calculatePotential(state: Float32Array): number {
    const inD = this.obsDim;
    const len = state[6] ?? 0;
    const localDensity = inD > 110 ? (state[110] ?? 0) : (state[93] ?? 0);
    const attackState = inD > 122 ? (state[122] ?? 0) : (state[105] ?? 0);
    const isShielded = (inD > 123 ? (state[123] ?? 0) : (state[106] ?? 0)) > 0.5 ? 1 : 0;
    const windingArea = inD > 124 ? (state[124] ?? 0) : 0;

    let phi = 0;
    // 1. Length preservation & expansion potential
    phi += len * 1.5;
    // 2. Encirclement loop trapping potential
    phi += windingArea * 3.0 * this.encirclementBonusWeight;
    // 3. Cluster density harvesting potential
    phi += localDensity * 1.8;
    // 4. Tactical barrier coverage
    if (isShielded > 0.5) phi += 1.0;
    // 5. Threat avoidance (penalize vulnerability when spear thrust is charging)
    if (attackState > 0.5 && isShielded <= 0.5) phi -= 1.5;

    return phi;
  }

  /**
   * Computes PBRS shaping reward: F(s, s') = gamma * Phi(s') - Phi(s)
   */
  public computePBRS(
    prevState: Float32Array | null,
    currState: Float32Array,
    gamma: number = 0.99
  ): number {
    if (!prevState || !this.usePBRS) {
      this.lastPBRSPotential = this.calculatePotential(currState);
      this.lastPBRSShapingReward = 0;
      return 0;
    }
    const prevPhi = this.calculatePotential(prevState);
    const currPhi = this.calculatePotential(currState);
    const shaping = gamma * currPhi - prevPhi;
    this.lastPBRSPotential = currPhi;
    this.lastPBRSShapingReward = shaping * this.pbrsWeight;
    return this.lastPBRSShapingReward;
  }

  /**
   * Samples a 7D continuous action a in [-1, 1]^7 given observation state s
   */
  public chooseAction(
    state: Float32Array,
    deterministic: boolean = false
  ): { action: Float32Array; logProb: number; rawMean: Float32Array; rawU: Float32Array; rawStd: Float32Array } {
    const actorOut = this.actor.forward(state);

    const mean = new Float32Array(this.actionDim);
    const logStd = new Float32Array(this.actionDim);
    const stdVec = new Float32Array(this.actionDim);
    const uVec = new Float32Array(this.actionDim);
    const action = new Float32Array(this.actionDim);
    let logProb = 0;

    for (let i = 0; i < this.actionDim; i++) {
      mean[i] = actorOut[i];
      // Clamp log standard deviation to [-3.0, 0.0] (std in [0.05, 1.0]) for smooth continuous control
      logStd[i] = Math.max(-3.0, Math.min(0.0, actorOut[this.actionDim + i]));
      const std = Math.exp(logStd[i]);
      stdVec[i] = std;

      let u = mean[i];
      if (!deterministic) {
        const eps = sampleStandardNormal();
        u = mean[i] + std * eps;
      }
      uVec[i] = u;

      // Squashed action via tanh
      const act = Math.tanh(u);
      action[i] = Math.max(-0.9999, Math.min(0.9999, act));

      // Gaussian log probability: -0.5 * ((u - mean) / std)^2 - log(std * sqrt(2pi))
      const normLogProb = -0.5 * Math.pow((u - mean[i]) / std, 2) - Math.log(std * 2.50662827463);
      // Tanh squashing Jacobian correction: - log(1 - tanh(u)^2 + 1e-6)
      const correction = Math.log(Math.max(1e-6, 1.0 - act * act));
      logProb += normLogProb - correction;
    }

    this.currentAction = Array.from(action);
    this.evaluateEmergentTactics(state, action);

    return { action, logProb, rawMean: mean, rawU: uVec, rawStd: stdVec };
  }

  /**
   * Calculates rich SAC reward based on length, net growth, 3D altitudinal evasion, tactical abilities, and action smoothness
   */
  public calculateReward(
    oldLength: number,
    newLength: number,
    action: Float32Array,
    prevAction: Float32Array,
    lengthScale: number = 50.0,
    variant: SACRewardVariant = 'variant_c_combined',
    alphaWeight: number = 1.0,
    betaWeight: number = 1.0,
    lambdaSmoothness: number = 0.005,
    extraMetrics?: {
      hitsCount?: number;
      repelledCount?: number;
      dashExecuted?: boolean;
      barrierPlaced?: boolean;
      trenchCarved?: boolean;
      altitudeClearanceBonus?: number;
      actionDiversityBonus?: number;
      spearEvasionBonus?: number;
      clusterBaitBonus?: number;
      encirclementBonus?: number;
      pbrsShaping?: number;
    }
  ): number {
    const normalizedLength = newLength / lengthScale;
    const growthDelta = (newLength - oldLength) / lengthScale;

    // Smoothness penalty: ||a_t - a_{t-1}||^2 (penalizes erratic twitching while allowing rapid tactical turns)
    let smoothness = 0;
    const count = Math.min(action.length, prevAction.length);
    for (let i = 0; i < count; i++) {
      const diff = action[i] - prevAction[i];
      smoothness += diff * diff;
    }

    let baseReward = 0;
    switch (variant) {
      case 'variant_a_net_growth':
        baseReward = growthDelta * 2.5 - lambdaSmoothness * smoothness;
        break;
      case 'variant_b_max_size':
        baseReward = normalizedLength * 1.5 - lambdaSmoothness * smoothness;
        break;
      case 'variant_c_combined':
      default:
        baseReward =
          alphaWeight * normalizedLength +
          betaWeight * growthDelta * 1.5 -
          lambdaSmoothness * smoothness;
        break;
    }

    // Curriculum stage weighting multipliers
    let hitPenaltyMult = 1.0;
    let growthBonusMult = 1.0;
    let tacticalBonusMult = 1.0;

    switch (this.curriculumStage) {
      case 'stage_1_survival':
        hitPenaltyMult = 1.6;
        growthBonusMult = 0.5;
        tacticalBonusMult = 0.4;
        break;
      case 'stage_2_tactical_harvesting':
        hitPenaltyMult = 1.0;
        growthBonusMult = 1.2;
        tacticalBonusMult = 1.3;
        break;
      case 'stage_3_adversarial_mastery':
      default:
        hitPenaltyMult = 1.0;
        growthBonusMult = 1.4;
        tacticalBonusMult = 1.5;
        break;
    }

    // Emergent 3D Altitudinal & Tactical Ability Reward Shaping
    let tacticalReward = 0;
    if (extraMetrics) {
      // 1. Penalty for taking swarm bites
      if (extraMetrics.hitsCount && extraMetrics.hitsCount > 0) {
        tacticalReward -= extraMetrics.hitsCount * 0.95 * hitPenaltyMult;
      }
      // 2. Reward for successful kinetic repel blast eliminating/scattering boids
      if (extraMetrics.repelledCount && extraMetrics.repelledCount > 0) {
        tacticalReward += 0.55 * Math.min(12, extraMetrics.repelledCount) * tacticalBonusMult;
      }
      // 3. Reward for tactical burst dash evasion
      if (extraMetrics.dashExecuted) {
        tacticalReward += 0.45 * tacticalBonusMult;
      }
      // 4. Reward for tactical barrier deflection deployment
      if (extraMetrics.barrierPlaced) {
        tacticalReward += 0.50 * tacticalBonusMult;
      }
      // 5. Reward for tactical trench carving
      if (extraMetrics.trenchCarved) {
        tacticalReward += 0.35 * tacticalBonusMult;
      }
      // 6. Bonus for 3D altitudinal maneuvering & vertical airspace exploration
      if (extraMetrics.altitudeClearanceBonus) {
        tacticalReward += extraMetrics.altitudeClearanceBonus * (this.curriculumStage === 'stage_1_survival' ? 1.5 : 1.0);
      }
      // 7. Bonus for evading incoming spear charge
      if (extraMetrics.spearEvasionBonus) {
        tacticalReward += extraMetrics.spearEvasionBonus;
      }
      // 8. Bonus for baiting high-density clusters into safe repel detonation
      if (extraMetrics.clusterBaitBonus) {
        tacticalReward += extraMetrics.clusterBaitBonus * tacticalBonusMult;
      }
      // 9. Encirclement & topological loop trapping bonus
      if (extraMetrics.encirclementBonus) {
        tacticalReward += extraMetrics.encirclementBonus * this.encirclementBonusWeight;
      }
      // 10. Action entropy & behavioral diversity bonus
      if (extraMetrics.actionDiversityBonus) {
        tacticalReward += extraMetrics.actionDiversityBonus;
      }
      // 11. Potential-Based Reward Shaping (PBRS) term
      if (this.usePBRS && extraMetrics.pbrsShaping !== undefined) {
        tacticalReward += extraMetrics.pbrsShaping;
      }
    }

    const totalReward = baseReward * growthBonusMult + tacticalReward;

    this.lastStepReward = totalReward;
    this.totalAccumulatedReward += totalReward;
    return totalReward;
  }

  /**
   * Stores a transition into the N-step accumulation queue, collapsing multi-step
   * returns when nStep transitions are collected or when an episode concludes.
   */
  public storeExperience(
    state: Float32Array,
    action: Float32Array,
    reward: number,
    nextState: Float32Array,
    done: boolean
  ) {
    const rawTrans: SACTransition = {
      state: new Float32Array(state),
      action: new Float32Array(action),
      reward,
      nextState: new Float32Array(nextState),
      done,
      gammaN: this.gamma,
      nStep: 1,
    };

    if (!this.useNStep || this.nStep <= 1) {
      this.replay.addTransition(rawTrans);
      return;
    }

    this.nStepBuffer.push(rawTrans);

    if (this.nStepBuffer.length < this.nStep && !done) {
      return;
    }

    while (this.nStepBuffer.length >= this.nStep || (done && this.nStepBuffer.length > 0)) {
      const k = this.nStepBuffer.length;
      let nStepReward = 0;
      let gammaAccum = 1.0;
      let isTerminated = false;
      let lastNextState = this.nStepBuffer[k - 1].nextState;

      for (let i = 0; i < Math.min(k, this.nStep); i++) {
        const stepTrans = this.nStepBuffer[i];
        nStepReward += gammaAccum * stepTrans.reward;
        gammaAccum *= this.gamma;
        if (stepTrans.done) {
          isTerminated = true;
          lastNextState = stepTrans.nextState;
          break;
        }
      }

      const firstTrans = this.nStepBuffer.shift()!;
      const actualN = isTerminated ? 1 : Math.min(k, this.nStep);

      this.replay.add(
        firstTrans.state,
        firstTrans.action,
        nStepReward,
        lastNextState,
        isTerminated || firstTrans.done,
        gammaAccum,
        actualN
      );

      if (!done) break;
    }
  }

  public addExperience(
    state: Float32Array,
    action: Float32Array,
    reward: number,
    nextState: Float32Array,
    done: boolean
  ) {
    this.storeExperience(state, action, reward, nextState, done);
  }

  /**
   * Executes one mini-batch Soft Actor-Critic update step with Prioritized Experience Replay
   * and N-Step Discounted Bellman Targets.
   */
  public learn(batchSize: number = 128): {
    criticLoss: number;
    actorLoss: number;
    alpha: number;
    qMinMean: number;
    meanTDError?: number;
    maxTDError?: number;
  } | null {
    if (this.replay.size < batchSize) {
      return null;
    }

    // Proportional beta annealing schedule: beta -> 1.0
    const betaProgress = Math.min(1.0, this.updateCount / 15000);
    const currentBeta = this.perBeta + (1.0 - this.perBeta) * betaProgress;

    const { batch, isWeights, treeIndices } = this.replay.sample(
      batchSize,
      this.usePER,
      currentBeta
    );

    this.updateCount++;
    const t = this.updateCount;
    const alphaVal = this.alpha;

    let totalCriticLoss = 0;
    let totalActorLoss = 0;
    let totalQMin = 0;
    let totalAlphaLossGrad = 0;
    const tdErrors = new Float32Array(batch.length);

    // =========================================================================
    // 1. COMPUTE TARGET Q-VALUES & TRAIN TWIN CRITICS (N-Step + PER Weighted)
    // =========================================================================
    for (let b = 0; b < batch.length; b++) {
      const trans = batch[b];
      const weight = this.usePER ? isWeights[b] : 1.0;

      // Sample a' ~ pi(s')
      const nextSample = this.chooseAction(trans.nextState, false);
      const nextAct = nextSample.action;
      const nextLogProb = nextSample.logProb;

      // Concatenate [s', a']
      this.nextCriticInputBuf.set(trans.nextState, 0);
      this.nextCriticInputBuf.set(nextAct, this.obsDim);

      const targetQ1 = this.targetCritic1.forward(this.nextCriticInputBuf)[0];
      const targetQ2 = this.targetCritic2.forward(this.nextCriticInputBuf)[0];
      const targetQMin = Math.min(targetQ1, targetQ2);

      // Multi-Step Discounted Bellman target:
      // y = R^{(n)} + (1 - done) * gamma^n * (min(Q1, Q2) - alpha * log_pi)
      const gammaN = trans.gammaN ?? Math.pow(this.gamma, trans.nStep ?? 1);
      const targetY = trans.reward + (trans.done ? 0.0 : gammaN * (targetQMin - alphaVal * nextLogProb));

      // Forward current [s, a]
      this.criticInputBuf.set(trans.state, 0);
      this.criticInputBuf.set(trans.action, this.obsDim);

      const q1 = this.critic1.forward(this.criticInputBuf)[0];
      const q2 = this.critic2.forward(this.criticInputBuf)[0];

      // TD-Error magnitude for PER priority update: delta = 0.5 * (|Q1 - y| + |Q2 - y|)
      const diff1 = q1 - targetY;
      const diff2 = q2 - targetY;
      const tdError = (Math.abs(diff1) + Math.abs(diff2)) * 0.5;
      tdErrors[b] = tdError;

      // Importance Sampling Weighted MSE Loss: w_i * (Q - y)^2
      totalCriticLoss += weight * (diff1 * diff1 + diff2 * diff2) * 0.5;

      // Weighted Critic Gradients
      this.critic1.backward(this.criticInputBuf, new Float32Array([(weight * diff1) / batch.length]), null);
      this.critic2.backward(this.criticInputBuf, new Float32Array([(weight * diff2) / batch.length]), null);
    }

    // Update priorities in Sum-Tree for PER
    if (this.usePER && treeIndices.length > 0) {
      this.replay.updatePriorities(treeIndices, tdErrors);
    }

    // Telemetry for TD Errors
    let tdSum = 0;
    let tdMax = 0;
    for (let i = 0; i < tdErrors.length; i++) {
      tdSum += tdErrors[i];
      if (tdErrors[i] > tdMax) tdMax = tdErrors[i];
    }
    this.meanTDError = tdSum / Math.max(1, tdErrors.length);
    this.maxTDError = tdMax;

    // Step Critic Optimizers
    this.critic1.stepAdam(this.lr, t);
    this.critic2.stepAdam(this.lr, t);

    // =========================================================================
    // 2. TRAIN ACTOR POLICY & AUTOMATIC ENTROPY TEMPERATURE
    // =========================================================================
    for (let b = 0; b < batch.length; b++) {
      const trans = batch[b];

      // Sample a_new ~ pi(s) with reparameterization variables
      const sample = this.chooseAction(trans.state, false);
      const newAct = sample.action;
      const logProb = sample.logProb;

      this.criticInputBuf.set(trans.state, 0);
      this.criticInputBuf.set(newAct, this.obsDim);

      const q1New = this.critic1.forward(this.criticInputBuf)[0];
      const q2New = this.critic2.forward(this.criticInputBuf)[0];
      const qMinNew = Math.min(q1New, q2New);
      totalQMin += qMinNew;

      // Actor Objective Loss: alpha * log_pi - min(Q1, Q2)
      const actorLoss = alphaVal * logProb - qMinNew;
      totalActorLoss += actorLoss;

      // Compute exact analytical gradient of min(Q1, Q2) w.r.t action vector a
      const bestCritic = q1New <= q2New ? this.critic1 : this.critic2;
      bestCritic.computeActionGradient(this.criticInputBuf, this.obsDim, this.actionDim, this.actionGradBuf);

      // Actor Reparameterized Gradient w.r.t network outputs [mean, log_std]
      const dActorOut = new Float32Array(this.actionDim * 2);
      for (let i = 0; i < this.actionDim; i++) {
        const ai = newAct[i];
        const ui = sample.rawU[i];
        const mui = sample.rawMean[i];
        const stdi = sample.rawStd[i];
        const invVar = 1.0 / (stdi * stdi + 1e-8);
        const dQda = this.actionGradBuf[i];

        // Gradient of tanh: d(ai)/d(ui) = 1 - ai^2
        const tanhGrad = Math.max(1e-5, 1.0 - ai * ai);

        // Analytical derivative of J_actor w.r.t squashed action pre-activation ui
        const dJ_dui = alphaVal * (-(ui - mui) * invVar + (2 * ai) / (tanhGrad + 1e-6)) - dQda * tanhGrad;

        // Chain rule w.r.t Actor output mean mu_i
        const dLoss_dMean = (dJ_dui + alphaVal * (ui - mui) * invVar) / batch.length;

        // Chain rule w.r.t Actor output log_std_i
        const dLoss_dLogStd = (dJ_dui * (ui - mui) + alphaVal * ((ui - mui) * (ui - mui) * invVar - 1.0)) / batch.length;

        // Clamp individual sample gradients to [-2.0 / B, 2.0 / B] for stable convergence
        const maxGrad = 2.0 / batch.length;
        dActorOut[i] = Math.max(-maxGrad, Math.min(maxGrad, dLoss_dMean));
        dActorOut[this.actionDim + i] = Math.max(-maxGrad, Math.min(maxGrad, dLoss_dLogStd));
      }
      this.actor.backward(trans.state, dActorOut, null);

      // Entropy temperature gradient: -logAlpha * (logProb + targetEntropy)
      totalAlphaLossGrad += -(logProb + this.targetEntropy) / batch.length;
    }

    // Step Actor Optimizer
    this.actor.stepAdam(this.lr, t);

    // Step Alpha Temperature Optimizer
    const alphaFix1 = 1.0 - Math.pow(0.9, t);
    const alphaFix2 = 1.0 - Math.pow(0.999, t);
    const clippedAlphaGrad = Math.max(-2.0, Math.min(2.0, totalAlphaLossGrad));
    this.mLogAlpha = 0.9 * this.mLogAlpha + 0.1 * clippedAlphaGrad;
    this.vLogAlpha = 0.999 * this.vLogAlpha + 0.001 * (clippedAlphaGrad * clippedAlphaGrad);
    const mHat = this.mLogAlpha / alphaFix1;
    const vHat = this.vLogAlpha / alphaFix2;
    this.logAlpha -= (this.lr * mHat) / (Math.sqrt(vHat) + 1e-8);
    this.logAlpha = Math.max(-5.0, Math.min(2.0, this.logAlpha));

    // =========================================================================
    // 3. SOFT POLYAK TARGET UPDATE: theta_target <- (1-tau)*theta_target + tau*theta
    // =========================================================================
    this.targetCritic1.polyakUpdate(this.critic1, this.tau);
    this.targetCritic2.polyakUpdate(this.critic2, this.tau);

    const avgCriticLoss = totalCriticLoss / batch.length;
    const avgActorLoss = totalActorLoss / batch.length;
    const avgQMin = totalQMin / batch.length;

    this.lastCriticLoss = avgCriticLoss;
    this.lastActorLoss = avgActorLoss;
    this.lastQMinMean = avgQMin;

    if (t % 10 === 0) {
      this.recentLossHistory.push({
        step: this.totalStepCount,
        actorLoss: avgActorLoss,
        criticLoss: avgCriticLoss,
        length: this.lengthHistory.length > 0 ? this.lengthHistory[this.lengthHistory.length - 1] : 24,
        alpha: this.alpha,
      });
      if (this.recentLossHistory.length > 30) {
        this.recentLossHistory.shift();
      }
    }

    return {
      criticLoss: avgCriticLoss,
      actorLoss: avgActorLoss,
      alpha: this.alpha,
      qMinMean: avgQMin,
      meanTDError: this.meanTDError,
      maxTDError: this.maxTDError,
    };
  }

  /**
   * Records step statistics for long-run SAC evaluation metrics
   */
  public logStepMetrics(currentLength: number, bitesTaken: number, growthSurge: boolean) {
    this.totalStepCount++;
    this.lengthHistory.push(currentLength);
    if (this.lengthHistory.length > 5000) {
      this.lengthHistory.shift();
    }

    if (currentLength > this.maxLengthSeen) {
      this.maxLengthSeen = currentLength;
    }

    if (bitesTaken > 0) {
      this.totalSwarmHits += bitesTaken;
      this.totalBiteDamage += bitesTaken;
    }
    if (growthSurge) {
      this.totalGrowthSurges++;
    }

    this.damageHistory.push(bitesTaken);
    if (this.damageHistory.length > 1000) {
      this.damageHistory.shift();
    }
  }

  public getMeanLength(): number {
    if (this.lengthHistory.length === 0) return 24;
    let sum = 0;
    for (let i = 0; i < this.lengthHistory.length; i++) {
      sum += this.lengthHistory[i];
    }
    return sum / this.lengthHistory.length;
  }

  public getDamagePer1000Steps(): number {
    if (this.damageHistory.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.damageHistory.length; i++) {
      sum += this.damageHistory[i];
    }
    return (sum / this.damageHistory.length) * 1000;
  }

  public resetTrajectory(initialLength: number = 24) {
    this.rolloutEpisode++;
    this.initialEpisodeLength = initialLength;
  }

  public resetAll(profile?: EmergentBehaviorProfile) {
    this.replay.clear();
    this.nStepBuffer = [];
    this.meanTDError = 0;
    this.maxTDError = 0;

    this.actor = new MLP(this.obsDim, 128, this.actionDim * 2, 0.15);
    this.critic1 = new MLP(this.obsDim + this.actionDim, 128, 1, 0.15);
    this.critic2 = new MLP(this.obsDim + this.actionDim, 128, 1, 0.15);
    this.targetCritic1 = new MLP(this.obsDim + this.actionDim, 128, 1, 0.15);
    this.targetCritic2 = new MLP(this.obsDim + this.actionDim, 128, 1, 0.15);
    
    this.initializeEmergentPriors(profile || this.behaviorProfile);
    this.targetCritic1.copyFrom(this.critic1);
    this.targetCritic2.copyFrom(this.critic2);

    this.logAlpha = 0.0;
    this.mLogAlpha = 0.0;
    this.vLogAlpha = 0.0;
    this.updateCount = 0;
    this.totalStepCount = 0;
    this.rolloutEpisode = 1;
    this.totalAccumulatedReward = 0;
    this.lastStepReward = 0;
    this.lastActorLoss = 0;
    this.lastCriticLoss = 0;
    this.lengthHistory = [];
    this.damageHistory = [];
    this.totalSwarmHits = 0;
    this.totalBiteDamage = 0;
    this.maxLengthSeen = 24;
    this.recentLossHistory = [];
    this.previousAction = new Float32Array([0, 0, 0, 0, 0, 0, 0]);
    this.currentAction = [0, 0, 0, 0, 0, 0, 0];
    this.abilityExecutions = { repels: 0, dashes: 0, barriers: 0, trenches: 0 };
  }

  public getMetrics(enabled: boolean = true, isEvaluation: boolean = false, rewardVariant: SACRewardVariant = 'variant_c_combined'): SACMetrics {
    const currentLen = this.lengthHistory.length > 0 ? this.lengthHistory[this.lengthHistory.length - 1] : 24;
    return {
      enabled,
      isEvaluation,
      actorLoss: this.lastActorLoss,
      criticLoss: this.lastCriticLoss,
      alpha: this.alpha,
      targetEntropy: this.targetEntropy,
      replayBufferSize: this.replay.size,
      replayBufferCapacity: this.replay.capacity,
      meanLength: this.getMeanLength(),
      maxLength: this.maxLengthSeen,
      damagePer1000Steps: this.getDamagePer1000Steps(),
      growthPerStep: this.totalGrowthSurges / Math.max(1, this.totalStepCount),
      netGrowth: currentLen - this.initialEpisodeLength,
      swarmHitsTotal: this.totalSwarmHits,
      qMinMean: this.lastQMinMean,
      currentAction: this.currentAction,
      currentVelocity: this.currentVelocity,
      currentEmergentTactic: this.currentEmergentTactic,
      emergentConfidence: this.emergentConfidence,
      behaviorProfile: this.behaviorProfile,
      tacticalIntentScores: { ...this.tacticalIntentScores },
      abilityIntents: {
        repel: this.currentAction[3] ?? 0,
        dash: this.currentAction[4] ?? 0,
        barrier: this.currentAction[5] ?? 0,
        trench: this.currentAction[6] ?? 0,
      },
      abilityExecutions: { ...this.abilityExecutions },
      lastStepReward: this.lastStepReward,
      totalAccumulatedReward: this.totalAccumulatedReward,
      stepCount: this.totalStepCount,
      rolloutEpisode: this.rolloutEpisode,
      rewardVariant,
      recentLossHistory: [...this.recentLossHistory],
      usePER: this.usePER,
      perAlpha: this.perAlpha,
      perBeta: this.perBeta,
      useNStep: this.useNStep,
      nStep: this.nStep,
      meanTDError: this.meanTDError,
      maxTDError: this.maxTDError,
      curriculumStage: this.curriculumStage,
      pbrsShapingReward: this.lastPBRSShapingReward,
      enclosedWindingArea: this.enclosedWindingArea,
      domainRandomizationEnabled: this.domainRandomization,
      layerNormEnabled: this.useLayerNorm,
      radarSectorDensities: [...this.radarSectorDensities],
    };
  }
}

