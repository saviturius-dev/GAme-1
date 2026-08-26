import {
  RLAction,
  RLAltitudeState,
  RLBarrierState,
  RLCooldownState,
  RLDensityState,
  RLHealthState,
  RLProximityState,
  QTableRecord,
  RLAgentStats,
  SurvivalEpisodeRecord,
} from '../types';

export const RL_ACTIONS: RLAction[] = [
  'evade_kite',
  'aerial_soar',
  'ground_dive',
  'altitudinal_corkscrew',
  'blast_repel',
  'carve_trench',
  'tactical_dash',
  'flank_reposition',
];

export class RedSurvivalRLAgent {
  public actions: RLAction[] = RL_ACTIONS;
  public qTable: Record<string, Record<RLAction, number>> = {};
  public alpha: number = 0.12; // Learning rate
  public gamma: number = 0.92; // Discount factor
  public epsilon: number = 0.35; // Exploration probability
  public epsilonDecay: number = 0.996; // Decay per step
  public minEpsilon: number = 0.03; // Minimum exploration floor
  public totalReward: number = 0;
  public lastReward: number = 0;
  public lastAction: RLAction | null = null;
  public lastStateKey: string = 'far_low_ready_exposed_healthy_mid';
  public isExploring: boolean = false;
  public decisionCount: number = 0;
  public episodesCount: number = 1;
  public bestSurvivalTime: number = 0;
  public recentEpisodes: SurvivalEpisodeRecord[] = [];

  public actionCounts: Record<RLAction, number> = {
    evade_kite: 0,
    aerial_soar: 0,
    ground_dive: 0,
    altitudinal_corkscrew: 0,
    blast_repel: 0,
    carve_trench: 0,
    tactical_dash: 0,
    flank_reposition: 0,
  };

  constructor(initialEpsilon: number = 0.35, alpha: number = 0.12, gamma: number = 0.92) {
    this.epsilon = initialEpsilon;
    this.alpha = alpha;
    this.gamma = gamma;
    this.bootstrapAllStates();
    this.loadFromLocalStorage();
  }

  // Pre-populate all 216 discrete 3D states for immediate convergence and exploration
  public bootstrapAllStates() {
    const proximities: RLProximityState[] = ['close', 'mid', 'far'];
    const densities: RLDensityState[] = ['low', 'high', 'critical'];
    const cooldowns: RLCooldownState[] = ['ready', 'cooling'];
    const barriers: RLBarrierState[] = ['shielded', 'exposed'];
    const healths: RLHealthState[] = ['critical', 'healthy'];
    const altitudes: RLAltitudeState[] = ['ground', 'mid', 'aerial'];

    for (const prox of proximities) {
      for (const den of densities) {
        for (const cd of cooldowns) {
          for (const bar of barriers) {
            for (const hp of healths) {
              for (const alt of altitudes) {
                const key = `${prox}_${den}_${cd}_${bar}_${hp}_${alt}`;
                if (!this.qTable[key]) {
                  this.qTable[key] = {
                    evade_kite: 0.0,
                    aerial_soar: (prox === 'close' || den === 'critical') && alt !== 'aerial' ? 0.6 : 0.0,
                    ground_dive: alt === 'aerial' && den === 'low' ? 0.3 : 0.0,
                    altitudinal_corkscrew: prox === 'close' ? 0.5 : 0.0,
                    blast_repel: cd === 'ready' && prox === 'close' ? 1.0 : 0.0,
                    carve_trench: alt === 'ground' && prox === 'close' ? 0.4 : 0.0,
                    tactical_dash: prox === 'close' && hp === 'critical' ? 0.8 : 0.0,
                    flank_reposition: 0.0,
                  };
                }
              }
            }
          }
        }
      }
    }
  }

  /**
   * Discretize continuous 3D environment metrics into a discrete state key:
   * - Proximity: close (< 3.5m), mid (3.5 - 8.0m), far (> 8.0m)
   * - Density: low (< 12 boids within 7m), high (12 - 40), critical (> 40)
   * - Cooldown: ready (repel available) vs cooling
   * - Barrier: shielded (barrier between player & swarm centroid) vs exposed
   * - Health: critical (< 40%) vs healthy (>= 40%)
   * - Altitude: ground (Y < 3.0m), mid (3.0 - 12.0m), aerial (Y > 12.0m)
   */
  public getStateKey(
    nearestBoidDist: number,
    localBoidsCount: number,
    isRepelReady: boolean,
    isShieldedByBarrier: boolean,
    healthRatio: number,
    playerAltitude: number = 1.0
  ): {
    stateKey: string;
    proximityState: RLProximityState;
    densityState: RLDensityState;
    cooldownState: RLCooldownState;
    barrierState: RLBarrierState;
    healthState: RLHealthState;
    altitudeState: RLAltitudeState;
  } {
    const proximityState: RLProximityState =
      nearestBoidDist < 3.5 ? 'close' : nearestBoidDist <= 8.0 ? 'mid' : 'far';
    const densityState: RLDensityState =
      localBoidsCount < 12 ? 'low' : localBoidsCount <= 40 ? 'high' : 'critical';
    const cooldownState: RLCooldownState = isRepelReady ? 'ready' : 'cooling';
    const barrierState: RLBarrierState = isShieldedByBarrier ? 'shielded' : 'exposed';
    const healthState: RLHealthState = healthRatio < 0.4 ? 'critical' : 'healthy';
    const altitudeState: RLAltitudeState =
      playerAltitude < 3.0 ? 'ground' : playerAltitude <= 12.0 ? 'mid' : 'aerial';

    const stateKey = `${proximityState}_${densityState}_${cooldownState}_${barrierState}_${healthState}_${altitudeState}`;
    this.ensureStateExists(stateKey);

    return {
      stateKey,
      proximityState,
      densityState,
      cooldownState,
      barrierState,
      healthState,
      altitudeState,
    };
  }

  private ensureStateExists(stateKey: string) {
    if (!this.qTable[stateKey]) {
      this.qTable[stateKey] = {
        evade_kite: 0.0,
        aerial_soar: 0.0,
        ground_dive: 0.0,
        altitudinal_corkscrew: 0.0,
        blast_repel: 0.0,
        carve_trench: 0.0,
        tactical_dash: 0.0,
        flank_reposition: 0.0,
      };
    }
  }

  /**
   * Epsilon-greedy action selection for Red Survivor:
   * With probability epsilon: explores random action.
   * Otherwise exploits maximum Q(s, a).
   */
  public chooseAction(stateKey: string): { action: RLAction; isExploring: boolean } {
    this.ensureStateExists(stateKey);

    let chosenAction: RLAction;
    let exploring = false;

    if (Math.random() < this.epsilon) {
      // Exploration: pick uniformly at random
      const randIdx = Math.floor(Math.random() * this.actions.length);
      chosenAction = this.actions[randIdx];
      exploring = true;
    } else {
      // Exploitation: pick action with highest Q-value
      const qValues = this.qTable[stateKey];
      let maxQ = -Infinity;
      const bestActions: RLAction[] = [];

      for (const act of this.actions) {
        const val = qValues[act];
        if (val > maxQ) {
          maxQ = val;
          bestActions.length = 0;
          bestActions.push(act);
        } else if (Math.abs(val - maxQ) < 1e-5) {
          bestActions.push(act);
        }
      }

      chosenAction = bestActions[Math.floor(Math.random() * bestActions.length)] || 'evade_kite';
      exploring = false;
    }

    this.lastAction = chosenAction;
    this.lastStateKey = stateKey;
    this.isExploring = exploring;
    this.actionCounts[chosenAction] = (this.actionCounts[chosenAction] || 0) + 1;
    this.decisionCount++;

    return { action: chosenAction, isExploring: exploring };
  }

  /**
   * Q-Learning update:
   * Q(s, a) = Q(s, a) + alpha * [ R + gamma * max_a' Q(s', a') - Q(s, a) ]
   */
  public learn(
    stateKey: string,
    action: RLAction,
    reward: number,
    nextStateKey: string
  ): { deltaQ: number; newQ: number } {
    this.ensureStateExists(stateKey);
    this.ensureStateExists(nextStateKey);

    const oldQ = this.qTable[stateKey][action];
    const nextMaxQ = Math.max(...Object.values(this.qTable[nextStateKey]));

    const deltaQ = this.alpha * (reward + this.gamma * nextMaxQ - oldQ);
    const newQ = oldQ + deltaQ;
    this.qTable[stateKey][action] = newQ;

    // Decay exploration factor over time
    if (this.epsilon > this.minEpsilon) {
      this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    }

    this.totalReward += reward;
    this.lastReward = reward;

    return { deltaQ, newQ };
  }

  public recordEpisodeEnd(record: SurvivalEpisodeRecord) {
    if (record.survivalTime > this.bestSurvivalTime) {
      this.bestSurvivalTime = record.survivalTime;
    }
    this.recentEpisodes.unshift(record);
    if (this.recentEpisodes.length > 20) {
      this.recentEpisodes.pop();
    }
    this.episodesCount++;
  }

  public resetQTable() {
    this.qTable = {};
    this.totalReward = 0;
    this.lastReward = 0;
    this.decisionCount = 0;
    this.episodesCount = 1;
    this.bestSurvivalTime = 0;
    this.recentEpisodes = [];
    this.epsilon = 0.35;
    this.actionCounts = {
      evade_kite: 0,
      aerial_soar: 0,
      ground_dive: 0,
      altitudinal_corkscrew: 0,
      blast_repel: 0,
      carve_trench: 0,
      tactical_dash: 0,
      flank_reposition: 0,
    };
    this.bootstrapAllStates();
  }

  public getQTableRecords(): QTableRecord[] {
    this.bootstrapAllStates();
    const records: QTableRecord[] = [];

    for (const [stateKey, actions] of Object.entries(this.qTable)) {
      const parts = stateKey.split('_');
      const proximityState = (parts[0] || 'far') as RLProximityState;
      const densityState = (parts[1] || 'low') as RLDensityState;
      const cooldownState = (parts[2] || 'ready') as RLCooldownState;
      const barrierState = (parts[3] || 'exposed') as RLBarrierState;
      const healthState = (parts[4] || 'healthy') as RLHealthState;
      const altitudeState = (parts[5] || 'mid') as RLAltitudeState;

      let bestAction: RLAction = 'evade_kite';
      let bestVal = -Infinity;
      for (const act of this.actions) {
        if (actions[act] > bestVal) {
          bestVal = actions[act];
          bestAction = act;
        }
      }

      records.push({
        stateKey,
        proximityState,
        densityState,
        cooldownState,
        barrierState,
        healthState,
        altitudeState,
        actions: { ...actions },
        bestAction,
        visitCount:
          (this.actionCounts.evade_kite +
            this.actionCounts.aerial_soar +
            this.actionCounts.ground_dive +
            this.actionCounts.altitudinal_corkscrew +
            this.actionCounts.blast_repel +
            this.actionCounts.carve_trench +
            this.actionCounts.tactical_dash +
            this.actionCounts.flank_reposition) || 0,
      });
    }

    const proxOrder = { close: 0, mid: 1, far: 2 };
    const denOrder = { critical: 0, high: 1, low: 2 };
    const cdOrder = { ready: 0, cooling: 1 };
    const barOrder = { exposed: 0, shielded: 1 };
    const hpOrder = { critical: 0, healthy: 1 };
    const altOrder = { aerial: 0, mid: 1, ground: 2 };

    return records.sort((a, b) => {
      if (proxOrder[a.proximityState] !== proxOrder[b.proximityState]) {
        return proxOrder[a.proximityState] - proxOrder[b.proximityState];
      }
      if (denOrder[a.densityState] !== denOrder[b.densityState]) {
        return denOrder[a.densityState] - denOrder[b.densityState];
      }
      if (cdOrder[a.cooldownState] !== cdOrder[b.cooldownState]) {
        return cdOrder[a.cooldownState] - cdOrder[b.cooldownState];
      }
      if (barOrder[a.barrierState] !== barOrder[b.barrierState]) {
        return barOrder[a.barrierState] - barOrder[b.barrierState];
      }
      if (hpOrder[a.healthState] !== hpOrder[b.healthState]) {
        return hpOrder[a.healthState] - hpOrder[b.healthState];
      }
      return (altOrder[a.altitudeState || 'mid'] || 0) - (altOrder[b.altitudeState || 'mid'] || 0);
    });
  }

  public getStats(enabled: boolean = true): RLAgentStats {
    return {
      enabled,
      epsilon: this.epsilon,
      alpha: this.alpha,
      gamma: this.gamma,
      totalReward: this.totalReward,
      lastReward: this.lastReward,
      lastAction: this.lastAction,
      lastStateKey: this.lastStateKey,
      isExploring: this.isExploring,
      decisionCount: this.decisionCount,
      episodesCount: this.episodesCount,
      qTable: JSON.parse(JSON.stringify(this.qTable)),
      actionCounts: { ...this.actionCounts },
      bestSurvivalTime: this.bestSurvivalTime,
      recentEpisodes: [...this.recentEpisodes],
    };
  }

  public exportModelJSON(): any {
    return {
      version: 1,
      modelType: 'TABULAR_Q_LEARNING',
      timestamp: Date.now(),
      savedAt: new Date().toISOString(),
      epsilon: this.epsilon,
      alpha: this.alpha,
      gamma: this.gamma,
      totalReward: this.totalReward,
      decisionCount: this.decisionCount,
      episodesCount: this.episodesCount,
      bestSurvivalTime: this.bestSurvivalTime,
      qTable: this.qTable,
      actionCounts: this.actionCounts,
      recentEpisodes: this.recentEpisodes.slice(-20),
    };
  }

  public importModelJSON(data: any): boolean {
    try {
      if (!data || typeof data !== 'object') return false;
      if (data.qTable && typeof data.qTable === 'object') {
        this.qTable = { ...data.qTable };
      }
      if (typeof data.epsilon === 'number') this.epsilon = data.epsilon;
      if (typeof data.alpha === 'number') this.alpha = data.alpha;
      if (typeof data.gamma === 'number') this.gamma = data.gamma;
      if (typeof data.totalReward === 'number') this.totalReward = data.totalReward;
      if (typeof data.decisionCount === 'number') this.decisionCount = data.decisionCount;
      if (typeof data.episodesCount === 'number') this.episodesCount = data.episodesCount;
      if (typeof data.bestSurvivalTime === 'number') this.bestSurvivalTime = data.bestSurvivalTime;
      if (data.actionCounts) this.actionCounts = { ...data.actionCounts };
      if (Array.isArray(data.recentEpisodes)) this.recentEpisodes = data.recentEpisodes;
      return true;
    } catch (err) {
      console.warn('Failed to parse QTable model JSON:', err);
      return false;
    }
  }

  public saveToLocalStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return false;
      const json = JSON.stringify(this.exportModelJSON());
      window.localStorage.setItem('ursina_qtable_model', json);
      return true;
    } catch (e) {
      return false;
    }
  }

  public loadFromLocalStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return false;
      const raw = window.localStorage.getItem('ursina_qtable_model');
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
        window.localStorage.removeItem('ursina_qtable_model');
      }
    } catch (e) {}
  }
}
