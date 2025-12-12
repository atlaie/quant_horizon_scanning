# Quantitative horizon scanning: a Proof of Concept

## Overview

We present a discrete-time, agent-based simulation framework designed to model the geopolitical dynamics of Artificial Intelligence development. It simulates the competition between major state actors (e.g., US, China, EU) under constraints of physical infrastructure (Compute, Energy), human capital (Talent), and economic limitations (Debt, GDP).

The framework integrates a **Principled Physics Engine** (based on Cobb-Douglas production functions) with **Game Theoretic Planning** (strategic agents optimizing for utility under uncertainty).

## Key Features

  * **Macro-Structural Physics:** Models AI capability ($K$) as a function of Compute ($C$), Data/Talent ($D$), Energy ($E$), and Governance ($G$). Includes depreciation, investment lags (e.g., energy infrastructure delays), and Leontief constraints (bottlenecks).
  * **Geopolitical Actions:** Actors can perform offensive and defensive moves, including:
      * **Investment:** Allocating GDP to $C$, $D$, or $E$.
      * **Sanctions & Blockades:** Targeted export bans on critical supply chain chokepoints (Lithography, EDA, HBM).
      * **Espionage:** Covert theft of talent/IP or sabotage of infrastructure.
      * **Alliances:** Forming coalitions to share technology and reduce "Fog of War."
  * **Strategic Planners:** Agents utilize a "Shadow Model" to simulate future trajectories and rival responses, optimizing for a utility function that balances Growth, Relative Power, Fiscal Health, and Political Stability.
  * **Fog of War:** Implements asymmetric information where actors act based on *perceived* capabilities of rivals, subject to noise ($\sigma$) and bias ($\mu$).
  * **Economic Reality:** Tracks sovereign debt, yield curves, and insolvency risks. High debt increases interest rates, potentially leading to a "death spiral" or forced austerity.

## Project Structure

```text
quant_horizon_scanning/
├── main.py                 # Entry point: Configuration setup and scenario execution
├── geoai/
│   ├── analytics.py        # Post-processing metrics (Leontief ratios, HHI, utilization)
│   ├── config.py           # Data classes for simulation parameters (Weights, Rates, Heterogeneity)
│   ├── constants.py        # Action definitions (INVEST_C, SANCTION_RIVAL, etc.)
│   ├── engine.py           # Core simulation logic (GeoAIGame class)
│   ├── plotting.py         # Visualization tools (Matplotlib/Seaborn dashboards)
│   ├── policies.py         # Agent logic (StrategicPlanner, ShadowModel, Heuristics)
│   ├── scenarios.py        # Preset configurations (Blockades, High Stakes, Governance)
│   ├── state.py            # Data classes for ActorState and StepLog
│   └── utils.py            # Math helpers (sigmoid, soft_min, gini)
└── figures/                # Output directory for simulation plots
```

## Installation & Requirements

The framework requires Python 3.10+ and the following scientific computing libraries:

```bash
pip install numpy pandas matplotlib seaborn networkx
```

## Usage

To run the standard simulation suite (Rationality Check and Semiconductor Stranglehold):

```bash
python main.py
```

This will execute the simulations defined in `main.py` and generate visualizations in the `figures/` directory.

## Simulation Mechanics

### 1\. The Production Function

AI Capability ($K$) is generated via a modified Cobb-Douglas function, modulated by Governance ($G$) and Safety constraints:

$$K_{t+1} = (1-\delta)K_t + \left( A_t \cdot \text{eff}(C_t)^\alpha \cdot D_t^\beta \cdot S_t^\gamma \right) \cdot \text{Spillover}$$

Where:

  * $\text{eff}(C_t)$ is utilized compute, constrained by Energy availability ($E_t$) and supply chain blockades.
  * $S_t$ represents Safety/Stability.
  * Spillover includes knowledge diffusion and technology transfers from allies.

### 2\. The Agent Model (`policies.py`)

The `StrategicPlanner` class implements bounded rationality. At each step $t$:

1.  **Observe:** The agent estimates the state of the world (potentially noisy).
2.  **Weight Adaptation:** Preferences (e.g., `w_growth`, `w_safety`) shift based on environmental stress (Debt crisis, high race intensity).
3.  **Shadow Simulation:** The agent projects the game $N$ steps forward for various action candidates (Invest, Sanction, Ally).
4.  **Decision:** It selects the action maximizing expected utility using a Softmax function.

### 3\. Supply Chains (`engine.py`)

The model tracks specific semiconductor chokepoints defined in `config.py`:

  * **EDA (Electronic Design Automation)**
  * **LITHO (Lithography Equipment)**
  * **HBM (High Bandwidth Memory)**
  * **CLOUD (Compute Access)**

Sanctions (e.g., `BAN` or `LICENSE_FEE`) increase costs or throttle the effective depreciation rate of hardware (representing spare parts starvation).

## Scenarios

### 1\. Rationality Check

Compares a **Reactive Agent** (heuristic-based) against a **Strategic Planner** (optimization-based).

  * **Output:** `figures/rationality_check_comparison.png`
  * **Goal:** Demonstrate that strategic planning leads to better long-term solvency and capability maintenance than greedy heuristics.

### 2\. The Semiconductor Stranglehold

Simulates a coordinated US-EU export blockade against China starting at $t=15$.

  * **Output:** `figures/showcase_stranglehold_detail.png`
  * **Dynamics:** Tests China's ability to pivot to indigenous investment ($D$) versus the crushing weight of hardware depreciation and efficiency loss.

### 3\. Fog of War

Visualizes the divergence between Real Capability ($K$) and Perceived Capability ($\hat{K}$).

  * **Output:** `figures/rationality_check_fog_dashboard.png`
  * **Dynamics:** Shows how intelligence errors accumulate and how alliances can improve observability.

## Configuration

Simulation parameters are centralized in `main.py` via the `create_empirically_grounded_config` function. These have been grounded in real-world data where possible. Key parameters include:

  * `C0`, `D0`, `E0`: Initial stocks for Compute, Talent, Energy.
  * `het`: Heterogeneity parameters (e.g., `beta_C` for capital efficiency, `insolvency_threshold` for debt tolerance).
  * `rates`: Physics constants (depreciation, growth bases, cost multipliers).
