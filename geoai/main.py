# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from geoai.config import Config, Heterogeneity, Exogenous, Rates, Weights
from geoai.plotting import (
    plot_mechanics_dashboard,
    plot_family_comparison_enhanced,
    plot_strategic_drivers_improved, 
    plot_fog_of_war_dashboard,        
    plot_network_breakdown,        
    plot_persona_evolution,
    plot_blockade_showcase
)
from geoai.policies import (
    policy_rational_planning, 
    policy_family1_baseline, 
    policy_strategic_personas,
    PLANNER_LOGS
)

from geoai.scenarios import (
    run_semiconductor_stranglehold
)

def create_empirically_grounded_config() -> Config:
    actors = ["US","EU","CN","IND","BRA","ROW","UK","JPN","KOR","AUS","CAN", "RUS", "KSA"]

    # --- 1. Initial Conditions (2025 Real-World Baselines) ---
    # Normalized roughly to US_2020 = 1.0 for model stability.
    
    # C0: Compute Stock (Data Center Capacity + Accelerator Inventory)
    # Proxy: Operational Data Center Capacity (GW) & H100 Equivalent Inventory.
    # Source: Visual Capitalist (July 2025), TRG Datacenters 2025.
    # Logic: US (53.7 GW) is the baseline (3.20). China (31.9 GW) would be ~1.9, 
    # but is penalized to 1.80 due to export controls (H100 bans) degrading effective utility.
    C0 = {
        "US": 3.20, # 53.7 GW Capacity, ~40M H100 Equivalents (TRG 2025)
        "CN": 1.80, # 31.9 GW Capacity, but constrained by legacy lithography (Ascend vs Nvidia gap)
        "EU": 0.70, # 11.9 GW Capacity (Visual Capitalist 2025)
        "JPN": 0.30, # ~3.0 GW (includes KOR in some agg, but approx separated here)
        "UK":  0.25, # ~2.6 GW 
        "IND": 0.25, # ~3.6 GW (Rapid growth, approx equal to UK/JPN tier)
        "KOR": 0.20, # ~1.5 GW
        "AUS": 0.15, # ~1.6 GW
        "CAN": 0.15, # ~1.5 GW
        "BRA": 0.12, # ~0.9 GW
        "RUS": 0.15, # Sanctioned but resource-rich
        "KSA": 0.12, # Cash-rich, buying compute aggressively
        "ROW": 0.05
    }

    # E0: Energy Infrastructure (Total Generation TWh + Grid Stability)
    # Source: IEA Electricity Review 2025 (China 9441 TWh vs US 4270 TWh).
    # Logic: China's generation is massive (2.2x US), but coal-heavy and industrially fragmented.
    E0 = {
        "CN": 4.00, # 9441 TWh. Massive capacity advantage.
        "US": 1.80, # 4270 TWh. Mature but aging grid.
        "EU": 1.10, # ~2700 TWh. Green focus but high cost.
        "IND": 0.70, # 1600 TWh. High growth, reliability issues.
        "RUS": 0.60, # Massive energy reserves
        "JPN": 0.40, # ~1000 TWh
        "BRA": 0.30, # ~700 TWh (Hydro dominant)
        "CAN": 0.25, # ~600 TWh
        "KOR": 0.25, # ~600 TWh
        "UK":  0.15, # ~300 TWh
        "AUS": 0.12, # ~270 TWh
        "KSA": 0.25,
        "ROW": 0.10
    }

    # D0: Talent & Data Stock (Top-Tier AI Researchers)
    # Source: MacroPolo Global AI Talent Tracker 2024.
    # Logic: US hosts ~60% of top-tier talent. China hosts ~12%.
    # US is set to 3.0. China should be ~0.6-0.8. UK/EU are the next tier.
    D0 = {
        "US": 3.00, # Hosts 60% of top-tier researchers.
        "CN": 0.80, # Hosts 12%, but high volume of mid-tier engineers.
        "EU": 0.40, # Combined EU hosts ~6-8%.
        "UK": 0.40, # Hosts ~4% (DeepMind/Oxbridge concentration).
        "RUS": 0.25,
        "CAN": 0.30, # Strong history (Hinton/Bengio), ~3-4%.
        "JPN": 0.25, 
        "KOR": 0.20, 
        "IND": 0.20, # High "Origin" (exporting talent), low "Location" (retention is improving).
        "AUS": 0.15, 
        "BRA": 0.10, 
        "KSA": 0.05, # Importing talent, low indigenous
        "ROW": 0.10
    }
    
    weights = Weights()
    rates = Rates()

    # --- Heterogeneity ---
    het = Heterogeneity(
        # Investment Efficiency (Beta_C)
        # US: Deepest capital markets (VC $109B in 2024).
        # CN: State-led ($47.5B "Big Fund III"), historically lower capital efficiency.
        beta_C={"US":0.14, "CN":0.09, "EU":0.07, "UK": 0.08, "JPN":0.07, "IND": 0.08}, 
        
        # Governance Friction (Kappa_G)
        # EU: AI Act (High compliance cost).
        # US: Executive Orders (Medium).
        # CN: Cyberspace Administration (High control, but targeted).
        kappa_G={"EU":0.65, "US":0.30, "CN":0.45, "UK":0.30},
        
        # Insolvency Threshold (Debt/GDP Limit for "Crisis")
        # Source: IMF General Government Gross Debt 2024.
        # US (125%), JPN (250%), CN (96% officially, higher with LGFV).
        insolvency_threshold={
            "US": 4.50, # Reserve currency privilege.
            "JPN": 5.50, # Domestic savings cushion allows >250% debt.
            "CN": 3.00, # High internal debt, but state-owned banking buffers.
            "EU": 2.20, # Maastricht legacy (~90% avg debt).
            "UK": 2.50, # ~100% Debt/GDP.
            "BRA": 1.50, # ~92% Debt/GDP, lower tolerance.
            "IND": 1.50, # ~82% Debt/GDP.
            "ROW": 1.20
        },

        # Panic Thresholds (Sensitivity to Compute Gaps)
        # CN is hypersensitive to the "Chip War".
        panic_threshold={
            "US":1.05, "CN":1.30, "EU":1.0,
            "RUS": 1.40, # Paranoid / Security-focused
            "KSA": 0.90  # Low tolerance, will buy capabilities quickly
        },
        
        # Tax Rates (Effective Corporate Tax)
        # Source: KPMG 2024 & OECD.
        tax_rate={
            "US": 0.26, "EU": 0.28, "CN": 0.25, "UK": 0.25, 
            "RUS": 0.20, # Flat tax regimes
            "KSA": 0.05, # Low/No corporate tax environments
            "BRA": 0.34, "IND": 0.25, "JPN": 0.30
        }
    )

    return Config(
        actors=actors,
        T=40, 
        seed=21,
        weights=weights,
        rates=rates,
        C0=C0, D0=D0, E0=E0, 
        # Governance Baseline: 2024 saw 21.3% rise in AI legislative mentions globally (Stanford Index)
        G0_sec={k: 0.60 for k in actors}, 
        A0_edges=[
            ("IND","BRA"),("CN","ROW"),("CN","IND"),("CN","BRA"), # BRICS+
            ("US","EU"),("US","UK"),("US","CAN"),("US","AUS"),("US","JPN"),("US","KOR"), # NATO/Five Eyes
            ("UK","CAN"),("UK","AUS"),("JPN","AUS"),("JPN","KOR"),("EU","UK"),
            ("IND","US"),("AUS","IND") # Quad
        ],
        X0=None, 
        het=het,
        gpaith_enabled=True,
        # Structural "Fog of War"
        # US underestimates CN indigenous chip progress (Huawei/SMIC yields).
        # CN overestimates US "secret" military AI programs.
        sigma_self_map={"US":0.01,"EU":0.01,"CN":0.08},
        sigma_ally_map={"US":0.02,"EU":0.02},
        sigma_rival_map={"US":0.25,"EU":0.20,"CN":0.35}, 
    )

def run_rationality_check(base_cfg: Config):
    from geoai.engine import GeoAIGame
    print("\n>>> Running Rationality Check: Fixed Reactive vs Optimized Strategic <<<")
    PLANNER_LOGS.clear()
    print("  > Simulating Reactive Agent...")
    game1 = GeoAIGame(deepcopy(base_cfg))
    df1 = game1.run(policy=policy_family1_baseline)
    
    print("  > Simulating Strategic Planner (Robust Mode)...")
    PLANNER_LOGS.clear()
    game2 = GeoAIGame(deepcopy(base_cfg))
    df2 = game2.run(policy=policy_strategic_personas)
    
    results = {
        "Reactive (Fixed)": {"cfg": deepcopy(base_cfg), "df": df1},
        "Strategic (Robust)": {"cfg": deepcopy(base_cfg), "df": df2}
    }
    
    print("  > Generating plots...")
    plot_family_comparison_enhanced(results, "Rationality Check: Solvency & Planning")
    plt.savefig('figures/rationality_check_comparison.png', dpi=100)
    
    print("  > Plotting Improved Mechanics...")
    plot_mechanics_dashboard(df2, deepcopy(base_cfg), title="Strategic Planner Mechanics (Robust)")
    plt.savefig('figures/rationality_check_mechanics.png', dpi=100)

    print("  > Plotting Clean Utility Drivers (Improved)...")
    # Change function call here:
    fig_util = plot_strategic_drivers_improved(PLANNER_LOGS, actors=None) 
    if fig_util: fig_util.savefig('figures/rationality_check_utility.png', dpi=100)
    
    print("  > Plotting Intelligence Matrix...")
    fig_fog = plot_fog_of_war_dashboard(df2, deepcopy(base_cfg))
    if fig_fog: fig_fog.savefig('figures/rationality_check_fog_dashboard.svg', transparent = True)
    
    print("  > Plotting Network Breakdown...")
    fig_net = plot_network_breakdown(df2, deepcopy(base_cfg))
    if fig_net: fig_net.savefig('figures/rationality_check_network.png', dpi=100)

    print("  > Plotting Persona Evolution...")
    fig_pers = plot_persona_evolution(PLANNER_LOGS, deepcopy(base_cfg), title="Adaptive Strategies")
    if fig_pers:
        fig_pers.savefig('figures/rationality_check_personas.png', dpi=100)
    
    PLANNER_LOGS.clear()

    print("\n>>> Running Scenario Showcase: The Semiconductor Stranglehold <<<")
    results_shock = run_semiconductor_stranglehold(base_cfg)
    
    print("  > Plotting Blockade Showcase...")
    fig_block = plot_blockade_showcase(results_shock)
    if fig_block: fig_block.savefig('figures/showcase_stranglehold_detail.png', dpi=100)

    print(">>> Done. Check 'figures/' folder.")

if __name__ == "__main__":
    base_cfg = create_empirically_grounded_config()
    run_rationality_check(base_cfg)

# main.py
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from geoai.engine import GeoAIGame
from geoai.policies import policy_strategic_personas, reset_planners
from geoai.config import Config
from geoai.plotting import set_publication_style, get_color

def run_with_tracking(cfg):
    game = GeoAIGame(cfg)
    history_allies = []
    history_network_bonus = []
    
    # [IMPORTANT] Reset global state
    reset_planners()
    np.random.seed(cfg.seed)
    
    print("Running simulation with network tracking...")
    for t in range(cfg.T):
        game.step(t, policy=policy_strategic_personas)
        
        # Snapshot Allies
        step_allies = {a: set(game.state[a].allies) for a in cfg.actors}
        history_allies.append(step_allies)
        
        # Snapshot Bonus
        total_K = sum(game.state[a].K for a in cfg.actors) + 1e-9
        step_bonus = {}
        for a in cfg.actors:
            ally_power = sum(game.state[ally].K for ally in game.state[a].allies)
            step_bonus[a] = 0.05 * (ally_power / total_K)
        history_network_bonus.append(step_bonus)
        
    df = pd.DataFrame([pd.Series(l.__dict__) for l in game.logs])
    return df, history_allies, history_network_bonus

def plot_alliance_dashboard(df, history_allies, history_network_bonus, cfg):
    set_publication_style()
    
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.2], wspace=0.15, hspace=0.3)
    
    # --- A. Network Snapshots (Fixed Layout) ---
    times = np.linspace(0, cfg.T-1, 4, dtype=int)
    
    G_total = nx.Graph()
    G_total.add_nodes_from(cfg.actors)
    G_total.add_edge("US", "CN", weight=0.0) # Force separation
    fixed_pos = nx.spring_layout(G_total, k=1.2, seed=42, iterations=100)

    for i, t in enumerate(times):
        ax = fig.add_subplot(gs[0, i])
        G = nx.Graph(); G.add_nodes_from(cfg.actors)
        snapshot = history_allies[t]
        edges = set()
        for a, allies in snapshot.items():
            for b in allies:
                u, v = sorted((a, b))
                edges.add((u, v))
        G.add_edges_from(edges)
        
        caps = df[df['t'] == t].set_index('actor')['Cap']
        node_sizes = [caps.get(n, 0.1) * 300 + 100 for n in G.nodes()]
        colors = [get_color(n) for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, fixed_pos, node_size=node_sizes, node_color=colors, alpha=1.0, ax=ax, edgecolors='white', linewidths=1.5)
        nx.draw_networkx_edges(G, fixed_pos, width=2.5, alpha=0.5, edge_color="#34495E", ax=ax)
        nx.draw_networkx_labels(G, fixed_pos, font_color='white', font_weight='bold', font_size=9, ax=ax)
        
        ax.set_title(f"Q{t} (Year {t//4})", fontweight='bold', fontsize=16, color="#2C3E50")
        ax.axis('off')

    # --- B. Diplomatic Centrality Heatmap ---
    ax_heat = fig.add_subplot(gs[1, :2])
    centrality_data = []
    for t in range(cfg.T):
        row = {a: len(history_allies[t][a]) for a in cfg.actors}
        row['t'] = t
        centrality_data.append(row)
    
    heat_df = pd.DataFrame(centrality_data).set_index('t').T
    sorted_actors = heat_df.sum(axis=1).sort_values(ascending=False).index
    heat_df = heat_df.loc[sorted_actors]
    
    sns.heatmap(heat_df, cmap="Blues", ax=ax_heat, cbar_kws={'label': 'Count of Allies'},
                linewidths=0.5, linecolor='white', vmin=0, vmax=3)
    
    ax_heat.set_title("Diplomatic Centrality: Who is Isolating Whom?", fontweight='bold', fontsize=18, loc='left')
    ax_heat.set_xlabel("Time (Quarters)", fontsize=14)

    # --- C. Network Effect Value ---
    ax_line = fig.add_subplot(gs[1, 2:])
    bonus_df = pd.DataFrame(history_network_bonus)
    active = bonus_df.columns[bonus_df.max() > 0.001]
    
    for a in active:
        color = get_color(a)
        lw = 4.0 if a in ["US", "CN", "EU"] else 1.5
        alpha = 1.0 if a in ["US", "CN", "EU"] else 0.6
        ax_line.plot(bonus_df.index, bonus_df[a] * 100, linewidth=lw, label=a, color=color, alpha=alpha)
        
    ax_line.set_title("Economic Value of Alliances (Tech Transfer)", fontweight='bold', fontsize=18, loc='left')
    ax_line.set_ylabel("Efficiency Bonus (%)", fontsize=14)
    ax_line.set_xlabel("Time (Quarters)", fontsize=14)
    ax_line.legend(loc='upper left', ncol=2, fontsize=10, frameon=False)
    
    fig.suptitle("Global Alliance Dynamics & Network Effects", fontsize=26, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/alliance_dashboard.png', dpi=150)

if __name__ == "__main__":
    cfg = create_empirically_grounded_config()
    df, allies, bonus = run_with_tracking(cfg)
    plot_alliance_dashboard(df, allies, bonus, cfg)