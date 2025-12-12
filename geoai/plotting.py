# geoai/plotting.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List
import math

# --- Professional Color Palette ---
GEO_COLORS = {
    "US": "#2C3E50", # Dark Slate
    "CN": "#C0392B", # Deep Red
    "EU": "#F1C40F", # Gold
    "IND": "#E67E22", # Orange
    "JPN": "#8E44AD", # Purple
    "UK":  "#3498DB", # Blue
    "KOR": "#1ABC9C", # Teal
    "BRA": "#27AE60", # Green
    "CAN": "#D35400", # Pumpkin
    "AUS": "#16A085", # Sea Green
    "ROW": "#95A5A6", # Grey
}

def get_color(actor):
    return GEO_COLORS.get(actor, "#95A5A6") 

def set_publication_style():
    """Sets a clean, professional plotting style with LARGE fonts for small-figure insertion."""
    sns.set_theme(style="white", context="talk") # 'talk' context sets larger defaults than 'paper'
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        # significantly increased font sizes
        "figure.titlesize": 26, 
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "figure.dpi": 150
    })

def plot_mechanics_dashboard(df: pd.DataFrame, cfg, title="Simulation Mechanics"):
    if df.empty: return
    set_publication_style()
    
    # Pre-calc metrics
    from .analytics import compute_leontief_metrics
    df = compute_leontief_metrics(df, cfg)
    df["reserves_to_gdp"] = df["B"] / (df["GDP"] + 1e-9)
    
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # 1. True Capability
    ax0 = fig.add_subplot(gs[0, 0])
    piv = df.pivot_table(index="t", columns="actor", values="Cap")
    
    minors = [c for c in piv.columns if c not in ["US", "CN", "EU"]]
    for m in minors:
        ax0.plot(piv.index, piv[m], color="#BDC3C7", alpha=0.4, linewidth=1.5, zorder=1)
    
    for actor in ["US", "CN", "EU"]:
        if actor in piv.columns:
            ax0.plot(piv.index, piv[actor], color=get_color(actor), linewidth=3.5, label=actor, zorder=10)
    
    ax0.set_title("Global Capability Distribution", fontweight='bold', loc='left')
    ax0.set_ylabel("Capability ($K$)")
    ax0.legend(loc='upper left', ncol=3, fontsize=14)

    # 2. Fiscal Discipline
    ax1 = fig.add_subplot(gs[0, 1])
    piv_res = df.pivot_table(index="t", columns="actor", values="reserves_to_gdp")
    
    for actor in ["US", "CN", "EU"]:
        if actor in piv_res.columns:
            ax1.plot(piv_res.index, piv_res[actor], color=get_color(actor), linewidth=3.0, label=actor)
            
    ax1.axhline(1.0, color="#E74C3C", linestyle=":", linewidth=2.5, alpha=0.7)
    ax1.text(0, 1.05, "Burn Threshold (100% GDP)", color="#E74C3C", fontsize=12)
    
    ax1.set_title("Fiscal Reserves (% GDP)", fontweight='bold', loc='left')

    # 3. Systemic Friction
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_r = ax2.twinx()
    
    global_waste = df.groupby("t")["wasted_C"].sum()
    global_C = df.groupby("t")["C"].sum()
    waste_pct = (global_waste / (global_C + 1e-9)) * 100
    
    ax2.fill_between(waste_pct.index, 0, waste_pct, color="#E74C3C", alpha=0.2)
    ax2.plot(waste_pct.index, waste_pct, color="#C0392B", linewidth=2.5, label="Global Inefficiency")
    ax2.set_ylabel("% Capital Wasted", color="#C0392B")
    ax2.spines['left'].set_color('#C0392B')
    ax2.yaxis.label.set_color('#C0392B')        
    ax2.tick_params(axis='y', colors='#C0392B')
    
    if "belief_error" in df.columns:
        err = df.groupby("t")["belief_error"].mean() * 100
        ax2_r.plot(err.index, err, color="#34495E", linestyle="--", linewidth=2.5, label="Intel Fog")
        ax2_r.set_ylabel("Avg Belief Error (%)", color="#34495E")
    
    ax2.set_title("Systemic Friction: Waste vs. Fog", fontweight='bold', loc='left')

    # 4. Chip Utilization
    ax3 = fig.add_subplot(gs[1, 1])
    piv_util = df.pivot_table(index="t", columns="actor", values="utilization_rate")
    for actor in ["US", "CN", "EU"]:
        if actor in piv_util.columns:
            ax3.plot(piv_util.index, piv_util[actor], color=get_color(actor), linewidth=3.0)
    ax3.set_ylim(0, 105)
    ax3.set_title("Chip Utilization Rate (%)", fontweight='bold', loc='left')

    # 5. Global Spend Mix
    ax4 = fig.add_subplot(gs[2, 0])
    maint = df.groupby("t")["maintenance_cost"].sum()
    invest = df.groupby("t")["action_cost"].sum()
    
    ax4.stackplot(maint.index, maint, invest, labels=["Maintenance", "New Investment"],
                  colors=["#BDC3C7", "#2ECC71"], alpha=0.8)
    ax4.legend(loc='upper left', fontsize=14)
    ax4.set_title("Global Expenditure Breakdown", fontweight='bold', loc='left')

    # 6. Constraint Heatmap
    ax5 = fig.add_subplot(gs[2, 1])
    piv_leo = df.pivot_table(index="t", columns="actor", values="leontief_ratio")
    sorted_actors = piv_leo.iloc[-1].sort_values(ascending=False).index
    
    sns.heatmap(piv_leo[sorted_actors].T, cmap="RdYlBu_r", center=1.0, ax=ax5, 
                cbar_kws={'label': 'Leontief Ratio (Demand/Supply)'},
                vmin=0.0, vmax=2.5)
    ax5.set_title("Constraint Heatmap (Red = Bottlecked)", fontweight='bold', loc='left')

    fig.suptitle(title, fontsize=28, fontweight='bold', y=1.03)
    plt.tight_layout()
    return fig

def plot_blockade_showcase(results: dict):
    set_publication_style()
    
    scenarios = list(results.keys())
    df_base = results[scenarios[0]]["df"] 
    df_block = results[scenarios[1]]["df"] 
    
    from .analytics import compute_leontief_metrics
    df_base = compute_leontief_metrics(df_base, None)
    df_block = compute_leontief_metrics(df_block, None)
    
    actors = ["US", "CN", "EU"]
    fig, axes = plt.subplots(len(actors), 2, figsize=(16, 14))
    BLOCK_START = 14 
    
    for i, actor in enumerate(actors):
        color = get_color(actor)
        
        # --- Left Col: Capability ---
        ax_l = axes[i, 0]
        base_line = df_base[df_base["actor"]==actor].set_index("t")["Cap"]
        block_line = df_block[df_block["actor"]==actor].set_index("t")["Cap"]
        
        ax_l.plot(base_line, color="#95A5A6", linestyle="--", linewidth=2.0, label="Baseline")
        ax_l.plot(block_line, color=color, linewidth=3.5, label="Blockade Scenario")
        
        ax_l.axvspan(BLOCK_START, 39, color="#E74C3C", alpha=0.05)
        ax_l.axvline(BLOCK_START, color="#C0392B", linestyle=":", linewidth=1.5)
        
        ax_l.set_ylabel(f"{actor} Capability", fontweight='bold', fontsize=16)
        if i == 0: 
            ax_l.legend(loc="upper left", fontsize=12)
            ax_l.set_title("Strategic Capability ($K$)", fontweight='bold', fontsize=18)
            ax_l.text(BLOCK_START + 1, base_line.max()*0.1, "Sanctions Start", color="#C0392B", rotation=90, fontsize=12)

        # --- Right Col: Efficiency ---
        ax_r = axes[i, 1]
        base_util = df_base[df_base["actor"]==actor].set_index("t")["utilization_rate"]
        block_util = df_block[df_block["actor"]==actor].set_index("t")["utilization_rate"]
        
        ax_r.plot(base_util, color="#95A5A6", linestyle="--", linewidth=2.0)
        ax_r.plot(block_util, color=color, linewidth=3.5)
        
        ax_r.axvspan(BLOCK_START, 39, color="#E74C3C", alpha=0.05)
        ax_r.set_ylim(0, 105)
        
        if i == 0: ax_r.set_title("Industrial Efficiency (%)", fontweight='bold', fontsize=18)

    fig.suptitle("Scenario Analysis: The Semiconductor Stranglehold", fontsize=24, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig

def plot_strategic_drivers_improved(logs: List[dict], actors=None):
    if not logs: return None
    set_publication_style()
    
    df = pd.DataFrame(logs)
    if actors is None:
        actors = df["actor"].unique().tolist()
    
    COLORS = {
        "u_growth": "#2ECC71", "u_relative": "#1ABC9C", "u_pol": "#3498DB",
        "u_terminal": "#9B59B6", "u_debt": "#E74C3C", "u_waste": "#E67E22", "u_hoard": "#F1C40F"
    }
    
    pos_groups = ["u_growth", "u_relative", "u_pol", "u_terminal"]
    neg_groups = ["u_debt", "u_waste", "u_hoard"]
    
    n = len(actors)
    cols = 4
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), sharex=True)
    axes = axes.flatten()
    
    for i in range(n, len(axes)): axes[i].set_visible(False)

    for i, actor in enumerate(actors):
        ax = axes[i]
        adf = df[df["actor"] == actor].set_index("t")
        if adf.empty: continue
        
        pos_data = adf[pos_groups].clip(lower=0)
        neg_data = adf[neg_groups].clip(upper=0)
        net_utility = adf["u_growth"] + adf["u_relative"] + adf["u_pol"] + \
                      adf["u_debt"] + adf["u_waste"] + adf.get("u_hoard", 0)

        peak_positive = pos_data.sum(axis=1).max()
        if peak_positive == 0: peak_positive = 1.0
        
        y_max = peak_positive * 1.2
        y_min = -1.5 * peak_positive 
        
        if not pos_data.empty:
            ax.stackplot(pos_data.index, pos_data.T, labels=pos_groups, 
                         colors=[COLORS[c] for c in pos_groups], alpha=0.9)
            
        if not neg_data.empty and neg_data.min().min() < 0:
            ax.stackplot(neg_data.index, neg_data.T, labels=neg_groups, 
                         colors=[COLORS[c] for c in neg_groups], alpha=0.9)

        total_neg = neg_data.sum(axis=1)
        crisis_mask = total_neg < (y_min * 0.95)
        
        if crisis_mask.any():
            crisis_starts = crisis_mask.index[crisis_mask & ~crisis_mask.shift(1).fillna(False)]
            for t_start in crisis_starts:
                ax.axvspan(t_start, adf.index.max(), color='#E74C3C', alpha=0.1, zorder=0)
                if t_start == crisis_starts[0]:
                    ax.text(t_start + 0.5, y_min * 0.9, "INSOLVENCY ZONE", 
                            color='#C0392B', fontweight='bold', fontsize=11, ha='left', va='bottom')

        ax.plot(net_utility.index, net_utility, color="#2C3E50", linewidth=3.0, linestyle="-", label="Net Utility", zorder=10)
        
        is_positive = net_utility > 0
        transitions = np.where(np.diff(is_positive.astype(int)) == -1)[0]
        # if len(transitions) > 0:
        #     tp = transitions[0]
        #     if net_utility.iloc[tp+1] < 0:
        #         ax.axvline(tp, color="#2C3E50", linestyle=":", linewidth=2.5, alpha=0.8)
        #         ax.text(tp, y_max * 0.95, "CRITICAL", ha='center', va='top', fontsize=11, color="#2C3E50", fontweight='bold')

        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="black", linewidth=1.5, alpha=0.5)
        ax.set_title(actor, fontweight='bold', loc='left', fontsize=16)
        
        if i % cols == 0:
            ax.set_ylabel("Utility (Scaled)", fontsize=14)

        if i == 0:
            h, l = ax.get_legend_handles_labels()
            net_h = [h[-1]]; net_l = [l[-1]]
            pos_h = h[:len(pos_groups)]; pos_l = l[:len(pos_groups)]
            neg_h = h[len(pos_groups):-1]; neg_l = l[len(pos_groups):-1]
            
            ax.legend(net_h + pos_h + neg_h, net_l + pos_l + neg_l, 
                      loc='lower left', fontsize=12, frameon=True, title="Drivers")

    plt.tight_layout()
    return fig

def plot_fog_of_war_dashboard(df: pd.DataFrame, cfg, title="Global Intelligence & Perception Dashboard"):
    if "perceptions" not in df.columns and "belief_error" not in df.columns: 
        return
    
    set_publication_style()
    fig = plt.figure(figsize=(24, 12)) 
    
    # [FIX] Use 4 columns: [Time Series, Heatmap, Colorbar, Bar Plot]
    # This prevents the heatmap/colorbar from crushing the side bar plot
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.3, 1.0, 0.05, 0.3], wspace=0.1, figure=fig) 
    
    def get_belief_series(observer, target):
        obs_logs = df[df["actor"] == observer].sort_values("t")
        return obs_logs["t"], obs_logs["perceptions"].apply(lambda d: d.get(target, np.nan))

    real_caps = df.pivot(index="t", columns="actor", values="Cap")

    # --- COL 1: Time Series (Left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    t_idx, us_belief_cn = get_belief_series("US", "CN")
    real_cn = real_caps["CN"]
    
    ax1.plot(real_cn.index, real_cn, color="#C0392B", linewidth=3.5, alpha=0.8, label="Real China Capability")
    ax1.plot(t_idx, us_belief_cn, color="#2C3E50", linestyle="--", linewidth=3.0, label="US Perception of CN")
    ax1.fill_between(t_idx, us_belief_cn, real_cn, color="#95A5A6", alpha=0.2, label="Intel Gap (Error)")
    
    ax1.set_title("The View from Washington: US Assessment of China", fontweight='bold', loc='left', fontsize=18)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.set_ylabel("Capability ($K$)")
    ax1.margins(x=0)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    t_idx, cn_belief_us = get_belief_series("CN", "US")
    real_us = real_caps["US"]
    
    ax2.plot(real_us.index, real_us, color="#2C3E50", linewidth=3.5, alpha=0.8, label="Real US Capability")
    ax2.plot(t_idx, cn_belief_us, color="#C0392B", linestyle="--", linewidth=3.0, label="CN Perception of US")
    ax2.fill_between(t_idx, cn_belief_us, real_us, color="#95A5A6", alpha=0.2, label="Intel Gap")
    
    ax2.set_title("The View from Beijing: China Assessment of US", fontweight='bold', loc='left', fontsize=18)
    ax2.legend(loc='upper left', fontsize=14)
    ax2.set_xlabel("Time ($t$)", fontsize=16)
    ax2.set_ylabel("Capability ($K$)")
    ax2.margins(x=0)

    # --- Data Prep for Matrix ---
    actors = cfg.actors
    error_matrix = pd.DataFrame(0.0, index=actors, columns=actors)
    
    for observer in actors:
        obs_df = df[df["actor"] == observer].set_index("t")
        for target in actors:
            if observer == target: continue
            if "perceptions" in obs_df.columns:
                believed = obs_df["perceptions"].apply(lambda d: d.get(target, 0.0))
            else:
                believed = 0.0
            actual = real_caps[target]
            total_error = (believed - actual).abs().sum()
            error_matrix.loc[observer, target] = total_error

    # --- COL 2: Heatmap ---
    ax3 = fig.add_subplot(gs[:, 1])
    
    # --- COL 3: Colorbar (Explicit Axis) ---
    cbar_ax = fig.add_subplot(gs[:, 2])

    sns.heatmap(error_matrix, ax=ax3, cbar_ax=cbar_ax, 
                annot=True, fmt=".0f", cmap="Reds", 
                cbar_kws={'label': 'Cumulative Intelligence Error ($\int |K - \hat{K}| dt$)'},
                linewidths=0.5, linecolor='white', annot_kws={"size": 12})
    
    ax3.set_title("Cumulative Intelligence Failure", fontweight='bold', fontsize=18)
    ax3.set_ylabel("Observer (Who is looking?)", fontsize=14)
    ax3.set_xlabel("Target (Who are they looking at?)", fontsize=14)

    # --- COL 4: Row Sums Bar Plot (Right) ---
    ax4 = fig.add_subplot(gs[:, 3], sharey=ax3)
    
    # Calculate Sums
    row_sums = error_matrix.sum(axis=1)
    
    # Colors aligned with actors
    bar_colors = [get_color(a) for a in row_sums.index]
    
    # Plot Horizontal Bars
    # Note: We use y=np.arange(len)+0.5 to align with Seaborn's heatmap cell centers
    y_pos = np.arange(len(row_sums)) + 0.5
    ax4.barh(y_pos, row_sums.values, height=0.8, color='white', edgecolor="gray")
    
    # Formatting
    ax4.axis("off") # Turn off spines and ticks
    
    # Add value labels to the right of the bars
    # Using row_sums.max() * 0.05 for padding to ensure text doesn't overlap bar
    x_pad = row_sums.max() * 0.05
    for i, v in enumerate(row_sums):
        ax4.text(v + x_pad, i + 0.5, f"{v:.0f}", 
                 va='center', fontsize=12, fontweight='bold', color="#2C3E50")
        
    ax4.set_title("Total Error", fontweight='bold', fontsize=14)

    fig.suptitle(title, fontsize=24, fontweight='bold', y=1.03)
    # Use explicit rect to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    return fig

def plot_network_breakdown(df, cfg):
    if "spillover_flow" not in df.columns: return
    set_publication_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    internal = df.groupby("actor")["Cap_Flow"].sum()
    spill = df.groupby("actor")["spillover_flow"].sum()
    
    total = internal + spill
    order = total.sort_values(ascending=False).index
    
    pd.DataFrame({"Self-Driven": internal, "Network Effect": spill}).loc[order].plot(
        kind='bar', stacked=True, ax=ax, color=["#2C3E50", "#E67E22"], width=0.75
    )
    ax.set_title("Sources of Capability Growth", fontweight='bold', fontsize=20)
    ax.legend(fontsize=14)
    plt.tight_layout()
    return fig

def plot_persona_evolution(logs, cfg, title="Persona Evolution"):
    if not logs: return None
    set_publication_style()
    df = pd.DataFrame(logs)
    
    w_cols = ["w_growth", "w_safety", "w_debt", "w_pol", "w_tech_adv"]
    display_map = {"w_growth":"Growth","w_safety":"Security","w_debt":"Fiscal","w_pol":"Political","w_tech_adv":"Catch-up"}
    
    actors = [a for a in cfg.actors if a in df["actor"].unique()]
    rows = int(np.ceil(len(actors)/4))
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4*rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    pal = sns.color_palette("RdYlBu", len(w_cols))
    
    for i, actor in enumerate(actors):
        ax = axes[i]
        adf = df[df["actor"] == actor].set_index("t")
        if adf.empty: ax.set_visible(False); continue
        
        valid = [c for c in w_cols if c in adf.columns]
        if not valid: ax.set_visible(False); continue
            
        data_pct = adf[valid].div(adf[valid].sum(axis=1), axis=0) * 100
        
        ax.stackplot(data_pct.index, data_pct.T, labels=[display_map[c] for c in valid], colors=pal, alpha=0.9)
        ax.set_title(actor, fontweight='bold', color=get_color(actor), fontsize=16)
        ax.set_ylim(0, 100)
        
        if i == 0: ax.legend(loc='lower left', fontsize=11, frameon=True)
        
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle(title, fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_family_comparison_enhanced(results, title):
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for label, res in results.items():
        df = res["df"]
        cfg = res["cfg"] 
        from .analytics import compute_leontief_metrics
        df = compute_leontief_metrics(df, cfg)
        
        axes[0,0].plot(df.groupby("t")["Cap"].mean(), label=label, linewidth=3.5)
        axes[0,1].plot(df.groupby("t")["maintenance_cost"].sum(), label=label, linewidth=3.5)
        axes[1,0].plot(df.groupby("t")["utilization_rate"].mean(), label=label, linewidth=3.5)
        axes[1,1].plot(df.groupby("t")["G"].mean(), label=label, linewidth=3.5)
        
    axes[0,0].set_title("Avg Global Capability", fontsize=18)
    axes[0,0].legend(fontsize=12)
    axes[0,1].set_title("Total Maintenance Cost", fontsize=18)
    axes[1,0].set_title("Avg Utilization", fontsize=18)
    axes[1,1].set_title("Avg Governance Level", fontsize=18)
    
    fig.suptitle(title, fontsize=24, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_pairwise_intelligence_heatmap(cfg, title="Structural Fog of War (Sigma Matrix)"):
    set_publication_style()
    actors = cfg.actors
    data = np.zeros((len(actors), len(actors)))
    
    for i, observer in enumerate(actors):
        for j, target in enumerate(actors):
            if observer == target:
                val = (cfg.sigma_self_map or {}).get(observer, cfg.sigma_self)
            else:
                is_ally = (observer, target) in cfg.A0_edges or (target, observer) in cfg.A0_edges
                if is_ally:
                    val = (cfg.sigma_ally_map or {}).get(observer, cfg.sigma_ally)
                else:
                    val = (cfg.sigma_rival_map or {}).get(observer, cfg.sigma_rival)
            data[i, j] = val

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(data, xticklabels=actors, yticklabels=actors, 
                cmap="Blues", annot=True, fmt=".2f", cbar_kws={'label': 'Noise (Sigma)'},
                linewidths=.5, linecolor='white', annot_kws={"size": 12})
    
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.set_xlabel("Target (Real State)", fontsize=16)
    ax.set_ylabel("Observer (Belief)", fontsize=16)
    plt.tight_layout()
    return fig

# Backward compatibility for old call if needed, though now replaced by plot_utility_breakdown_improved
plot_utility_breakdown_clean = plot_strategic_drivers_improved