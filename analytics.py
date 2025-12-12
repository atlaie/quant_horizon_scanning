import pandas as pd
import numpy as np
from .utils import gini, soft_min

def compute_leontief_metrics(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Computes physical constraints, efficiency, and fiscal health metrics.
    """
    if "energy_efficiency" in df.columns:
        eff_series = df["energy_efficiency"]
    else:
        eff_series = cfg.rates.energy_efficiency
    
    # 1. Leontief Ratio (Demand / Supply)
    df["leontief_ratio"] = df["C"] / (df["E"] * eff_series + 1e-9)
    
    # 2. Effective Capability (Soft Min)
    available_power = df["E"] * eff_series
    
    # [FIX] Apply row-wise soft_min safely using numpy vectorization if possible,
    # but since soft_min is scalar in utils, we use apply or direct numpy approx.
    # Here we use the numpy approximation directly for speed and safety.
    # S(x,y) = 0.5 * (x + y - sqrt((x-y)^2 + 1.0))
    diff = df["C"] - available_power
    smooth_min = 0.5 * (df["C"] + available_power - np.sqrt(diff**2 + 1.0))
    df["constrained_C"] = smooth_min.clip(lower=0.0) # Safety Clamp
    
    # 3. Wasted Capital (Absolute)
    df["wasted_C"] = (df["C"] - df["constrained_C"]).clip(lower=0)
    
    # [NEW] 4. Normalized Metrics
    df["inefficiency_rate"] = df["wasted_C"] / (df["C"] + 1e-9)
    
    # Utilization Rate: % of Hardware Active
    # [FIX] Clamp to [0, 100] to prevent visualization explosions
    raw_util = (df["constrained_C"] / (df["C"] + 1e-9)) * 100.0
    df["utilization_rate"] = raw_util.clip(0.0, 100.0)
    
    # [NEW] 5. Fiscal Health
    df["debt_to_gdp"] = df["Debt"] / (df["GDP"] + 1e-9)
    
    return df

def parse_action_mix(df: pd.DataFrame) -> pd.DataFrame:
    if "actions" not in df.columns: return df
    targets = ["invest_compute", "invest_energy", "invest_data", 
               "governance_up_security", "governance_up_eval"]
    for tgt in targets:
        short_name = tgt.replace("invest_", "Inv_").replace("governance_up_", "Gov_")
        df[f"act_{short_name}"] = df["actions"].apply(lambda x: 1 if tgt in str(x) else 0)
    return df

def compute_core_panels(df: pd.DataFrame) -> dict:
    panels = {}
    panels["cap_share"] = df.pivot_table(index="t", columns="actor", values="Cap", aggfunc="mean")
    panels["cap_share"] = panels["cap_share"].div(panels["cap_share"].sum(axis=1)+1e-9, axis=0)
    panels["race"]      = df.groupby("t")["race_intensity"].mean()
    panels["haz_count"] = (df.groupby("t")["misuse"].sum() + df.groupby("t")["escalation"].sum())
    panels["cap_total"] = df.groupby("t")["total_capability"].mean()
    panels["crisis"]    = df.groupby("t")["crisis_active"].max()
    return panels

def _hhi_series(df: pd.DataFrame) -> pd.Series:
    share = df.pivot_table(index="t", columns="actor", values="Cap", aggfunc="mean")
    share = share.div(share.sum(axis=1)+1e-9, axis=0)
    return share.pow(2).sum(axis=1)

def summarize(df: pd.DataFrame, name: str, print_: bool = False) -> dict:
    if df.empty: return {}
    end = int(df["t"].max())
    hz = df[["misuse","escalation"]].sum()
    final_slice = df[df["t"]==end]
    final_cap = final_slice["Cap"].sum()
    final_G   = final_slice["G"].mean()
    
    out = {
        "name": name, "end_t": end, "hazards_total": int(hz.sum()),
        "race_mean": float(df["race_intensity"].mean()),
        "crisis_pct": 100.0 * float(df["crisis_active"].mean()),
        "final_global_cap": float(final_cap), "final_avg_gov": float(final_G)
    }
    if print_:
        print(f"\n== {name} Summary ==")
        print(f"  Crisis Time: {out['crisis_pct']:.1f}%")
        print(f"  Hazards: {out['hazards_total']}")
        print(f"  Final Global Cap: {out['final_global_cap']:.2f}")
        print(f"  Final Avg Gov: {out['final_avg_gov']:.2f}")
    return out