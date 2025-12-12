import numpy as np
from copy import deepcopy
from typing import Dict, Set, List
from .config import Config
from .engine import GeoAIGame
from .policies import (
    policy_family1_baseline,
    policy_family1_eu_govfirst,
    policy_family1_eu_compete,
    policy_strategic_personas,
    reset_planners
)

# --- Helpers (Unchanged) ---
def _nudge_governance(cfg: Config, actors: Set[str], delta: float):
    for attr in ["G0_sec", "G0_eval", "G0_deploy"]:
        if not hasattr(cfg, attr): continue
        d = dict(getattr(cfg, attr))
        for a in actors:
            base = d.get(a, getattr(cfg, "G0", {}).get(a, 0.6))
            d[a] = max(0.3, min(0.97, base + delta))
        setattr(cfg, attr, d)

def _scale_hazard_sens(cfg: Config, actors: Set[str], factor: float):
    hs = dict(cfg.het.hazard_sensitivity)
    for a in actors:
        hs[a] = hs.get(a, 1.0) * factor
    cfg.het.hazard_sensitivity = hs

def _scale_sender_power(cfg: Config, actors: Set[str], factor: float):
    esp = dict(cfg.het.export_sender_power)
    for a in actors:
        esp[a] = esp.get(a, 1.0) * factor
    cfg.het.export_sender_power = esp

def _tweak_regime(cfg: Config, snd: str, tgt: str, new_vals: Dict[str, str]):
    key = (snd, tgt)
    if key not in cfg.X_regimes: return
    reg = dict(cfg.X_regimes[key])
    for cp, v in new_vals.items():
        if cp in reg: reg[cp] = v
    cfg.X_regimes[key] = reg

def _set_X0(cfg: Config, snd: str, tgt: str, val: float):
    if cfg.X0 is not None and snd in cfg.X0.index and tgt in cfg.X0.columns:
        cfg.X0.loc[snd, tgt] = val

def _set_panic(cfg: Config, actors: Set[str], val: float):
    """Helper to set risk tolerance."""
    for a in actors:
        cfg.het.panic_threshold[a] = val


# --- Family 1 ---
SAFETY_BLOC = {"US", "EU", "UK", "JPN", "KOR", "AUS", "CAN"}

def make_family1_config(base_cfg: Config, label: str) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.flags.use_belief_Gsec_in_hazards = True
    
    # Default Panic: 1.1 (Moderate)
    _set_panic(cfg, SAFETY_BLOC, 1.1)
    _set_panic(cfg, {"CN", "ROW", "IND", "BRA"}, 1.15) 

    if label == "baseline":
        return cfg

    if label == "eu_gov_first":
        _nudge_governance(cfg, SAFETY_BLOC, 0.08)
        _set_panic(cfg, SAFETY_BLOC, 0.95) 
        cfg.het.kappa_G["EU"] = 0.6 
        
    elif label == "eu_compete":
        _nudge_governance(cfg, SAFETY_BLOC, -0.05)
        _set_panic(cfg, SAFETY_BLOC, 1.35) 
        cfg.het.beta_C["EU"] = cfg.het.beta_C.get("EU", 0.6) + 0.15
        
    return cfg

def run_family1_scenarios(base_cfg: Config) -> Dict[str, dict]:
    scenarios = {
        "baseline":     (make_family1_config(base_cfg, "baseline"),     policy_family1_baseline),
        "eu_gov_first": (make_family1_config(base_cfg, "eu_gov_first"), policy_family1_eu_govfirst),
        "eu_compete":   (make_family1_config(base_cfg, "eu_compete"),   policy_family1_eu_compete), 
    }
    return _run_scenarios(scenarios)

# --- Family 2 & 3 ---
def make_family2_config(base_cfg: Config, label: str) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.T = max(cfg.T, 60)
    cfg.flags.use_belief_Gsec_in_hazards = True
    if label == "baseline_struct": return cfg
    if label == "hard_safety_bloc":
        _nudge_governance(cfg, SAFETY_BLOC, 0.06)
        _set_panic(cfg, SAFETY_BLOC, 0.95)
        for snd in SAFETY_BLOC:
            if snd == "CN": continue
            _tweak_regime(cfg, snd, "CN", {"LITHO":"BAN", "EDA":"BAN", "HBM":"LICENSE_FEE"})
            _set_X0(cfg, snd, "CN", 0.3)
        return cfg
    if label == "soft_hedging":
        _nudge_governance(cfg, SAFETY_BLOC, -0.03)
        _set_panic(cfg, SAFETY_BLOC, 1.2) 
        _tweak_regime(cfg, "EU", "CN", {"LITHO":"LICENSE_FEE", "EDA":"OPEN"})
        return cfg
    raise ValueError(f"Unknown: {label}")

def run_family2_scenarios(base_cfg: Config) -> Dict[str, dict]:
    return _run_scenarios({
        "baseline_struct":  (make_family2_config(base_cfg, "baseline_struct"),  policy_family1_baseline),
        "hard_safety_bloc": (make_family2_config(base_cfg, "hard_safety_bloc"), policy_family1_baseline),
        "soft_hedging":     (make_family2_config(base_cfg, "soft_hedging"),     policy_family1_baseline),
    })

def make_family3_config(base_cfg: Config, label: str) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.T = max(cfg.T, 60)
    if label == "baseline_coalitions": return cfg
    if label == "fragmented_west":
        cfg.flags.enable_alliance_sharing = False
        if cfg.sigma_ally_map: cfg.sigma_ally_map = {k: v*1.5 for k,v in cfg.sigma_ally_map.items()}
        _nudge_governance(cfg, SAFETY_BLOC, -0.05)
        _set_panic(cfg, SAFETY_BLOC, 1.3) 
        return cfg
    if label == "global_gov_pact":
        major = SAFETY_BLOC | {"CN", "IND", "BRA"}
        _nudge_governance(cfg, major, 0.08)
        _set_panic(cfg, major, 0.90) 
        if cfg.sigma_rival_map: cfg.sigma_rival_map = {k: v*0.8 for k,v in cfg.sigma_rival_map.items()}
        cfg.gpaith_enabled = True
        return cfg
    raise ValueError(f"Unknown: {label}")

def run_family3_scenarios(base_cfg: Config) -> Dict[str, dict]:
    return _run_scenarios({
        "baseline_coalitions": (make_family3_config(base_cfg, "baseline_coalitions"), policy_family1_baseline),
        "fragmented_west":     (make_family3_config(base_cfg, "fragmented_west"),     policy_family1_baseline),
        "global_gov_pact":     (make_family3_config(base_cfg, "global_gov_pact"),     policy_family1_baseline),
    })

# [NEW] Family 4: High Stakes (Testing Governance Value)
def make_high_stakes_config(base_cfg: Config, label: str) -> Config:
    cfg = deepcopy(base_cfg)
    # Make the world dangerous
    cfg.rates.crisis_hazard_mult = 3.0
    cfg.rates.lambda_mis = 0.15 # Triple probability of accidents
    cfg.rates.lambda_esc = 0.10
    
    if label == "danger_baseline": return cfg
    
    if label == "reckless_race":
        # Everyone cuts corners
        _nudge_governance(cfg, SAFETY_BLOC | {"CN"}, -0.15)
        _set_panic(cfg, SAFETY_BLOC | {"CN"}, 1.5)
        return cfg

    if label == "prudent_coop":
        # High governance
        _nudge_governance(cfg, SAFETY_BLOC | {"CN"}, 0.15)
        _set_panic(cfg, SAFETY_BLOC | {"CN"}, 0.9)
        return cfg
        
    return cfg

def run_high_stakes_scenarios(base_cfg: Config) -> Dict[str, dict]:
    return _run_scenarios({
        "danger_baseline": (make_high_stakes_config(base_cfg, "danger_baseline"), policy_family1_baseline),
        "reckless_race":   (make_high_stakes_config(base_cfg, "reckless_race"),   policy_family1_baseline),
        "prudent_coop":    (make_high_stakes_config(base_cfg, "prudent_coop"),    policy_family1_baseline),
    })

def _run_scenarios(scenarios: Dict[str, tuple]) -> Dict[str, dict]:
    out = {}
    for name, (cfg, pol) in scenarios.items():
        game = GeoAIGame(cfg)
        df = game.run(policy=pol)
        out[name] = {"cfg": cfg, "df": df}
    return out

# [NEW] Family 5: The Semiconductor Stranglehold
def make_stranglehold_config(base_cfg: Config, label: str) -> Config:
    cfg = deepcopy(base_cfg)
    
    if label == "no_blockade": return cfg
    
    if label == "us_eu_blockade":
        # Event: At t=15, US and EU ban exports to CN
        blockade_events = []
        # We trigger this at t=15. 
        # Note: The engine processes events at the start of the step.
        t_trigger = 15
        
        # 1. US bans CN on everything
        blockade_events.append({
            "t": t_trigger, 
            "fn": "blockade",
            "args": {
                "sender": "US", "target": "CN",
                "regimes": {"LITHO": "BAN", "EDA": "BAN", "HBM": "BAN", "CLOUD": "BAN"}
            }
        })
        
        # 2. EU bans CN on Lithography (their choke point)
        blockade_events.append({
            "t": t_trigger, 
            "fn": "blockade",
            "args": {
                "sender": "EU", "target": "CN",
                "regimes": {"LITHO": "BAN", "EDA": "BAN"}
            }
        })
        
        cfg.events.extend(blockade_events)
        return cfg

    return cfg

def run_semiconductor_stranglehold(base_cfg: Config) -> Dict[str, dict]:
    np.random.seed(base_cfg.seed)
    reset_planners()
    return _run_scenarios({
        "No Blockade (Counterfactual)": (make_stranglehold_config(base_cfg, "no_blockade"), policy_strategic_personas),
        "US-EU Blockade":               (make_stranglehold_config(base_cfg, "us_eu_blockade"), policy_strategic_personas),
    })