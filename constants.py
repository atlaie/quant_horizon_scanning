# geoai/constants.py
from typing import Dict, Set

# --- Investment Actions ---
INVEST_C = "invest_compute"
INVEST_D = "invest_data"
INVEST_E = "invest_energy"

# --- Governance Actions (Defensive) ---
GOV_UP     = "governance_up"       # legacy
GOV_DOWN   = "governance_down"     # legacy

GOV_UP_SEC      = "governance_up_security"
GOV_UP_EVAL     = "governance_up_evaluation"
GOV_UP_DEPLOY   = "governance_up_deployment"

GOV_DOWN_SEC    = "governance_down_security"
GOV_DOWN_EVAL   = "governance_down_evaluation"
GOV_DOWN_DEPLOY = "governance_down_deployment"

# --- Geopolitics Actions (Offensive) ---
SANCTION_RIVAL  = "sanction_rival" 
LIFT_SANCTION   = "lift_sanction"
COVERT_SABOTAGE = "covert_sabotage" # Target: Energy Infrastructure / Reliability
COVERT_THEFT    = "covert_theft"    # Target: Data / IP

# --- Diplomatic Actions (New) ---
SIGN_TREATY     = "sign_treaty"     # Reduces Global Race Intensity
FORM_ALLIANCE   = "form_alliance"   # Shares Intel (Reduces Fog of War)

# --- Governance Action Maps ---
G_UP: Dict[str, str] = {
    GOV_UP_SEC:    "G_sec",
    GOV_UP_EVAL:   "G_eval",
    GOV_UP_DEPLOY: "G_deploy",
}
G_DN: Dict[str, str] = {
    GOV_DOWN_SEC:    "G_sec",
    GOV_DOWN_EVAL:   "G_eval",
    GOV_DOWN_DEPLOY: "G_deploy",
}
LEGACY = {GOV_UP: GOV_UP_EVAL, GOV_DOWN: GOV_DOWN_DEPLOY}
UP_VARIANTS: Set[str] = set(G_UP)
DN_VARIANTS: Set[str] = set(G_DN)

def apply_gov_delta(state, acts: Set[str], dG: float) -> None:
    from .utils import clip01 
    
    for act, attr in G_UP.items():
        if act in acts:
            setattr(state, attr, clip01(getattr(state, attr) + dG))

    for act, attr in G_DN.items():
        if act in acts:
            setattr(state, attr, clip01(getattr(state, attr) - dG))