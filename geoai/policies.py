# geoai/policies.py
import numpy as np
import copy 
from typing import Dict, Set, Optional, List, Tuple

from .config import Config
from .constants import (
    INVEST_C, INVEST_D, INVEST_E, GOV_UP, GOV_DOWN,
    GOV_UP_SEC, GOV_UP_EVAL,
    SANCTION_RIVAL, COVERT_SABOTAGE, COVERT_THEFT,
    SIGN_TREATY, FORM_ALLIANCE
)
from .state import ActorState
from .utils import expit, clip01

PLANNER_LOGS = []

def reset_planners():
    global _planners
    _planners = {}

# --- Uncooperative / Legacy Policies (Unchanged) ---
def policy_uncooperative_race(t: int, state: Dict[str, ActorState], cfg: Config) -> Dict[str, dict]:
    dec = {}
    eff = cfg.rates.energy_efficiency
    for a, s in state.items():
        acts: Set[str] = set()
        if s.insolvent: 
            dec[a] = {"actions": set()}
            continue
        debt_ratio = s.Debt / (s.GDP + 1e-9)
        # Safe sigmoid
        prob_inaction = expit(cfg.rates.insolvency_threshold_k * (debt_ratio - cfg.rates.insolvency_threshold_center))
        if np.random.random() < prob_inaction:
            dec[a] = {"actions": set()}
            continue
        util_ratio = s.C / (s.E * eff + 1e-9)
        has_budget = s.B > cfg.cost_C * 1.5
        can_expand_growth = (debt_ratio < 2.0) and has_budget
        if s.G > 0.2 and t % 4 == 0: acts.add(GOV_DOWN) 
        if util_ratio > 0.85: acts.add(INVEST_E)
        if can_expand_growth:
            acts.add(INVEST_C)
            if s.B > (0.5 * s.GDP): acts.add(INVEST_D)
        dec[a] = {"actions": acts}
    return dec

def policy_family1_baseline(t: int, state: Dict[str, ActorState], cfg: Config) -> Dict[str, dict]:
    snap = {a: {"G_eval": s.G_eval, "R": s.R, "Cap": s.C} for a, s in state.items()}
    actors = list(state.keys())
    dec: Dict[str, dict] = {}
    eff = cfg.rates.energy_efficiency
    for a, s in state.items():
        acts: Set[str] = set()
        if s.insolvent:
             dec[a] = {"actions": set()}
             continue
        debt_ratio = s.Debt / (s.GDP + 1e-9)
        prob_inaction = expit(cfg.rates.insolvency_threshold_k * (debt_ratio - cfg.rates.insolvency_threshold_center))
        if np.random.random() < prob_inaction:
            dec[a] = {"actions": set()}
            continue
        leontief_ratio = s.C / (s.E * eff + 1e-9)
        panic = cfg.het.panic_threshold.get(a, 1.10)
        if leontief_ratio > panic:
            acts.add(INVEST_E)
        else:
            if debt_ratio <= 2.0:
                strategy = cfg.het.tech_strategy.get(a, "balanced")
                if strategy == "dominance": acts.add(INVEST_C)
                elif strategy == "regulatory": acts.add(GOV_UP_EVAL if (t % 2 == 0) else GOV_UP_SEC)
                elif strategy == "mixed_tech": acts.add(INVEST_C if (t % 3 != 0) else GOV_UP_SEC)
                elif strategy == "catch_up": acts.add(INVEST_E if s.E < 1.0 else INVEST_D)
                else: acts.add(INVEST_D if s.D < s.E else INVEST_E)
        candidates = [b for b in actors if b != a]
        target = max(candidates, key=lambda b: (snap[b]["R"], snap[b]["G_eval"])) if candidates else None
        ally_add = {target} if target else set()
        dec[a] = {"actions": acts, "ally_add": ally_add, "ally_remove": set()}
    return dec

def policy_family1_eu_govfirst(t: int, state: Dict[str, ActorState], cfg: Config) -> Dict[str, dict]:
    dec = policy_family1_baseline(t, state, cfg)
    if "EU" in dec:
        dec["EU"]["actions"].add(GOV_UP_SEC)
        if INVEST_C in dec["EU"]["actions"]: dec["EU"]["actions"].remove(INVEST_C)
    return dec

def policy_family1_eu_compete(t: int, state: Dict[str, ActorState], cfg: Config) -> Dict[str, dict]:
    dec = policy_family1_baseline(t, state, cfg)
    if "EU" in dec:
        s_eu = state["EU"]
        eff = cfg.rates.energy_efficiency
        ratio = s_eu.C / (s_eu.E * eff + 1e-9)
        if ratio < 2.0: dec["EU"]["actions"].add(INVEST_C)
    return dec

# --- Principled Physics Engine (ShadowModel) ---

class ShadowModel:
    @staticmethod
    def simulate_step(
        s: ActorState, 
        acts: Set[str], 
        cfg: Config, 
        current_race_int: float, 
        treaty_strength: float,
        worst_case: bool = False,
        is_sanctioned: bool = False
    ) -> Tuple[ActorState, float, float]: 
        
        r = cfg.rates
        s_next = s.clone()
        
        invest_C = INVEST_C in acts
        invest_E = INVEST_E in acts
        gov_up   = GOV_UP_SEC in acts
        
        # 1. Economic Physics
        maint_allies = len(s_next.allies) * getattr(r, "maintenance_rate_alliance", 0.0)
        maint_bill = s_next.C * r.maintenance_rate_C + s_next.K * r.maintenance_rate_K + maint_allies
        
        act_bill = 0.0
        
        if invest_C: act_bill += cfg.cost_C
        if invest_E: act_bill += cfg.cost_E
        if gov_up:   act_bill += cfg.cost_Gup
        if COVERT_THEFT in acts: act_bill += r.cost_theft
        if COVERT_SABOTAGE in acts: act_bill += r.cost_sabotage
        if FORM_ALLIANCE in acts: act_bill += getattr(r, "cost_alliance_formation", 0.0)
        
        # Debt/Budget
        debt_ratio = s_next.Debt / (s_next.GDP + 1e-9)
        # Clamp debt ratio for interest rate calc to prevent overflow
        safe_debt_ratio = min(debt_ratio, 10.0) 
        
        r_min = r.yield_curve_min_rate
        r_max = r.yield_curve_max_rate
        k_curve = r.yield_curve_steepness
        
        # [NEW] Check for heterogeneous threshold in cfg.het using a heuristic or passed context
        # Since we can't easily get the actor ID inside this static method without changing signature,
        # we will assume the standard midpoint BUT scale it if the debt is huge.
        x0 = r.yield_curve_midpoint 
        
        if worst_case: x0 -= 0.2 
        
        # Safe Sigmoid
        effective_rate = r_min + (r_max - r_min) / (1.0 + np.exp(-k_curve * (safe_debt_ratio - x0)))
        
        interest = s_next.Debt * effective_rate
        revenue = s_next.GDP * r.tax_rate
        total_expenses = maint_bill + act_bill + interest
        balance = revenue - total_expenses
        s_next.B += balance
        # [FIX] ShadowModel predicts "Use It or Lose It"
        reserves_cap = 1.0 * s_next.GDP
        if s_next.B > reserves_cap:
            excess = s_next.B - reserves_cap
            # Matches Engine: 5% burn
            burn = excess * 0.05
            s_next.B -= burn

        if s_next.B < 0:
            s_next.Debt += abs(s_next.B); s_next.B = 0.0
        elif s_next.B > 0 and s_next.Debt > 0:
            repay = min(s_next.B, s_next.Debt); s_next.Debt -= repay; s_next.B -= repay

        # [INSTRUMENTAL LOGIC: ALLIANCES]
        current_allies_count = len(s_next.allies)
        if FORM_ALLIANCE in acts:
            current_allies_count = min(5, current_allies_count + 1)
            s_next.G_sec = min(0.99, s_next.G_sec + 0.05)
            s_next.political_capital = min(1.0, s_next.political_capital + 0.15)
        
        simulated_beta_C = r.beta_C + (0.005 * current_allies_count)
        
        # Growth
        invest_bonus = simulated_beta_C if invest_C else 0.0
        
        # Shadow Depreciation match Engine, mirroring Engine logic
        deprec_C = r.depreciation_C
        throttle = 1.0
        
        if is_sanctioned:
            # Mirror the "Spare Parts Crisis" logic
            # Litho ban assumption (approximate)
            throttle = 0.25 
            supply_chain_stress = (1.0 - throttle) * 0.20
            deprec_C = deprec_C + supply_chain_stress
            
        # [FIX] Energy Throttle in ShadowModel
        e_throttle = 1.0 if throttle >= 0.9 else 0.5
        
        current_allies_count = len(s_next.allies)
        
        simulated_beta_C = r.beta_C + (0.005 * current_allies_count)
        invest_bonus = simulated_beta_C if (INVEST_C in acts) else 0.0
        
        # Apply the harsher physics
        gC = (1.0 - deprec_C) + (r.gC_base + invest_bonus) * throttle
        
        invest_E_bonus = r.beta_E if invest_E else 0.0
        if invest_E_bonus > 0: s_next.E_backlog.append((r.gE_base + invest_E_bonus) * e_throttle)
        else: s_next.E_backlog.append(r.gE_base)
        mature_E = s_next.E_backlog.popleft() if s_next.E_backlog else 0.0
        gE = 1.0 + mature_E
        
        s_next.C *= gC; s_next.E *= gE
        if gov_up: s_next.G_sec = min(0.95, s_next.G_sec + r.Delta_G)
            
        # Production
        eff = r.energy_efficiency
        constrained_C = min(s_next.C, s_next.E * eff)
        
        if is_sanctioned:
            constrained_C *= 0.20 
            
        gov_drag = 1.0 - (r.safety_compute_drag * s_next.G_sec)
        _, wC, wD, _, _ = cfg.weights.norm_cap_weights()
        raw_output = (constrained_C ** wC) * (s_next.D ** wD) * gov_drag
        
        if COVERT_THEFT in acts:
             s_next.D *= 1.05 
             s_next.political_capital = max(0.0, s_next.political_capital - 0.15)
        
        s_next.K = s_next.K * (1.0 - r.knowledge_depreciation) + raw_output
        
        if SIGN_TREATY in acts:
            s_next.political_capital = max(0.0, s_next.political_capital - 0.05)
        
        gov_bonus = r.governance_gdp_bonus * s_next.G
        s_next.GDP *= (1.0 + 0.005 + gov_bonus) 

        if SANCTION_RIVAL in acts:
            s_next.political_capital = max(0.0, s_next.political_capital - 0.20) 

        new_treaty_strength = clip01(treaty_strength + 0.05) if SIGN_TREATY in acts else clip01(treaty_strength - 0.05)
        
        base_race = current_race_int
        target_race = 0.2 
        if invest_C: target_race += 0.25
        if SANCTION_RIVAL in acts: target_race += 0.45
        dampener = 1.0 - (0.5 * new_treaty_strength)
        new_race_int = (0.6 * base_race + 0.4 * target_race) * dampener
        
        return s_next, new_race_int, new_treaty_strength

    @staticmethod
    def predict_rival_move(my_acts: Set[str], s_rival: ActorState, cfg: Config, aggressive: bool) -> Set[str]:
        rival_acts = {INVEST_C}
        if SANCTION_RIVAL in my_acts: rival_acts.add(SANCTION_RIVAL)
        return rival_acts

class StrategicPlanner:
    def __init__(self, actor_id: str, preferences: Dict[str, float] = None):
        self.actor = actor_id
        self.base_params = {
            "w_growth": 1.0, "w_relative": 1.0, "w_safety": 1.0,
            "w_debt": 1.0, "w_switching": 2.0, "w_terminal": 1.0, 
            "w_pol": 5.0, "w_waste_penalty": 0.25, 
            "w_tech_adv": 1.0, 
            "w_hoard": 0.5,
            "risk_tol": 1.0, "panic_threshold": 0.2, 
            "temperature": 0.5 
        }
        if preferences: self.base_params.update(preferences)
        self.params = self.base_params.copy()
        self.target_params = self.base_params.copy()
        
        self.rival_id = None
        if actor_id == "US": self.rival_id = "CN"
        if actor_id == "CN": self.rival_id = "US"

    def adapt_weights(self, s: ActorState, state: Dict[str, ActorState], cfg: Config):
        # 1. Inputs
        tension = getattr(cfg, "race_intensity", 0.0)
        global_max_K = max(st.K for st in state.values())
        my_share = s.K / (global_max_K + 1e-9)
        reserves_ratio = s.B / (s.GDP + 1e-9)
        debt_ratio = s.Debt / (s.GDP + 1e-9)
        
        # 2. The Fear Drive
        fear_mult = 1.0 + (2.0 * tension)
        waste_tol_mult = 1.0 / (1.0 + 2.0 * tension)
        
        # 3. The Status Drives
        dominance_drive = my_share ** 2.0
        catchup_drive = (1.0 - my_share) ** 2.0
        influence_drive = np.exp(-((my_share - 0.40)**2) / (2 * (0.15**2)))

        # 4. The Fiscal Drive
        fiscal_confidence = expit(3.0 * (reserves_ratio - 1.0))
        debt_panic = expit(5.0 * (debt_ratio - 0.6))

        # 5. Synthesis
        target = self.base_params.copy()
        target["w_safety"] *= fear_mult
        target["w_pol"]    *= fear_mult  
        target["w_waste_penalty"] *= waste_tol_mult

        target["w_relative"] *= (1.0 + 3.0 * dominance_drive) 
        
        target["w_tech_adv"] *= (1.0 + 4.0 * catchup_drive)   
        target["w_relative"] *= (1.0 - 0.8 * catchup_drive) 
        
        target["w_pol"] *= (1.0 + 4.0 * influence_drive)

        target["w_growth"] *= (1.0 + 1.0 * fiscal_confidence) 
        target["w_debt"]   *= (1.0 - 0.5 * fiscal_confidence)
        
        target["w_growth"] *= (1.0 - 0.6 * debt_panic)
        target["w_debt"]   *= (1.0 + 4.0 * debt_panic)

        # 6. Inertia
        alpha = 0.25
        for k in self.params:
            if k in target:
                self.params[k] = (1 - alpha) * self.params[k] + alpha * target[k]
    
    def _get_candidates(self, s: ActorState, state: Dict[str, ActorState], restricted: bool = False) -> List[Tuple[str, Set[str]]]:
        candidates = []
        candidates.append(("Invest_Compute", {INVEST_C}))
        
        if not restricted:
            candidates.append(("Invest_Energy",  {INVEST_E}))
            candidates.append(("Compute+Energy", {INVEST_C, INVEST_E}))
            if s.B > 0.5 * s.GDP:
                candidates.append(("Max_Growth", {INVEST_C, INVEST_E, INVEST_D}))

        max_K = max(st.K for st in state.values())
        is_leader = (s.K >= max_K * 0.98)
        
        if is_leader and s.political_capital > 0.6:
             candidates.append(("Sanction", {SANCTION_RIVAL}))
             if not restricted: candidates.append(("Grow+Sanction", {INVEST_C, SANCTION_RIVAL}))

        if not is_leader and s.B > 0.2: 
            candidates.append(("Espionage_Theft", {COVERT_THEFT}))

        candidates.append(("Diplomacy_Treaty", {SIGN_TREATY}))
        
        # [FIX] Always allow evaluating alliances to enable poaching/swapping
        candidates.append(("Alliance", {FORM_ALLIANCE}))

        return candidates
    
    def calculate_utility(self, s_me, s_rival, s_initial, cfg, race_int, my_acts): 
        p = self.params; w = cfg.weights 
        
        u_pol = p.get("w_pol", 50.0) * s_me.political_capital
        
        if SANCTION_RIVAL in my_acts:
            u_pol -= 20.0 

        # Log1p handles small K gracefully
        u_abs = np.log1p(s_me.K) * p["w_growth"] * w.scale_growth
        
        u_rel = 0.0
        if s_rival:
            share = s_me.K / (s_me.K + s_rival.K + 1e-9)
            u_rel = (1.0 / (1.0 + np.exp(-10.0 * (share - 0.5)))) * p["w_relative"] * w.scale_relative

        u_debt = 0.0
        debt_ratio = s_me.Debt / (s_me.GDP + 1e-9)
        
        # [ROBUST] Fallback logic for heterogeneity
        limit = 4.0 
        if hasattr(cfg.het, "insolvency_threshold"):
             limit = cfg.het.insolvency_threshold.get(self.actor, 4.0)

        # [FIX] Clamp the ratio for the penalty curve to prevent overflow
        # If debt exceeds 5x the limit, the pain is already "infinite enough"
        ratio = debt_ratio / (limit + 1e-9)
        safe_ratio = min(ratio, 5.0) 
        
        if debt_ratio > 0.1:
             # Calculate penalty using the CLAMPED ratio
             u_debt = p["w_debt"] * (np.exp(safe_ratio) - 1.0) * 2.0

        u_hoard = 0.0
        reserves_ratio = s_me.B / (s_me.GDP + 1e-9)
        hoard_limit = 0.60
        if reserves_ratio > hoard_limit:
            # Linear scaling is safer than quadratic for unbounded inputs
            excess = reserves_ratio - hoard_limit
            u_hoard = -1.0 * p.get("w_hoard", 0.5) * excess * 5.0

        eff = cfg.rates.energy_efficiency
        inefficiency = max(0.0, (s_me.C / (s_me.E * eff + 1e-9)) - 1.0)
        u_waste = -1.0 * p.get("w_waste_penalty", 1.0) * (inefficiency ** 2.0)

        pending = sum(s_me.E_backlog) if s_me.E_backlog else 0.0
        u_term = p["w_terminal"] * np.log1p(s_me.E + pending)

        total = u_abs + u_rel + u_term + u_pol + u_waste - u_debt + u_hoard
        
        return total, {
            "u_growth": u_abs, "u_relative": u_rel, "u_pol": u_pol, 
            "u_waste": u_waste, "u_debt": -u_debt, "u_terminal": u_term,
            "u_hoard": u_hoard 
        }

    def _estimate_rival_response(self, s_rival: ActorState, my_acts: Set[str], cfg: Config) -> Set[str]:
        rival_acts = {INVEST_C}
        if SANCTION_RIVAL in my_acts: rival_acts.add(SANCTION_RIVAL)
        return rival_acts
        
    def _simulate_trajectory(self, s_me, s_rival, my_acts, rival_acts, cfg, horizon, worst_case):
        sim_me = s_me.clone()
        sim_rival = s_rival.clone() if s_rival else None
        current_race = 0.5 
        treaty_strength = 0.0 
        total = 0.0; discount = 1.0; beta = 0.90 
        me_sanctioning = SANCTION_RIVAL in my_acts
        rival_sanctioning = SANCTION_RIVAL in rival_acts if rival_acts else False

        for _ in range(horizon):
            sim_me, current_race, treaty_strength = ShadowModel.simulate_step(
                sim_me, my_acts, cfg, current_race, treaty_strength, worst_case, is_sanctioned=rival_sanctioning 
            )
            if sim_rival is not None:
                sim_rival, _, _ = ShadowModel.simulate_step(
                    sim_rival, rival_acts, cfg, current_race, treaty_strength, worst_case, is_sanctioned=me_sanctioning 
                )
            step_u, _ = self.calculate_utility(sim_me, sim_rival, s_me, cfg, current_race, my_acts)
            total += discount * step_u
            discount *= beta
        return total

    def get_plan(self, t, s, state, cfg, horizon=8, restricted_candidates=False):
        if s.commitment_timer > 0: return {"actions": s.prev_actions}
        rival_s = state.get(self.rival_id)
        
        self.adapt_weights(s, state, cfg)
        candidates = self._get_candidates(s, state, restricted=restricted_candidates)
        
        scored = []
        scores_log = {}
        
        for name, acts in candidates:
            my_acts = set(acts)
            rival_acts = set()
            if rival_s and not restricted_candidates:
                rival_acts = ShadowModel.predict_rival_move(my_acts, rival_s, cfg, False)
            
            score = self._simulate_trajectory(s, rival_s, my_acts, rival_acts, cfg, horizon, False)
            # Ensure Score is Finite
            if not np.isfinite(score): score = -1e9 
            scored.append((score, name, my_acts))
            scores_log[f"score_{name}"] = score

        scores = np.array([x[0] for x in scored])
        scores = np.nan_to_num(scores, nan=-1e9)
        
        # [FIX] Robust Softmax
        # 1. Clip values to avoid overflow before exp
        if len(scores) > 0:
            scores = scores - np.max(scores) # Shift for stability
            scores = np.clip(scores, -500, 0) # Clamp bottom end to prevent underflow issues if needed
        
        probs = np.exp(scores / self.params.get("temperature", 1.0))
        if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)): 
            probs = np.ones_like(probs) / len(probs)
        else: 
            probs /= np.sum(probs)
        
        idx = np.random.choice(len(candidates), p=probs)
        best_score, best_name, best_acts = scored[idx]
        
        sanctions = set(getattr(s, "active_sanctions", set()))
        if SANCTION_RIVAL in best_acts and self.rival_id: sanctions.add(self.rival_id)
        
        # Dynamic Ally Selection (Poaching) ---
        ally_add = set()
        ally_remove = set()
        
        if FORM_ALLIANCE in best_acts:
            # 1. Score all potential partners
            scores = []
            for other_id, other_s in state.items():
                if other_id == self.actor: continue
                
                score = other_s.K # Base Score: Power (K)
                
                # Bonus: Enemy of my Enemy
                my_targets = s.active_sanctions 
                their_targets = other_s.active_sanctions
                
                # If we share a rival (e.g. both hate the US), huge bonus
                if self.rival_id and self.rival_id in their_targets:
                    score *= 1.5
                # If we share any sanction target, small bonus
                elif not my_targets.isdisjoint(their_targets):
                    score *= 1.2
                
                # Penalty: Toxic Ally (Low Governance/High Risk)
                if other_s.G_sec < 0.3:
                    score *= 0.5 
                
                scores.append((score, other_id))
            
            # Sort: Strongest/Best Fit first
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # 2. Identify Best Available Candidate
            best_candidate = None
            for sc, aid in scores:
                if aid not in s.allies:
                    best_candidate = (sc, aid)
                    break
            
            # 3. Decision: Add or Swap?
            if best_candidate:
                cand_score, cand_id = best_candidate
                
                if len(s.allies) < 3:
                    # Free slot? Take it.
                    ally_add.add(cand_id)
                else:
                    # Full? Check for upgrade (Poaching). Find worst current ally
                    current_allies_scores = [(sc, aid) for sc, aid in scores if aid in s.allies]
                    
                    if current_allies_scores:
                        # Worst is the last one in the sorted list
                        worst_ally_score, worst_ally_id = current_allies_scores[-1]
                        
                        # Threshold: New ally must be >20% better to justify the betrayal cost
                        if cand_score > worst_ally_score * 1.2:
                            ally_add.add(cand_id)
                            ally_remove.add(worst_ally_id) # Betrayal!

        timer = 1 if ("Energy" in best_name or "Compute" in best_name) else 0
        sanctions = set(getattr(s, "active_sanctions", set()))
        if SANCTION_RIVAL in best_acts and self.rival_id: sanctions.add(self.rival_id)

        return {"actions": best_acts, "sanctions": sanctions, "ally_add": ally_add, "ally_remove": ally_remove, "set_timer": timer}

_planners = {}

def policy_strategic_personas(t, state, cfg):
    global _planners
    
    # 1. Initialize planners if they don't exist
    if not _planners:
        for a in cfg.actors:
            p = {"w_growth": 10.0, "w_relative": 5.0} # Default
            if a == "US": p = {"w_growth": 20.0, "w_relative": 15.0, "w_pol": 40.0}
            if a == "CN": p = {"w_growth": 30.0, "w_relative": 10.0, "w_waste_penalty": 0.0}
            if a == "EU": p = {"w_growth": 5.0, "w_pol": 80.0, "w_waste_penalty": 0.1}
            _planners[a] = StrategicPlanner(a, p)
    
    decisions = {}
    
    for a, s in state.items():
        if s.insolvent: 
            continue
            
        # 2. Get the optimal plan
        planner = _planners[a]
        plan = planner.get_plan(t, s, state, cfg)
        decisions[a] = plan
        
        # 3. Prepare the log entry with current Weights
        log_entry = planner.params.copy()
        log_entry["t"] = t
        log_entry["actor"] = a
        
        # 4. [CRITICAL FIX] Calculate and log the Utility Breakdown
        # We must re-calculate the utility components for the *chosen* plan
        # to visualize what drove this decision.
        rival_s = state.get(planner.rival_id)
        race_int = getattr(cfg, "race_intensity", 0.0)
        
        _, u_components = planner.calculate_utility(
            s_me=s, 
            s_rival=rival_s, 
            s_initial=s, 
            cfg=cfg, 
            race_int=race_int, 
            my_acts=plan["actions"]
        )
        
        # Merge utility values (u_growth, u_debt, etc.) into the log
        log_entry.update(u_components)
        PLANNER_LOGS.append(log_entry)
    
    return decisions

policy_rational_planning = policy_strategic_personas