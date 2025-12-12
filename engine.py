# geoai/engine.py
from __future__ import annotations
import dataclasses as dc
from typing import Dict, List, Set, Tuple, Callable, Optional
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import networkx as nx

from .utils import rate_per_step, expit, logit, clip01, mean_or0, get_map, soft_min
from .config import Config
from .constants import (
    INVEST_C, INVEST_D, INVEST_E, GOV_UP, GOV_DOWN, 
    UP_VARIANTS, DN_VARIANTS, G_UP, G_DN, LEGACY,
    SANCTION_RIVAL, LIFT_SANCTION, COVERT_SABOTAGE, COVERT_THEFT,
    SIGN_TREATY, FORM_ALLIANCE,
    apply_gov_delta
)
from .state import ActorState, StepLog

class GeoAIGame:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.idx = {a:i for i,a in enumerate(cfg.actors)}
        self._gpaith_tripped: Set[str] = set()
        self.freeze_timer: Dict[str, int] = {a: 0 for a in cfg.actors}
        self.credit_freeze: Dict[str, int] = {a: 0 for a in cfg.actors}

        self._gC_step = rate_per_step(cfg.rates.gC_base, cfg.steps_per_year)
        self._gD_step = rate_per_step(cfg.rates.gD_base, cfg.steps_per_year)
        self._gE_step = rate_per_step(cfg.rates.gE_base, cfg.steps_per_year)
        def _get(d, a, default): return float(d.get(a, default))
        
        self.state: Dict[str, ActorState] = {}
        # [NEW] Track max talent to prevent explosions
        self._max_possible_D = 10.0
        
        for a in cfg.actors:
            S0 = clip01(_get(cfg.S0, a, 0.8))
            tau_s = cfg.rates.s_pipeline_tau
            tau_e = cfg.rates.energy_build_lag
            
            initial_pipeline_flow = cfg.rates.gE_base
            backlog_e = deque([initial_pipeline_flow] * max(0, tau_e - 1), maxlen=max(0, tau_e - 1))
            backlog_s = deque([cfg.rates.s_pipeline_base] * max(0, tau_s - 1), maxlen=max(0, tau_s - 1))
            
            g_base  = clip01(_get(cfg.G0, a, 0.5))
            gdp_base = 10.0 if a in ["US", "CN", "EU"] else 3.0 
            
            self.state[a] = ActorState(
                C=_get(cfg.C0, a, 1.0), D=_get(cfg.D0, a, 1.0), E=_get(cfg.E0, a, 1.0),
                K=0.1, 
                G_sec=clip01(cfg.G0_sec.get(a, g_base)), 
                G_eval=clip01(cfg.G0_eval.get(a, g_base)), 
                G_deploy=clip01(cfg.G0_deploy.get(a, g_base)), 
                S=S0, S_backlog=backlog_s,
                E_backlog=backlog_e,
                GDP=gdp_base, Debt=0.0, B=gdp_base * 0.2,
                R=_get(cfg.R0, a, 0.0),
                insolvent=False,
                active_sanctions=set(),
                prev_actions=set(),
                commitment_timer=0
            )
        
        self.A = nx.Graph(); self.A.add_nodes_from(cfg.actors); self.A.add_edges_from(cfg.A0_edges)
        
        for u, v in self.A.edges:
            self.state[u].allies.add(v)
            self.state[v].allies.add(u)
        
        if cfg.X0 is not None:
            self.X = cfg.X0.copy().reindex(index=cfg.actors, columns=cfg.actors, fill_value=0.0)
        else:
            self.X = pd.DataFrame(0.0, index=cfg.actors, columns=cfg.actors)
        np.fill_diagonal(self.X.values, 0.0)

        self.X_regimes = dict(cfg.X_regimes) if cfg.X_regimes else {}
        for snd in cfg.actors:
            for tgt in cfg.actors:
                if snd == tgt: continue
                if float(self.X.loc[snd, tgt]) > 0.0 and (snd, tgt) not in self.X_regimes:
                    self.X_regimes[(snd, tgt)] = {cp: "LICENSE_FEE" for cp in cfg.chokepoints}
                    
        for (snd, tgt), reg in self.X_regimes.items():
            if any(v == "BAN" for v in reg.values()):
                self.state[snd].active_sanctions.add(tgt)

        eta_arr = cfg.exog.eta_series if cfg.exog.eta_series is not None else np.ones(cfg.T)
        self.eta = np.maximum(1.0, eta_arr)
        self.w_eta, self.wC, self.wD, self.wE, self.wS = cfg.weights.norm_cap_weights()

        self.crisis_timer = 0
        self.logs: List[StepLog] = []
        self._race_prev: float = 0.0 
        self.treaty_strength: float = 0.0
        self._g_up_ctr = defaultdict(int)
        self.beliefs: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {a:{} for a in self.cfg.actors}
        for i in self.cfg.actors:
            self.beliefs[i] = {
                j: {
                    "Cap": None, "S": None, "G": None, 
                    "G_sec": None, "G_eval": None, 
                    "C": None, "D": None, "E": None,
                    "R": None, "B": None 
                } for j in self.cfg.actors
            }
    
    def _calculate_training_flow(self, a: str, t: int) -> Tuple[float, float]:
        if self.freeze_timer[a] > 0 or self.state[a].insolvent:
            return 0.0, 0.0 
        
        s = self.state[a]
        r = self.cfg.rates
        
        # 1. Get Heterogeneity Weights
        w_eta = self._het("w_eta", a, self.w_eta)
        wD    = self._het("w_D",   a, self.wD)
        wS    = self._het("w_S",   a, self.wS)
        
        # 2. Blockade Logic
        is_blockaded = False
        for (snd, tgt), reg in self.X_regimes.items():
            if tgt == a and any(v == "BAN" for v in reg.values()):
                if self.state[snd].K > 0.3 * s.K:
                    is_blockaded = True
                    break
        
        throttle = 0.20 if is_blockaded else 1.0 
        
        # 3. Compute Internal Output
        available_power = s.E * r.energy_efficiency
        utilized_compute = soft_min(s.C, available_power, smooth=1.0)
        utilized_compute = max(0.10 * s.C, utilized_compute) 
        utilized_compute *= throttle 

        gov_drag = 1.0 - (getattr(r, "safety_compute_drag", 0.0) * s.G)
        utilized_compute *= max(0.1, gov_drag)
        
        process_drag = 1.0 / (1.0 + r.kappa_G * s.G)
        tech_level = self.eta[t] ** w_eta
        raw_output = tech_level * (utilized_compute ** self.wC) * (s.D ** wD) * (max(s.S, self.cfg.epsilon) ** wS)
        
        # 4. Spillovers & Diffusion
        global_max_K = max([st.K for st in self.state.values()])
        gap = max(0.0, global_max_K - s.K)
        
        max_C = max([st.C for st in self.state.values()])
        hardware_ratio = s.C / (max_C + 1e-9)
        
        # Absorptive capacity
        absorptive_capacity = (s.D ** 0.5) * (1.0 - 0.5 * s.G_sec) * hardware_ratio
        
        # [CRITICAL STEP] Initialize 'spillover' with standard diffusion FIRST
        spillover = gap * getattr(r, "knowledge_diffusion_rate", 0.01) * absorptive_capacity
        
        # [NEW LOGIC] Then ADD the Allied Innovation Transfer
        # Find the smartest ALLY
        if s.allies:
            ally_max_K = max([self.state[ally].K for ally in s.allies])
            
            # If an ally is ahead of me, I learn from them at the accelerated 'alliance' rate
            if ally_max_K > s.K:
                ally_gap = ally_max_K - s.K
                # This represents tech transfer agreements, joint research, shared IP
                ally_transfer = ally_gap * getattr(r, "alliance_innovation_rate", 0.0) * absorptive_capacity
                spillover += ally_transfer
        
        # 5. Final Output
        exponent = getattr(r, "knowledge_production_exponent", 0.6)
        internal_flow = (raw_output * process_drag) ** exponent
        
        return float(internal_flow), float(spillover)
    
    def _het(self, key: str, actor: str, default: float) -> float:
        het = getattr(self.cfg.het, key, None)
        if isinstance(het, dict) and actor in het: return float(het[actor])
        return float(default)

    def _hets(self, a: str, **defaults) -> dict:
        return {k: self._het(k, a, v) for k, v in defaults.items()}
        
    def _talent_dynamics(self):
        r = self.cfg.rates; cfg = self.cfg
        agglomeration_rate = getattr(r, "talent_agglomeration_rate", 0.015)
        base_growth = getattr(r, "talent_growth_base", 0.005)
        
        # [FIX] Logistic Constraint on D to prevent t=50 explosion
        for a in cfg.actors:
            s = self.state[a]
            # Saturation term: Growth slows as D approaches 10.0
            saturation = max(0.0, 1.0 - (s.D / self._max_possible_D))
            
            cluster_effect = agglomeration_rate * np.log1p(s.K) * s.G
            growth = (base_growth + cluster_effect) * saturation
            s.D *= (1.0 + growth)
            
        # ... [Migration logic unchanged] ...
        attractiveness = {}
        total_pull = 0.0
        for a in cfg.actors:
            s = self.state[a]
            # Attractiveness also capped by stability
            score = (max(0.1, s.B)**0.2) * (max(0.1, s.K)**0.4) * (1.0 + np.tanh(s.R))
            # --- Allied Visa/Market Bonus ---
            # "If I move there, I have access to X other markets/universities"
            num_allies = len(s.allies)
            
            # Bonus multiplier. 3 allies @ 0.10 bonus = 1.3x attractiveness
            network_multiplier = 1.0 + (num_allies * getattr(r, "alliance_talent_bonus", 0.0))
            
            score *= network_multiplier
            
            attractiveness[a] = score
            total_pull += score
        
        if total_pull > 0:
            migration_fraction = r.migration_fraction
            global_pool = 0.0
            for a in cfg.actors:
                leaving = self.state[a].D * migration_fraction
                self.state[a].D -= leaving
                global_pool += leaving
            
            for a in cfg.actors:
                share = attractiveness[a] / total_pull
                self.state[a].D += global_pool * share
    
    # geoai/engine.py

    def _economic_step(self, a: str, maintenance_bill: float, invest_bill: float, export_revenue: float):
        s = self.state[a]; r = self.cfg.rates
        
        debt_ratio = s.Debt / (s.GDP + 1e-9)
        r_min = r.yield_curve_min_rate; r_max = r.yield_curve_max_rate
        k = r.yield_curve_steepness
        
        # --- [UPDATED] Actor-Specific Yield Curve ---
        # Look up the specific threshold for this actor
        limit = self._het("insolvency_threshold", a, r.insolvency_threshold_center)
        
        # Scientific adjustment: The "midpoint" of the yield curve (where rates spike)
        # should be roughly 40-50% of the way to the hard insolvency limit.
        # e.g., if Limit=4.5 (US), rates spike at 2.0. If Limit=1.2 (India), rates spike at 0.6.
        x0 = limit * 0.45 
        
        effective_rate = r_min + (r_max - r_min) / (1.0 + np.exp(-k * (debt_ratio - x0)))
        
        s.current_risk_premium = effective_rate - r_min
        s.risk_premium = s.current_risk_premium
        interest_payment = s.Debt * effective_rate
        
        # ... [Keep remaining GDP Growth & Balance logic] ...
        safe_debt = min(debt_ratio, 50.0)
        drag_factor = 0.01 * (safe_debt ** 2)
        inflation_drag = 1.0 + drag_factor
        
        s.GDP *= (1.0 / (1.0 + drag_factor)) 
        gov_bonus = getattr(r, "governance_gdp_bonus", 0.008) * s.G
        base_growth = getattr(r, "gdp_growth_base", 0.005)
        s.GDP *= (1.0 + base_growth + gov_bonus)

        local_tax_rate = self._het("tax_rate", a, getattr(r, "tax_rate", 0.20))
        revenue = (s.GDP * local_tax_rate) + export_revenue
        
        total_expenses = (maintenance_bill + invest_bill) * inflation_drag + interest_payment
        net_balance = revenue - total_expenses
        s.B += net_balance
        
        reserves_cap = 1.0 * s.GDP 
        if s.B > reserves_cap:
            excess = s.B - reserves_cap
            burn = excess * 0.05 
            s.B -= burn

        if s.B < 0:
            new_debt = abs(s.B); s.Debt += new_debt; s.B = 0.0 
        elif s.B > 0 and s.Debt > 0:
            repay = min(s.B, s.Debt); s.Debt -= repay; s.B -= repay
    
    def _export_pressure(self, target: str) -> Tuple[float,float]:
        cfg, r = self.cfg, self.cfg.rates
        bite_train = bite_infer = 0.0
        for (snd, tgt), regmap in self.X_regimes.items():
            if tgt != target or snd == tgt: continue
            sender_weight = (1.0 + 0.3 * np.tanh(self.state[snd].R)) * self._het("export_sender_power", snd, 1.0)
            for cp in cfg.chokepoints:
                regime = regmap.get(cp, "OPEN")
                base = r.regime_bite.get(regime, 0.0)
                bite_train += sender_weight * base * r.w_choke_train.get(cp, 0.0)
                bite_infer += sender_weight * base * r.w_choke_infer.get(cp, 0.0)
        normT = max(1e-9, sum(r.w_choke_train.values()))
        normI = max(1e-9, sum(r.w_choke_infer.values()))
        return float(min(1.0, bite_train/normT)), float(min(1.0, bite_infer/normI))

    def _leak_terms(self, a:str):
        r = self.cfg.rates
        allies = list(self.A.neighbors(a))
        leak_base = 1.0 - r.lambda_leak*(0.6*mean_or0([self.state[x].S for x in allies]) +
                                         0.4*mean_or0([self.state[k].S for k in self.cfg.actors if k!=a and k not in allies]))
        subst = min(1.0, (self.state[a].D + self.state[a].E) / max(r.subst_norm, self.cfg.epsilon))
        leakT = float(np.clip(leak_base*(1 - r.phi_subst_train*subst), 0.0, 1.0))
        leakI = float(np.clip(leak_base*(1 - r.phi_subst_infer*subst), 0.0, 1.0))
        return leakT, leakI, 0.5*(leakT+leakI)

    def _effective_bite(self, a:str, leakT:float, leakI:float):
        r = self.cfg.rates
        biteT, biteI = self._export_pressure(a)
        rep = 1.0 + (-0.2*np.tanh(self.state[a].R))
        hT = float(np.clip(biteT*leakT*rep, 0.0, r.h_cap))
        hI = float(np.clip(biteI*leakI*rep, 0.0, r.h_cap))
        return biteT, hT, hI
    
    def _race_intensity(self, action_sets: Dict[str, Set[str]]) -> float:
        invests = sum(((INVEST_C in u)+(INVEST_D in u)+(INVEST_E in u)) for u in action_sets.values())
        n = max(1, len(self.cfg.actors))
        raw = self.cfg.rates.alpha_R * (float(invests) / n)
        rho = float(self.cfg.rates.rho_race)
        race = rho * self._race_prev + (1.0 - rho) * raw
        self._race_prev = race
        return race

    def _apply_gov_delta(self, s, acts: Set[str], dG: float):
        for act, attr in G_UP.items():
            if act in acts: setattr(s, attr, clip01(getattr(s, attr) + dG))
        for act, attr in G_DN.items():
            if act in acts: setattr(s, attr, clip01(getattr(s, attr) - dG))

    def _calculate_ban_cost_multiplier(self, actor: str) -> float:
        banned = False
        for (snd, tgt), regmap in self.X_regimes.items():
            if tgt == actor and (regmap.get("LITHO") == "BAN" or regmap.get("EDA") == "BAN"):
                banned = True; break
        return self.cfg.rates.ban_cost_mult if banned else 1.0

    # Logic: Throttle affects maintenance, not just growth
    def _calculate_ban_throttle(self, actor: str) -> float:
        throttle = 1.0
        for (snd, tgt), regmap in self.X_regimes.items():
            if tgt == actor:
                # Litho is the choke point. If banned, you are stuck with legacy nodes.
                if regmap.get("LITHO") == "BAN": throttle *= 0.25 
                # EDA is the design bottleneck.
                if regmap.get("EDA") == "BAN": throttle *= 0.75
        return throttle

    def _act_cost(self, a:str, act:str, base:dict, gup_mult:float, gdn_mult:float, surcharge:float, c_mult:float)->float:
        c = base.get(act, 0.0)
        current_C = self.state[a].C
        cost_scaling = 1.15 ** current_C
        c *= cost_scaling
        if act == INVEST_C: c *= c_mult 
        if act in UP_VARIANTS: c = c * gup_mult + surcharge
        if act in DN_VARIANTS: c = c * gdn_mult
        return c

    def _apply_event(self, ev: dict):
        kind = ev.get("fn", "")
        args = ev.get("args", {})
        
        if kind == "override":
            for scope, mapping in args.items():
                if scope == "rates":
                     for k, v in mapping.items(): setattr(self.cfg.rates, k, v)
                elif scope == "het":
                     for pkey, pd in mapping.items(): getattr(self.cfg.het, pkey).update(pd)
        
        elif kind == "blockade":
            snd = args.get("sender")
            tgt = args.get("target")
            regimes = args.get("regimes", {})
            
            if snd and tgt and snd in self.cfg.actors and tgt in self.cfg.actors:
                is_ban = any(v == "BAN" for v in regimes.values())
                if is_ban:
                    self.state[snd].active_sanctions.add(tgt)
                
                key = (snd, tgt)
                if key not in self.X_regimes:
                    self.X_regimes[key] = {cp: "OPEN" for cp in self.cfg.chokepoints}
                self.X_regimes[key].update(regimes)
    
    def _sigma_mu(self, i, j):
        cfg = self.cfg
        if i == j:
            sigma, mu = get_map(cfg.sigma_self_map, i, cfg.sigma_self), 0.0
        elif self.A.has_edge(i, j):
            sigma, mu = get_map(cfg.sigma_ally_map, i, cfg.sigma_ally), 0.0
            sigma *= (0.6 + 0.4 * min(self.state[i].G_eval, self.state[j].G_eval))
        else:
            sigma, mu = get_map(cfg.sigma_rival_map, i, cfg.sigma_rival), get_map(cfg.mu_bias_map, (i, j), cfg.mu_bias_rival)
        
        spy_factor = 1.0 / (1.0 + cfg.rates.spy_efficiency * np.log1p(max(0, self.state[i].B)))
        sigma *= spy_factor
        if i != j: sigma *= (1.0 + cfg.rates.counter_spy_efficiency * self.state[j].G_sec)
        return sigma, mu

    def _obs_for_actor(self, i: str, t: int) -> Dict[str, Dict[str, float]]:
        cfg = self.cfg; obs = {}
        for j in cfg.actors:
            s = self.state[j]
            sigma, mu = self._sigma_mu(i, j)
            def _noise(val): return float(val * np.exp(self.rng.normal(mu, sigma)))
            obs[j] = {
                "C_hat": _noise(s.C), "D_hat": _noise(s.D), "E_hat": _noise(s.E),
                "K_hat": _noise(s.K), "R_hat": float(s.R + self.rng.normal(0, sigma)), 
                "B_hat": _noise(s.B),
                "G_sec": float(expit(logit(s.G_sec, cfg.epsilon) + self.rng.normal(mu, sigma))),
                "G_eval": float(expit(logit(s.G_eval, cfg.epsilon) + self.rng.normal(mu, sigma))),
                "S": float(expit(logit(s.S, cfg.epsilon) + self.rng.normal(mu, sigma)))
            }
        return obs

    def _update_beliefs(self, i: str, obs_i: Dict[str, Dict[str,float]]):
        beta = (self.cfg.belief_beta_map or {}).get(i, self.cfg.belief_beta)
        for j in self.cfg.actors:
            b = self.beliefs[i][j]
            mapping = [
                ("Cap", "K_hat"), ("S", "S"), ("G_sec", "G_sec"), ("G_eval", "G_eval"),
                ("C", "C_hat"), ("D", "D_hat"), ("E", "E_hat"), 
                ("R", "R_hat"), ("B", "B_hat")
            ]
            for key, obs_key in mapping:
                new = obs_i[j].get(obs_key)
                if new is not None:
                    b[key] = float(new) if b[key] is None else float(beta * b[key] + (1 - beta) * new)

    def decide(self, t: int, policy, policy_obs=None) -> Dict[str, Dict[str, Set[str]]]:
        self.cfg.race_intensity = self._race_prev 
        if policy_obs is not None and self.cfg.flags.enable_observation_asymmetry:
             decision = {}
             for a in self.cfg.actors:
                 obs_i = self._obs_for_actor(a, t)
                 self._update_beliefs(a, obs_i)
                 acts = set(policy_obs(t, obs_i, a, self.beliefs[a]))
                 decision[a] = {"actions": acts}
             return decision

        if self.cfg.flags.enable_observation_asymmetry:
            decision = {}
            for a in self.cfg.actors:
                obs_i = self._obs_for_actor(a, t)
                self._update_beliefs(a, obs_i)
            
            for a in self.cfg.actors:
                proxy_state = {}
                for target in self.cfg.actors:
                    real = self.state[target]
                    if target == a: proxy_state[target] = real
                    else:
                        b = self.beliefs[a][target]
                        def _v(k, fallback): return b.get(k) if b.get(k) is not None else fallback
                        proxy_state[target] = ActorState(
                            C=_v("C", real.C), D=_v("D", real.D), E=_v("E", real.E),
                            K=_v("Cap", real.K), S=_v("S", real.S),
                            G_sec=_v("G_sec", real.G_sec), G_eval=_v("G_eval", real.G_eval),
                            G_deploy=real.G_deploy, GDP=real.GDP, Debt=real.Debt, 
                            B=_v("B", real.B), R=_v("R", real.R), 
                            S_backlog=real.S_backlog, E_backlog=deque(),
                            insolvent=real.insolvent, active_sanctions=real.active_sanctions,
                            prev_actions=real.prev_actions, allies=real.allies,
                            commitment_timer=real.commitment_timer if target == a else 0 
                        )
                all_decisions = policy(t, proxy_state, self.cfg)
                decision[a] = all_decisions.get(a, {"actions": set()})
            return decision

        return policy(t, self.state, self.cfg)

    def _resolve_geopolitics(self, decision: Dict[str, Dict], action_bill: Dict[str, float]):
        r = self.cfg.rates
        for actor, d in decision.items():
            if "sanctions" in d:
                targets = d["sanctions"]
                current = self.state[actor].active_sanctions
                if isinstance(targets, set): new_targets = targets
                else: new_targets = set(targets.keys())
                for target in self.cfg.actors:
                    if target == actor: continue
                    key = (actor, target)
                    if target in new_targets and target not in current:
                        self.state[actor].active_sanctions.add(target)
                        self.X_regimes[key] = {cp: "BAN" for cp in self.cfg.chokepoints}
                    if target not in new_targets and target in current:
                        self.state[actor].active_sanctions.remove(target)
                        self.X_regimes[key] = {cp: "LICENSE_FEE" for cp in self.cfg.chokepoints}

        for actor, d in decision.items():
            covert = d.get("covert", {}) 
            for target, op_type in covert.items():
                if target == actor or self.state[actor].insolvent: continue
                s_actor = self.state[actor]; s_target = self.state[target]
                
                cost = r.cost_sabotage if op_type == COVERT_SABOTAGE else r.cost_theft
                if s_actor.B < cost: continue
                action_bill[actor] += cost
                
                base_prob = r.prob_sabotage_success if op_type == COVERT_SABOTAGE else r.prob_theft_success
                prob = base_prob * (1.0 + r.spy_efficiency * np.log1p(s_actor.G_sec))
                prob /= (1.0 + r.counter_spy_efficiency * np.log1p(s_target.G_sec))
                
                success = self.rng.random() < clip01(prob)
                if success:
                    if op_type == COVERT_SABOTAGE:
                        if len(s_target.E_backlog) > 0:
                            new_len = int(len(s_target.E_backlog) / 2)
                            s_target.E_backlog = deque(list(s_target.E_backlog)[:new_len], maxlen=s_target.E_backlog.maxlen)
                            s_target.R -= 0.2
                    elif op_type == COVERT_THEFT:
                        stolen = s_target.D * 0.05
                        s_target.D -= stolen; s_actor.D += stolen

    def step(self, t: int, policy=None, policy_obs=None) -> Tuple[bool, Dict[str, float]]:
        cfg, r = self.cfg, self.cfg.rates
        cfg.rates.energy_efficiency *= getattr(r, "efficiency_growth_rate", 1.015)
        
        in_crisis = (self.crisis_timer > 0)
        if in_crisis: self.crisis_timer -= 1
        for ev in getattr(self.cfg, "events", []):
            if ev.get("t", -1) == t: self._apply_event(ev)

        for a in self.cfg.actors:
            if self.state[a].commitment_timer > 0: self.state[a].commitment_timer -= 1

        decision = self.decide(t, policy, policy_obs)
        
        treaty_power = 0.0
        total_power = sum(self.state[a].K for a in cfg.actors) + 1e-9
        for a, d in decision.items():
            if SIGN_TREATY in d.get("actions", set()):
                treaty_power += self.state[a].K
        
        participation = treaty_power / total_power
        if participation > 0.50:
            self.treaty_strength = clip01(self.treaty_strength + 0.05)
        else:
            self.treaty_strength = clip01(self.treaty_strength - 0.05)
            
        U_sets = {a:set(decision[a].get("actions", set())) for a in cfg.actors}
        raw_race_int = self._race_intensity(U_sets)
        dampener = 1.0 - (0.5 * self.treaty_strength)
        final_race_int = raw_race_int * dampener
        self._race_prev = final_race_int

        action_bill = {a:0.0 for a in cfg.actors}
        self._resolve_geopolitics(decision, action_bill)

        credit_throttled = set()
        for a in cfg.actors:
            debt_ratio = self.state[a].Debt / (self.state[a].GDP + 1e-9)
            if debt_ratio > r.debt_hard_limit:
                credit_throttled.add(a)
                self.state[a].C *= 0.85 
            
            if self.state[a].insolvent:
                decision[a]["actions"] = set()
                self.state[a].G_sec = max(0.0, self.state[a].G_sec - 0.05)
            elif self.freeze_timer[a] > 0:
                self.freeze_timer[a] -= 1
                decision[a]["actions"] = set() 
                self.state[a].G_sec = max(self.state[a].G_sec, 0.95)

            c_mult = self._calculate_ban_cost_multiplier(a) 
            if c_mult > 1.0: 
                decay = getattr(r, "ban_efficiency_decay", 0.0)
                self.state[a].E *= (1.0 - decay)

        for a, d in decision.items():
            acts = d.get("actions", set())
            if FORM_ALLIANCE in acts:
                self.state[a].political_capital = min(1.0, self.state[a].political_capital + 0.15)
                self.state[a].G_sec = min(0.99, self.state[a].G_sec + 0.05)
            if SANCTION_RIVAL in acts:
                self.state[a].political_capital = max(0.0, self.state[a].political_capital - 0.10)

            for b in d.get("ally_add", []): 
                if b != a: 
                    self.A.add_edge(a, b)
                    self.state[a].allies.add(b); self.state[b].allies.add(a)
            for b in d.get("ally_remove", []): 
                if self.A.has_edge(a, b): 
                    self.A.remove_edge(a, b)
                    if b in self.state[a].allies: self.state[a].allies.remove(b)
                    if a in self.state[b].allies: self.state[b].allies.remove(a)

        demand_multiplier = 1.0
        if cfg.flags.enable_price_mechanism:
            demand_multiplier = 1.0 + 0.5 * self._race_prev
            
        COST = {
            INVEST_C: self.cfg.cost_C * demand_multiplier, 
            INVEST_D: self.cfg.cost_D, 
            INVEST_E: self.cfg.cost_E,
            GOV_UP: self.cfg.cost_Gup, GOV_DOWN: self.cfg.cost_Gdown,
            **{k:self.cfg.cost_Gup for k in UP_VARIANTS}, 
            **{k:self.cfg.cost_Gdown for k in DN_VARIANTS}
        }
        
        # [REALPOLITIK FIX]
        pol_modifiers = {}
        for a in cfg.actors:
            s = self.state[a]
            maint_discount = 1.0 - (0.1 * s.political_capital)
            
            # Dynamic Network Effect. Bonus depends on the Sum of Allies' Capability, not just count.
            # "Tech Transfer" is stronger if you ally with the leader.
            ally_power = sum(self.state[ally].K for ally in s.allies)
            global_power = sum(self.state[x].K for x in cfg.actors) + 1e-9
            
            # You get a efficiency boost proportional to the % of global power you are allied with.
            # Max bonus capped at ~15% efficiency
            network_bonus = 0.15 * (ally_power / global_power)
            
            pol_modifiers[a] = {"maint": maint_discount, "beta_bonus": network_bonus}

        maintenance_bill = {a:0.0 for a in cfg.actors}
        rep_delta = {a:0.0 for a in cfg.actors}
        c_mult_map = {a: self._calculate_ban_cost_multiplier(a) for a in cfg.actors}
        throttle_map = {a: self._calculate_ban_throttle(a) for a in cfg.actors}
        gdp_impact = {a: 0.0 for a in cfg.actors}

        for a, d in decision.items():
            maint_C = self.state[a].C * r.maintenance_rate_C * c_mult_map[a] * pol_modifiers[a]["maint"]
            maint_K = self.state[a].K * getattr(r, "maintenance_rate_K", 0.05)
            # Charge for every active ally
            maint_A = len(self.state[a].allies) * getattr(r, "maintenance_rate_alliance", 0.0) 
            maintenance_bill[a] = maint_C + maint_K + maint_A
            
            if self.state[a].insolvent: continue

            acts = sorted({LEGACY.get(x, x) for x in d.get("actions", set())})
            valid_acts = []
            special_acts = {SANCTION_RIVAL, LIFT_SANCTION, COVERT_SABOTAGE, COVERT_THEFT, SIGN_TREATY, FORM_ALLIANCE}
            
            gup_mult = self._het("cost_Gup_mult", a, 1.0)
            gdn_mult = self._het("cost_Gdown_mult", a, 1.0)
            base_surcharge = 0.03 * min(3, self._g_up_ctr[a])
            
            for act in acts:
                if act in special_acts: 
                    valid_acts.append(act) 
                    if act == FORM_ALLIANCE:
                        action_bill[a] += getattr(r, "cost_alliance_formation", 0.0)
                        
                    continue
                
                cost = self._act_cost(a, act, COST, gup_mult, gdn_mult, base_surcharge, c_mult_map[a])
                projected_debt = self.state[a].Debt + max(0, cost - self.state[a].B)
                
                if projected_debt < self.state[a].GDP * 5.0:
                    valid_acts.append(act)
                    action_bill[a] += cost
                    if act in DN_VARIANTS: rep_delta[a] -= r.r_down
                    if act in UP_VARIANTS: self._g_up_ctr[a] += 1
            
            final_acts = set(valid_acts)
            decision[a]["actions"] = final_acts
            self.state[a].prev_actions = final_acts 
            self._apply_gov_delta(self.state[a], final_acts, r.Delta_G)

        training_flows = {}
        spillover_flows = {} 
        
        for a, d in decision.items():
            u = d.get("actions", set())
            h = self._hets(a, beta_C=r.beta_C, kappa_G=r.kappa_G)
            crisis_mult = r.crisis_growth_mult if (cfg.flags.enable_crisis_mode and in_crisis) else 1.0
            
            # [LOGIC CHANGE 1] Get the physical throttle (Supply Chain Health)
            throttle = throttle_map[a]
            if a in credit_throttled: throttle *= 0.10

            effective_beta_C = h["beta_C"] + pol_modifiers[a]["beta_bonus"]
            invest_C_bonus = effective_beta_C if INVEST_C in u else 0.0
            
            invest_E_bonus = r.beta_E if INVEST_E in u else 0.0
            invest_D_bonus = getattr(r, "beta_D", 0.0) if INVEST_D in u else 0.0
            
            if invest_D_bonus > 0.0: self.state[a].D *= (1.0 + invest_D_bonus * crisis_mult * throttle)
            # [FIX] Blockade affects Energy too (Capital Flight)
            e_throttle = 1.0 if throttle >= 0.9 else 0.5
            
            if invest_E_bonus > 0: self.state[a].E_backlog.append((self._gE_step + invest_E_bonus) * e_throttle)
            else: self.state[a].E_backlog.append(self._gE_step * e_throttle)
            
            mature_E = self.state[a].E_backlog.popleft()
            
            is_blockaded = False
            for (snd, tgt), reg in self.X_regimes.items():
                if tgt == a and any(v == "BAN" for v in reg.values()):
                    if self.state[snd].K > 0.2 * self.state[a].K:
                        is_blockaded = True
                        break
            
            # [LOGIC CHANGE 2] Spare Parts Crisis
            # If throttle < 1.0, it means supply chains are cut. 
            # Existing machines break faster because you can't get lenses/mirrors.
            # Base deprec is ~0.06. If throttle is 0.25 (Litho Ban), deprec jumps to ~0.20
            # Formula: Base + (1 - Throttle) * 0.20
            base_deprec = r.depreciation_C
            supply_chain_stress = (1.0 - throttle) * 0.20
            real_deprec_C = base_deprec + supply_chain_stress
            
            gC = (1.0 - real_deprec_C) + (self._gC_step + invest_C_bonus) * crisis_mult * throttle
            gC = max(0.5, gC) 
            gE = 1.0 + mature_E * crisis_mult
            
            self.state[a].C *= gC; self.state[a].E *= gE
            
            leakT, leakI, _ = self._leak_terms(a)
            _, h_train, _ = self._effective_bite(a, leakT, 0)
            
            internal, spillover = self._calculate_training_flow(a, t)
            internal *= (1.0 - h_train)
            
            # [FIX] Effective Bite logic
            leakT, _, _ = self._leak_terms(a)
            _, h_train, _ = self._effective_bite(a, leakT, 0)
            internal *= (1.0 - h_train)
            
            total_flow = internal + spillover
            training_flows[a] = total_flow
            spillover_flows[a] = spillover
            
            self.state[a].K = self.state[a].K * (1.0 - getattr(r, "knowledge_depreciation", 0.02)) + total_flow

        self._talent_dynamics()
        
        export_revenue = {a: 0.0 for a in cfg.actors}
        if r.export_fee_share > 0:
             for snd in cfg.actors:
                 for tgt in cfg.actors:
                     if snd == tgt: continue
                     val = float(self.X.loc[snd, tgt])
                     if val > 0:
                         fee = r.export_fee_share * r.export_fee_scale * (1-np.exp(-val)) * self.state[snd].GDP
                         export_revenue[snd] += fee
        
        for a in cfg.actors:
            self._economic_step(a, maintenance_bill[a], action_bill[a], export_revenue[a])

        U_sets = {a:set(decision[a].get("actions", set())) for a in cfg.actors}
        race_int = self._race_intensity(U_sets)
        total_K = sum(self.state[x].K for x in cfg.actors) + cfg.epsilon
        misuse_flags, esc_flags, any_esc = {}, {}, False
        
        for a in cfg.actors:
            s = self.state[a]
            if s.insolvent: 
                misuse_flags[a]=0; esc_flags[a]=0; continue
            
            k_share = s.K / total_K
            haz_base = (r.crisis_hazard_mult if in_crisis else 1.0) * (1.0 - s.G_sec) * (1.0 + r.theta_R * race_int)
            pmis = r.lambda_mis * haz_base * (1 - s.S)
            misuse = int(self.rng.random() < clip01(pmis)); misuse_flags[a] = misuse
            pesc = r.lambda_esc * haz_base * k_share
            esc = int(self.rng.random() < clip01(pesc)); esc_flags[a] = esc
            any_esc = any_esc or bool(esc)

            if misuse: 
                rep_delta[a] -= r.r_mis; s.C *= 0.90; s.D *= 0.90 
                hit = getattr(r, "gdp_hit_misuse", 0.05); gdp_impact[a] += hit
                s.GDP *= (1.0 - hit); self.freeze_timer[a] += 2 
            if esc:    
                rep_delta[a] -= r.r_esc; s.C *= 0.75; s.K *= 0.80 
                hit = getattr(r, "gdp_hit_esc", 0.15); gdp_impact[a] += hit
                s.GDP *= (1.0 - hit); self.freeze_timer[a] += 4

        if any_esc and cfg.flags.enable_crisis_mode:
            self.crisis_timer = max(self.crisis_timer, r.crisis_steps)
        
        for a in cfg.actors:
            s = self.state[a]
            fear = getattr(s, "public_fear", 0.0) * r.fear_decay
            if misuse_flags[a] or esc_flags[a]: fear += r.fear_shock_misuse
            fear += 0.10 * race_int; fear -= 0.08 * s.G_sec 
            s.public_fear = float(np.clip(fear, 0.0, 1.0))
            pc = getattr(s, "political_capital", 0.0) * r.political_capital_decay
            if gdp_impact[a] <= 0.0 and not (cfg.flags.enable_crisis_mode and self.crisis_timer > 0): pc += 0.05
            pc -= 0.10 * s.public_fear
            if cfg.flags.enable_crisis_mode and self.crisis_timer > 0: pc -= 0.05
            s.political_capital = float(np.clip(pc, 0.0, 1.0))

        # [NEW] Entrapment Risk (Shared Hazards)
        for a in cfg.actors:
            if misuse_flags[a] or esc_flags[a]:
                # If Actor A screws up...
                for ally in self.state[a].allies:
                    # ...Ally B takes a reputation hit (Guilt by Association)
                    # 20% of the shock transfers to allies
                    shock = r.fear_shock_misuse * 0.20
                    self.state[ally].public_fear += shock
                    self.state[ally].political_capital -= 0.05

        caps = {a: self.state[a].K for a in cfg.actors}
        for a in cfg.actors:
            s = self.state[a]
            s.R = float(r.r_decay * (s.R + rep_delta[a]))
            U = (cfg.weights.omega_E * np.log1p(s.K)) + (cfg.weights.omega_S * (s.K/total_K))
            leakT, _, _ = self._leak_terms(a)
            _, h_train, _ = self._effective_bite(a, leakT, 0)
            
            errs = []
            perceived_caps = []
            for target in cfg.actors:
                if target == a: continue
                real_K = self.state[target].K
                believed_K = self.beliefs[a][target].get("Cap", real_K)
                if believed_K is None: believed_K = real_K
                if real_K > 1e-6: errs.append(abs(believed_K - real_K) / real_K)
                perceived_caps.append(believed_K)
            
            avg_err = mean_or0(errs)
            log_belief_cap = max(perceived_caps) if perceived_caps else 0.0

            # [FIX] Capture granular perceptions for the Fog of War plot
            # We save what Actor 'a' thinks about every 'target's' Capability (K)
            current_perceptions = {}
            for target in cfg.actors:
                # Get the belief from the belief state (or default to 0 if missing)
                val = self.beliefs[a][target].get("Cap")
                current_perceptions[target] = float(val) if val is not None else 0.0

            self.logs.append(StepLog(
                t=t, actor=a,
                C=s.C, D=s.D, E=s.E, K=s.K,
                GDP=s.GDP, Debt=s.Debt, B=s.B, R=s.R,
                G=s.G, G_sec=s.G_sec, G_eval=s.G_eval, G_deploy=s.G_deploy, S=s.S,
                Cap=s.K, Cap_Flow=training_flows[a], Cap_Effective=s.K,
                Econ=0.0, Strat=0.0, Stab=0.0, U=U,
                misuse=misuse_flags[a], escalation=esc_flags[a], crisis_active=int(in_crisis),
                actions=",".join(sorted(U_sets[a])), action_cost=action_bill[a],
                race_intensity=race_int, total_capability=total_K,
                incoming_controls=0.0, leak_factor=leakT, openness=0.0,
                belief_Cap=log_belief_cap, belief_S=0.0, belief_G=0.0, belief_G_sec=0.0,
                h_effective=h_train, deliveries_eff=0.0, 
                belief_error=avg_err, 
                maintenance_cost=maintenance_bill[a],
                energy_efficiency=r.energy_efficiency,
                insolvent=int(s.insolvent),
                gdp_penalty=gdp_impact[a],
                insolvency_event=int(s.insolvency_event),
                risk_premium=s.current_risk_premium,
                collapse_prob=s.current_collapse_prob,
                commitment_timer=s.commitment_timer,
                spillover_flow=spillover_flows[a], perceptions=current_perceptions
            ))

        return (any_esc and cfg.flags.terminal_on_escalation), caps
    
    def run(self, policy=None, policy_obs=None) -> pd.DataFrame:
        self.logs.clear()
        self.crisis_timer = 0
        if self.cfg.flags.enable_observation_asymmetry:
            for a in self.cfg.actors:
                obs = self._obs_for_actor(a, 0)
                self._update_beliefs(a, obs)

        for t in range(self.cfg.T):
            terminal, caps = self.step(t, policy=policy, policy_obs=policy_obs)
            if terminal: 
                print(f"!!! TERMINAL ESCALATION AT t={t} !!!")
                break
                
        return pd.DataFrame([dc.asdict(l) for l in self.logs])