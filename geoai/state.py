# geoai/state.py
from dataclasses import dataclass, field as dc_field
from collections import deque
from typing import Deque, Set, Optional, Dict
import numpy as np

@dataclass
class ActorState:
    # --- Non-Default Arguments ---
    C: float
    D: float
    E: float
    
    # --- Arguments with Defaults ---
    K: float = 0.1           
    G_sec: float = 0.5
    G_eval: float = 0.5
    G_deploy: float = 0.5
    S: float = 0.8
    
    GDP: float = 1.0
    Debt: float = 0.0
    B: float = 1.0
    R: float = 0.0
    
    insolvent: bool = False
    insolvency_event: bool = False

    S_backlog: Deque[float] = dc_field(default_factory=deque)
    E_backlog: Deque[float] = dc_field(default_factory=deque)

    # Dynamic Geopolitics State
    active_sanctions: Set[str] = dc_field(default_factory=set)
    prev_actions: Set[str] = dc_field(default_factory=set)
    
    # [NEW] Track allies for planning context
    allies: Set[str] = dc_field(default_factory=set)

    # Bounded Rationality
    commitment_timer: int = 0
    
    current_risk_premium: float = 0.0
    current_collapse_prob: float = 0.0
    
    # Domestic Politics
    public_fear: float = 0.1        
    tech_lobby_power: float = 0.5   
    political_capital: float = 1.0  

    @property
    def G(self) -> float:
        return (self.G_sec + self.G_eval + self.G_deploy) / 3.0

    def clone(self) -> 'ActorState':
        new_state = ActorState(
            C=self.C, D=self.D, E=self.E,
            K=self.K, 
            G_sec=self.G_sec, G_eval=self.G_eval, G_deploy=self.G_deploy,
            S=self.S,
            GDP=self.GDP, Debt=self.Debt, B=self.B, R=self.R,
            insolvent=self.insolvent,
            insolvency_event=self.insolvency_event,
            commitment_timer=self.commitment_timer,
            current_risk_premium=self.current_risk_premium,
            current_collapse_prob=self.current_collapse_prob,
            public_fear=self.public_fear,
            tech_lobby_power=self.tech_lobby_power,
            political_capital=self.political_capital
        )
        new_state.S_backlog = deque(self.S_backlog, maxlen=self.S_backlog.maxlen)
        new_state.E_backlog = deque(self.E_backlog, maxlen=self.E_backlog.maxlen)
        new_state.active_sanctions = set(self.active_sanctions)
        new_state.prev_actions = set(self.prev_actions)
        new_state.allies = set(self.allies) 
        
        return new_state

@dataclass
class StepLog:
    t: int
    actor: str
    C: float; D: float; E: float; K: float
    GDP: float; Debt: float; B: float; R: float
    G: float; G_sec: float; G_eval: float; G_deploy: float; S: float
    
    Cap: float
    Cap_Flow: float        
    Cap_Effective: float   
    
    Econ: float; Strat: float; Stab: float; U: float
    misuse: int; escalation: int; crisis_active: int
    actions: str; action_cost: float
    race_intensity: float; total_capability: float
    
    incoming_controls: float
    leak_factor: float
    openness: float
    
    belief_Cap: float; belief_S: float; belief_G: float
    h_effective: float; deliveries_eff: float
    belief_error: float
    maintenance_cost: float
    energy_efficiency: float 
    
    insolvent: int 

    # --- Optional Fields ---
    belief_G_sec: float = np.nan
    insolvency_event: int = 0
    risk_premium: float = 0.0
    collapse_prob: float = 0.0
    gdp_penalty: float = 0.0 
    
    public_fear: float = 0.0
    political_capital: float = 0.0
    commitment_timer: int = 0
    spillover_flow: float = 0.0 # [NEW]
    # [NEW] Real Pairwise Perceptions
    # Stores {target_actor: perceived_K}
    perceptions: Dict[str, float] = dc_field(default_factory=dict)