# geoai/config.py
from __future__ import annotations
import dataclasses as dc
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Tuple, Optional
import pandas as pd

@dataclass
class Weights:
    # --- Production Function Exponents (Cobb-Douglas) ---
    # Source: Epoch AI (2024) "How much does it cost to train frontier AI models?"
    # Findings: Hardware (47-67%), R&D Staff (29-49%), Energy (2-6%).
    
    w_eta: float = 0.20 # Total Factor Productivity (Exogenous Tech Progress)
    w_C:   float = 0.55 # Compute Capital (Hardware). Updated from 0.45 to reflect 2024 hardware dominance.
    w_D:   float = 0.40 # Talent & Data. Updated from 0.25. Staff/IP remains a huge cost driver.
    w_E:   float = 0.05 # Energy Flow. Lowered to 5% to match direct training energy costs.
    w_S:   float = 0.03 # Security/Stability (Institutional Friction)
    
    # Utility Function Weights (Preferences)
    omega_E: float = 1.0
    omega_S: float = 1.0
    omega_B: float = 1.0
    omega_R: float = 0.0
    
    # Planner Weights (Strategic Drivers)
    w_switching: float = 2.0   
    w_credit: float = 5.0      
    w_terminal: float = 20.0   
    
    # Instrumental Weights
    w_pol: float = 50.0 # Political Capital acts as "Hard Currency" for alliances/sanctions
    w_hoard: float = 1.0 # Penalty for hoarding excess cash (B > 50% GDP)
    
    # Scaling Factors (For numerical stability in optimization)
    scale_growth: float = 5.0 
    scale_relative: float = 10.0
    scale_imbalance: float = 15.0
    w_safety: float = 40.0 

    def norm_cap_weights(self) -> Tuple[float, float, float, float, float]:
        s = self.w_eta + self.w_C + self.w_D + self.w_E + self.w_S
        if s <= 0: return (0.0,1.0,0.0,0.0,0.0)
        return (self.w_eta/s, self.w_C/s, self.w_D/s, self.w_E/s, self.w_S/s)

@dataclass
class Rates:
    # --- 1. Physics of the Simulation ---
    
    # Growth Bases (Quarterly)
    # Global real GDP growth projected ~3.2% annually for 2025.
    # 3.2% annual ~= 0.8% quarterly.
    gC_base: float = 0.008  # Baseline economic growth (non-AI).
    gD_base: float = 0.02   # Talent pool grows faster due to education/training.
    gE_base: float = 0.01   # Energy infrastructure grows slowly.
    
    # Depreciation
    # Source: CITP (2025). AI chips have ~1.5-3 years useful economic life.
    # 2 years = 8 quarters. Rate ~= 1/8 = 0.12.
    # We use 0.07 (approx 25-28% annualized) to capture rapid obsolescence (H100 -> Blackwell).
    depreciation_C: float = 0.12 
    knowledge_depreciation: float = 0.06 # Ideas/IP depreciate at similar rates to hardware in fast-moving fields.
    
    # Delays
    # Data center construction lag: 18-24 months. 6 quarters is realistic.
    energy_build_lag: int = 6 
    
    # Investment Efficiency (Marginal Returns)
    beta_C: float = 0.12
    beta_D: float = 0.08
    beta_E: float = 0.04
    
    # --- 2. Asymmetric Conflict Pricing ---
    # Relative costs normalized to GDP scale
    cost_C: float = 0.50 # High capital intensity
    cost_D: float = 0.10 # Talent is expensive but granular
    cost_E: float = 0.35 # Infrastructure is lumpy and costly
    
    # Security Costs
    cost_Gup: float = 0.05 # ~2-3% of GDP equivalent for major defense overhaul
    cost_Gdown: float = 0.05
    cost_sabotage: float = 0.15 
    cost_theft: float = 0.02    
    
    # --- 3. Maintenance & Efficiency ---
    maintenance_rate_C: float = 0.12 # High opex (power, cooling, parts)
    maintenance_rate_K: float = 0.05
    
    # Tech Progress
    efficiency_growth_rate: float = 1.045 # Algorithmic efficiency gains
    knowledge_production_exponent: float = 0.60 
    kappa_G: float = 0.40  
    ban_maintenance_penalty: float = 0.20 # Spare parts starvation effect
    ban_efficiency_decay: float = 0.02 
    
    # Macro-Financial Links
    gdp_growth_base: float = 0.005
    gdp_ai_multiplier: float = 0.003
    governance_gdp_bonus: float = 0.008 
    deployment_friction: float = 0.20
    safety_compute_drag: float = 0.35
    # Cost to perform the "FORM_ALLIANCE" action (Diplomatic push)
    cost_alliance_formation: float = 0.05  # e.g., 5% of GDP equivalent
    # Recurring cost per ally per quarter (Diplomatic missions, aid, coordination)
    maintenance_rate_alliance: float = 0.01 # e.g., 1% of GDP per ally
    # --- Alliance Benefits ---
    # Innovation: How much faster you learn from an ALLY than from a RIVAL
    # If knowledge_diffusion_rate is 0.02, allies might share 4x faster
    alliance_innovation_rate: float = 0.08 
    
    # Talent: How much "Attractiveness" you gain per ally (Visa-free travel, etc.)
    # e.g., 0.10 means each ally boosts your talent magnetism by 10%
    alliance_talent_bonus: float = 0.10
    
    # Tax Rate
    # Source: OECD Corporate Tax Statistics 2024.
    # Average statutory CIT rate is ~21.1%. Weighted average closer to 25%.
    tax_rate: float = 0.21 #
    
    # --- 4. Financials (Sovereign Debt) ---
    # Yield Curve (Quarterly Interest Rates)
    # Base rates ~3-5% annual -> ~0.75-1.25% quarterly.
    # Distress rates >10% annual -> >2.5% quarterly.
    yield_curve_min_rate: float = 0.01   # ~4% annual (Safe)
    yield_curve_max_rate: float = 0.05   # ~20% annual (Distress)
    yield_curve_midpoint: float = 1.60   # Debt/GDP ratio where rates spike
    yield_curve_steepness: float = 6.0   
    
    # Debt Limits
    max_debt_ratio: float = 1.10 
    debt_hard_limit: float = 6.0 
    debt_interest_rate: float = 0.02 
    
    # Insolvency Thresholds
    # Source: IMF/World Bank. "Growth drag" starts at ~90% Debt/GDP.
    # "Crisis" typically 150-200% for majors.
    # We set center at 2.5 (250%) to allow for "Japan-style" leverage before total collapse.
    insolvency_threshold_center: float = 2.5 
    insolvency_threshold_k: float = 10.0
    insolvency_duration_years: float = 6.0
    insolvency_shock_C: float = 0.40
    insolvency_shock_GDP: float = 0.40

    # --- 5. Talent & Demographics ---
    talent_growth_base: float = 0.01
    talent_agglomeration_rate: float = 0.02 
    energy_efficiency: float = 1.32
    
    # --- 6. Geopolitics & Espionage ---
    ban_cost_mult: float = 1.5
    spy_efficiency: float = 0.3      
    counter_spy_efficiency: float = 0.4 
    prob_sabotage_success: float = 0.30
    prob_theft_success: float = 0.40
    migration_fraction: float = 0.01
    
    # Global Talent Pool
    # Source: MacroPolo 2024. US hosts 60%, China 12%.
    # 42% of top-tier researchers are foreign nationals (mobile).
    global_talent_pool: float = 0.42 
    
    talent_stickiness: float = 0.98   
    talent_g_pull: float = 0.25 
    safety_deployment_bonus: float = 0.60 
    
    # --- 7. Learning & Diffusion ---
    learn_zeta: float = 0.020
    learn_lambda: float = 1.00
    Delta_G: float = 0.05
    lambda_A: float = 0.03
    a_trust: float = 0.60
    a_gov:   float = 0.40
    a_press: float = 0.75
    a_bias:  float = 0.00
    lam_cap: float = 0.15
    k_belief_damp: float = 0.75
    
    # --- 8. Conflict & Escalation ---
    alpha_R: float = 0.08
    rho_race: float = 0.60
    lambda_mis: float = 0.10
    lambda_esc: float = 0.12
    gdp_hit_misuse: float = 0.08  
    gdp_hit_esc: float = 0.25     
    theta_G_mis: float = 0.55
    theta_G_esc: float = 0.45
    theta_R: float = 0.20
    theta_R2: float = 0.40
    theta_Scarcity: float = 0.30
    theta_BeliefErr: float = 0.25
    pmax: float = 0.20
    gamma_X: float = 0.35
    
    # --- 9. Leakage & Spillovers ---
    s_regrow: float = 0.02
    lambda_leak: float = 0.05 
    h_cap: float = 0.70         
    phi_subst: float = 0.02    
    subst_norm: float = 4.0    
    phi_subst_train: float = 0.04
    phi_subst_infer: float = 0.14
    xi_S: float = 0.30
    E_norm: float = 0.90
    
    # --- 10. Scars & Crisis ---
    crisis_steps: int = 4
    crisis_growth_mult: float = 0.80 
    crisis_hazard_mult: float = 1.50 
    scars_steps: int = 4
    scars_budget_mult: float = 0.95
    scars_s_regrow_mult: float = 0.40
    
    # Misc
    rho_B: float = 0.95
    k_pi:  float = 0.60
    chi_B: float = 1.00
    export_fee_share: float = 0.0
    export_fee_scale: float = 0.0
    s_pipeline_tau: int = 4
    s_pipeline_base: float = 0.01
    target_S: float = 0.98
    k_backlog: float = 0.04
    r_up: float = 0.01
    r_down: float = 0.01
    r_mis: float = 0.15
    r_esc: float = 0.40
    r_decay: float = 0.995
    k_open: float = 0.20
    fat_tail_p: float = 0.08
    fat_tail_alpha: float = 2.2
    fat_tail_scale: float = 1.75
    hysteresis_steps: int = 6
    hysteresis_mult: float = 1.20
    knowledge_diffusion_rate: float = 0.02
    talent_hardware_elasticity: float = 1.5
    debt_call_center: float = 2.5   
    debt_call_slope: float  = 2.0   
    debt_service_threshold: float = 0.20
    debt_service_sensitivity: float = 1.0

    fear_decay: float = 0.90
    fear_shock_misuse: float = 0.15
    political_capital_decay: float = 0.98

    w_choke_train: Dict[str, float] = dc_field(default_factory=lambda: {"EDA":0.25,"LITHO":0.40,"HBM":0.30,"CLOUD":0.05})
    w_choke_infer: Dict[str, float] = dc_field(default_factory=lambda: {"EDA":0.15,"LITHO":0.15,"HBM":0.30,"CLOUD":0.40})
    
    regime_bite: Dict[str, float] = dc_field(default_factory=lambda: {
        "OPEN":0.00, "CARVEOUT":0.20, "LICENSE_FEE":0.50, "BAN":0.95 
    })

@dataclass
class Exogenous:
    eta_series: Optional[any] = None

@dataclass
class Flags:
    terminal_on_escalation: bool = False
    enable_alliance_sharing: bool = True
    enable_dynamic_alliances: bool = False
    enable_exports: bool = True
    enable_governance_slowdown: bool = True
    enable_crisis_mode: bool = True
    enable_observation_asymmetry: bool = True
    use_belief_Gsec_in_hazards: bool = False
    enable_price_mechanism: bool = True

@dataclass
class Heterogeneity:
    beta_C: Dict[str, float] = dc_field(default_factory=dict)
    kappa_G: Dict[str, float] = dc_field(default_factory=dict)
    xi_S: Dict[str, float] = dc_field(default_factory=dict)
    gamma_X: Dict[str, float] = dc_field(default_factory=dict)
    K_D: Dict[str, float] = dc_field(default_factory=dict)
    K_E: Dict[str, float] = dc_field(default_factory=dict)
    E_norm: Dict[str, float] = dc_field(default_factory=dict)
    w_eta: Dict[str, float] = dc_field(default_factory=dict)
    w_C:   Dict[str, float] = dc_field(default_factory=dict)
    w_D:   Dict[str, float] = dc_field(default_factory=dict)
    w_E:   Dict[str, float] = dc_field(default_factory=dict)
    w_S:   Dict[str, float] = dc_field(default_factory=dict)
    omega_E: Dict[str, float] = dc_field(default_factory=dict)
    omega_S: Dict[str, float] = dc_field(default_factory=dict)
    omega_B: Dict[str, float] = dc_field(default_factory=dict)
    omega_R: Dict[str, float] = dc_field(default_factory=dict)
    openness_bias: Dict[str, float] = dc_field(default_factory=dict)
    export_sender_power: Dict[str, float] = dc_field(default_factory=dict)
    hazard_sensitivity: Dict[str, float] = dc_field(default_factory=dict)
    cost_Gup_mult: Dict[str, float] = dc_field(default_factory=dict)
    cost_Gdown_mult: Dict[str, float] = dc_field(default_factory=dict)
    panic_threshold: Dict[str, float] = dc_field(default_factory=dict)
    tech_strategy: Dict[str, str] = dc_field(default_factory=dict)
    tax_rate: Dict[str, float] = dc_field(default_factory=dict)
    insolvency_threshold: Dict[str, float] = dc_field(default_factory=dict)

@dataclass
class Config:
    actors: List[str]
    T: int = 40
    weights: Weights = dc_field(default_factory=Weights)
    rates: Rates = dc_field(default_factory=Rates)
    flags: Flags = dc_field(default_factory=Flags)
    het: Heterogeneity = dc_field(default_factory=Heterogeneity)
    epsilon: float = 1e-9
    seed: int = 42
    time_unit: str = "quarter"
    steps_per_year: int = 4
    C0: Dict[str, float] = dc_field(default_factory=dict)
    D0: Dict[str, float] = dc_field(default_factory=dict)
    E0: Dict[str, float] = dc_field(default_factory=dict)
    S0: Dict[str, float] = dc_field(default_factory=dict)
    B0: Dict[str, float] = dc_field(default_factory=dict)
    R0: Dict[str, float] = dc_field(default_factory=dict)
    G0: Dict[str, float] = dc_field(default_factory=dict)
    G0_sec: Dict[str, float] = dc_field(default_factory=dict)
    G0_eval: Dict[str, float] = dc_field(default_factory=dict)
    G0_deploy: Dict[str, float] = dc_field(default_factory=dict)
    K_D: float = 3.0
    K_E: float = 3.0
    A0_edges: List[Tuple[str,str]] = dc_field(default_factory=list)
    X0: Optional[pd.DataFrame] = None
    X_regimes: Dict[Tuple[str,str], Dict[str,str]] = dc_field(default_factory=dict)
    chokepoints: Tuple[str,...] = ("EDA","LITHO","HBM","CLOUD")
    exog: Exogenous = dc_field(default_factory=Exogenous)
    sigma_self: float = 0.02   
    sigma_ally: float = 0.10
    sigma_rival: float = 0.20 
    mu_bias_rival: float = 0.05
    obs_lag: int = 4
    sigma_self_map: Optional[Dict[str, float]] = None
    sigma_ally_map: Optional[Dict[str, float]] = None
    sigma_rival_map: Optional[Dict[str, float]] = None
    mu_bias_map: Optional[Dict[Tuple[str,str], float]] = None
    obs_lag_map: Optional[Dict[str, int]] = None
    belief_beta_map: Optional[Dict[str, float]] = None
    belief_beta: float = 0.50 
    cost_C: float = 0.45
    cost_D: float = 0.10
    cost_E: float = 0.35
    cost_Gup: float = 0.05
    cost_Gdown: float = 0.05
    events: List[dict] = dc_field(default_factory=list)
    gpaith_enabled: bool = True
    gpaith_cap_threshold: float = 3.0
    gpaith_delta_G: float = 0.10
    gpaith_sigma_scale: float = 0.85
    gpaith_cost_bump: float = 0.02
    gpaith_subjects: Optional[List[str]] = None