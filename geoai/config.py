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
    # Findings: Hardware (30-37%), R&D Staff (30-37%), Energy (2-6%).
    
    w_eta: float = 0.20 # Total Factor Productivity (Exogenous Tech Progress)
    w_C:   float = 0.40 # Compute Capital (Hardware).
    w_D:   float = 0.35 # Talent & Data. Reflects parity with hardware costs.
    w_E:   float = 0.05 # Energy Flow. Matches direct training energy cost share (~2-6%).
    w_S:   float = 0.00 # Security/Stability (Institutional Friction).
    
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
    w_pol: float = 50.0 # Political Capital acts as "Hard Currency" for alliances/sanctions.
    w_hoard: float = 1.0 # Penalty for hoarding excess cash (B > 50% GDP).
    
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
    # Global real GDP growth ~3.2% annually (IMF 2025) -> ~0.8% quarterly.
    gC_base: float = 0.008  # Baseline economic growth (non-AI).
    gD_base: float = 0.02   # Talent pool grows faster due to education/training lags.
    gE_base: float = 0.01   # Energy infrastructure baseline growth.
    
    # Depreciation
    # Source: CITP (2025). AI chips have ~1.5-3 years useful economic life.
    # 0.12 quarterly ~= 40% annualized. Reflects rapid obsolescence (H100 -> Blackwell).
    depreciation_C: float = 0.12 
    
    # Source: Bloom et al. (2020) "Are Ideas Getting Harder to Find?".
    # Ideas depreciate/commoditize over time.
    knowledge_depreciation: float = 0.06 
    
    # Delays
    # Source: JLL 2025 Global Data Center Outlook.
    # Power transmission delays can extend timelines to 4+ years.
    # 4 years = 16 quarters. Updated from original 6.
    energy_build_lag: int = 16 
    
    # Investment Efficiency (Marginal Returns)
    beta_C: float = 0.12
    beta_D: float = 0.08
    beta_E: float = 0.04
    
    # --- 2. Asymmetric Conflict Pricing ---
    # Relative costs normalized to GDP scale (Action Costs)
    # Source: Epoch AI (2024). Hardware and Staff costs are roughly 1:1 for frontier models.
    cost_C: float = 0.40 # High capital intensity.
    cost_D: float = 0.40 # High talent intensity.
    cost_E: float = 0.20 # Infrastructure investment.
    
    # Security Costs
    # Source: Center for Data Innovation (2021). EU AI Act compliance costs ~17% of AI investment.
    cost_Gup: float = 0.05 
    cost_Gdown: float = 0.05
    cost_sabotage: float = 0.15 
    
    # Source: FBI/USTR estimates. IP theft costs US ~1-2% of GDP.
    cost_theft: float = 0.02    
    
    # --- 3. Maintenance & Efficiency ---
    # Source: Encor Advisors (2025). OpEx is ~40% of TCO annually.
    maintenance_rate_C: float = 0.12 
    maintenance_rate_K: float = 0.05
    
    # Tech Progress
    # Source: Epoch AI (2024). Algorithmic efficiency doubles every 8-12 months.
    # 1.20^4 ~= 2.07 (Doubles every year).
    efficiency_growth_rate: float = 1.20 
    
    # Source: Bloom et al. (2020). Diminishing returns to research.
    knowledge_production_exponent: float = 0.60 
    
    # Governance Friction
    # Source: EU AI Act. Compliance adds ~17-25% overhead.
    kappa_G: float = 0.25  
    
    ban_maintenance_penalty: float = 0.20 
    ban_efficiency_decay: float = 0.02 
    
    # Macro-Financial Links
    gdp_growth_base: float = 0.005 
    # Source: Goldman Sachs (2023). AI boosts GDP ~0.7% annually.
    gdp_ai_multiplier: float = 0.002
    governance_gdp_bonus: float = 0.008 
    deployment_friction: float = 0.20
    
    # "Alignment Tax"
    # Source: Elicit/OpenReview (2024). Safety incurs 15-30% overhead.
    safety_compute_drag: float = 0.25
    
    cost_alliance_formation: float = 0.05 
    maintenance_rate_alliance: float = 0.01 
    
    # Source: PLOS (2024). Allied spillovers are significant.
    alliance_innovation_rate: float = 0.08 
    alliance_talent_bonus: float = 0.10
    
    # Source: OECD (2024). Global avg CIT rate ~21%.
    tax_rate: float = 0.21 
    
    # --- 4. Financials (Sovereign Debt) ---
    yield_curve_min_rate: float = 0.01   
    yield_curve_max_rate: float = 0.05   
    yield_curve_midpoint: float = 1.60   
    yield_curve_steepness: float = 6.0   
    max_debt_ratio: float = 1.10 
    debt_hard_limit: float = 6.0 
    debt_interest_rate: float = 0.02 
    insolvency_threshold_center: float = 2.5 
    insolvency_threshold_k: float = 10.0
    insolvency_duration_years: float = 6.0
    insolvency_shock_C: float = 0.40
    insolvency_shock_GDP: float = 0.40

    # --- 5. Talent & Demographics ---
    talent_growth_base: float = 0.01
    talent_agglomeration_rate: float = 0.02 
    energy_efficiency: float = 1.32
    ban_cost_mult: float = 1.5
    spy_efficiency: float = 0.3      
    counter_spy_efficiency: float = 0.4 
    prob_sabotage_success: float = 0.30
    prob_theft_success: float = 0.40
    
    # Source: GIS Reports (2025). High-skill migration ~1-5% annual.
    migration_fraction: float = 0.01
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
    
    # Economic Impact of Shocks
    # Source: TechUK/KPMG (2025). Systemic cyber incidents cost ~2.8% of weekly GDP.
    # Scaled to 0.03 (3% of quarterly GDP) for major AI misuse events.
    gdp_hit_misuse: float = 0.03  
    gdp_hit_esc: float = 0.25     
    
    # Governance Efficacy
    # Source: Relyance AI (2025). Governance reduces risk but doesn't eliminate "Fat Tail" events.
    # theta=0.55 implies governance reduces risk by ~45% (1-0.55).
    theta_G_mis: float = 0.55
    theta_G_esc: float = 0.45
    
    # Race Intensity Impact
    # Source: Axify (2025). Software defect rates rise with development speed (Agile/DevOps trade-offs).
    theta_R: float = 0.20
    theta_R2: float = 0.40
    theta_Scarcity: float = 0.30
    theta_BeliefErr: float = 0.25
    pmax: float = 0.20
    gamma_X: float = 0.35
    
    # --- 9. Leakage & Spillovers ---
    s_regrow: float = 0.02
    
    # Knowledge Leakage
    # Source: Centre for Economic Performance (2013). International spillovers account for 37% of returns.
    # lambda_leak = 0.10 (10% per quarter) matches the high diffusion rates of digital tech.
    lambda_leak: float = 0.10 
    h_cap: float = 0.70         
    phi_subst: float = 0.02    
    subst_norm: float = 4.0    
    
    # Substitution Elasticity (Hardware vs Talent)
    # Source: NBER (2024). AI and high-skill labor are complements in production, not perfect substitutes.
    # 0.20 allows for some substitution but maintains bottlenecks.
    phi_subst_train: float = 0.20
    phi_subst_infer: float = 0.30
    
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

    # Political/Social Decay
    # Source: Westbourne Partners (2025). Stock prices recover from data breaches in 46-90 days.
    # 0.60 per quarter implies a fast recovery (shorter attention span).
    fear_decay: float = 0.60
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