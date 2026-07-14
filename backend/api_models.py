# FinShield-AI API Models & Pydantic Schemas v1.0.0
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TransactionSimulationRequest(BaseModel):
    transaction_type: str = Field(..., example="TRANSFER")
    amount: float = Field(..., example=250000.0)
    old_balance_orig: float = Field(..., example=250000.0)
    new_balance_orig: float = Field(..., example=0.0)
    old_balance_dest: float = Field(..., example=12000.0)
    new_balance_dest: float = Field(..., example=12000.0)
    tx_velocity_1h: int = Field(1, example=3)
    is_mule_victim: int = Field(0, example=1)
    is_mule_hop_receiver: int = Field(0, example=0)

class SHAPContribution(BaseModel):
    feature: str
    impact: float
    value: Any

class SimulationResponse(BaseModel):
    risk_score: float
    risk_level: str  # "HIGH RISK (FLAGGED)", "MODERATE RISK", "LEGITIMATE"
    decision_threshold: float
    shap_contributions: List[SHAPContribution]
    computed_features: Dict[str, Any]
