from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, create_model

class SearchQueries(BaseModel):
    """Schema for search query generation"""
    queries: List[str] = Field(
        description="List of search queries generated from the base query",
        examples=[
            ["latest cryptocurrency trends December 2024", 
             "top performing crypto assets Q4 2024"]
        ]
    )

class AnalysisType(str, Enum):
    ASSET = "asset"
    SECTOR = "sector"
    MACRO = "macro"
    GENERAL = "general"

class AssetAnalysis(BaseModel):
    ticker: str = Field(..., description="Asset ticker (e.g., stock symbol or crypto token).")
    rating: Optional[str] = Field(None, description="Analyst rating (e.g., Buy/Hold/Sell) and brief rationale.")
    target_price: Optional[str] = Field(None, description="Target price or price range from sources.")
    sentiment: Optional[str] = Field(None, description="Overall sentiment (Bullish, Bearish, Neutral).")
    catalysts: List[str] = Field(default_factory=list, description="Key qualitative catalysts for the asset.")
    kpis: List[str] = Field(default_factory=list, description="Key performance indicators (quantitative) for the asset.")
    action: Optional[str] = Field(None, description="Recommended portfolio action (e.g., Add Long, Close Short).")
    sources: List[str] = Field(default_factory=list, description="Information sources (e.g., brokers, analysts).")
    custom_fields: Dict[str, str] = Field(default_factory=dict, description="Custom analysis fields")

class SectorInfo(BaseModel):
    name: str = Field(..., description="Name of the sector or industry.")
    sentiment: Optional[str] = Field(None, description="Overall sentiment for the sector.")
    catalysts: List[str] = Field(default_factory=list, description="Sector-wide catalysts.")
    kpis: List[str] = Field(default_factory=list, description="Sector-level KPIs or metrics.")
    top_assets: List[str] = Field(default_factory=list, description="Representative assets with brief notes.")
    recommendation: Optional[str] = Field(None, description="Recommended sector exposure (e.g., Overweight/Underweight).")
    sources: List[str] = Field(default_factory=list, description="Key sector-level sources.")

class MacroTrends(BaseModel):
    indicators: List[str] = Field(default_factory=list, description="Macro indicators (e.g., GDP, inflation, interest rates).")
    interest_rates: Optional[str] = Field(None, description="Interest rate outlook.")
    global_factors: Optional[str] = Field(None, description="Global geopolitical or economic factors.")
    sentiment: Optional[str] = Field(None, description="Overall macro-level sentiment (risk-on/off).")

class MarketResearch(BaseModel):
    analysis_type: AnalysisType = Field(..., description="Type of analysis: asset, sector, macro, or general.")
    assets: List[AssetAnalysis] = Field(default_factory=list, description="Asset-level insights if 'asset' type.")
    sector: Optional[SectorInfo] = Field(None, description="Sector-level details if 'sector' type.")
    macro: Optional[MacroTrends] = Field(None, description="Macro-level insights if 'macro' type.")
