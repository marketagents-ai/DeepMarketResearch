export interface ResearchData {
  url: string;
  title: string;
  content: string;
  timestamp: string;
  status: string;
  summary: {
    summary: string;
    key_points: string[];
    market_impact: {
      short_term: string;
      medium_term: string;
      long_term: string;
    };
    trading_implications: {
      entry_points: string[];
      exit_targets: string[];
      stop_loss_levels: string[];
      position_sizing: string;
    };
    technical_analysis: {
      trend_direction: string;
      support_levels: string[];
      resistance_levels: string[];
      indicators: Record<string, string>;
    };
    sentiment_analysis: {
      overall_sentiment: string;
      sentiment_score: string;
      social_metrics: {
        social_volume: string;
        sentiment_trend: string;
      };
      market_confidence: string;
    };
    risk_assessment: {
      risk_level: string;
      risk_factors: string[];
      mitigation_strategies: string[];
      risk_reward_ratio: string;
    };
    price_analysis: {
      current_price: string;
      target_prices: {
        short_term: string[];
        medium_term: string[];
        long_term: string[];
      };
      price_drivers: string[];
      volatility_assessment: string;
    };
  };
  agent_id: string;
  extraction_method: string;
}

export const sendResearchRequest = async (query: string, customSchemas: any[]) => {
  console.log('Sending research request with:', {
    query,
    customSchemas
  });
  
  const response = await fetch('/api/research', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      custom_schemas: customSchemas
    })
  });