import React from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { ResearchData } from '../types/research';

interface ResearchCardProps {
  data: ResearchData;
}

export const ResearchCard: React.FC<ResearchCardProps> = ({ data }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-4">
      <div 
        className="flex justify-between items-center cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2 className="text-xl font-semibold">{data.title.split('#').pop() || 'Research Result'}</h2>
        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-6">
          <div>
            <h3 className="text-lg font-medium mb-2">Summary</h3>
            <p className="text-gray-700">{data.summary.summary}</p>
            
            <div className="mt-4">
              <h4 className="font-medium mb-2">Key Points</h4>
              <ul className="list-disc pl-5">
                {data.summary.key_points.map((point, index) => (
                  <li key={index} className="text-gray-700">{point}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-3">Market Impact</h3>
              <div className="space-y-2">
                <p><span className="font-medium">Short Term:</span> {data.summary.market_impact.short_term}</p>
                <p><span className="font-medium">Medium Term:</span> {data.summary.market_impact.medium_term}</p>
                <p><span className="font-medium">Long Term:</span> {data.summary.market_impact.long_term}</p>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-3">Price Analysis</h3>
              <p><span className="font-medium">Current Price:</span> {data.summary.price_analysis.current_price}</p>
              <p><span className="font-medium">Volatility:</span> {data.summary.price_analysis.volatility_assessment}</p>
              <div className="mt-2">
                <h4 className="font-medium">Price Drivers</h4>
                <ul className="list-disc pl-5">
                  {data.summary.price_analysis.price_drivers.map((driver, index) => (
                    <li key={index} className="text-gray-700">{driver}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-3">Technical Analysis</h3>
              <p><span className="font-medium">Trend:</span> {data.summary.technical_analysis.trend_direction}</p>
              <div className="mt-2">
                <h4 className="font-medium">Indicators</h4>
                <p><span className="font-medium">RSI:</span> {data.summary.technical_analysis.indicators.rsi}</p>
                <p><span className="font-medium">MACD:</span> {data.summary.technical_analysis.indicators.macd}</p>
                <p><span className="font-medium">Moving Averages:</span> {data.summary.technical_analysis.indicators.moving_averages}</p>
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-3">Risk Assessment</h3>
              <p><span className="font-medium">Risk Level:</span> {data.summary.risk_assessment.risk_level}</p>
              <p><span className="font-medium">Risk/Reward Ratio:</span> {data.summary.risk_assessment.risk_reward_ratio}</p>
              <div className="mt-2">
                <h4 className="font-medium">Risk Factors</h4>
                <ul className="list-disc pl-5">
                  {data.summary.risk_assessment.risk_factors.map((factor, index) => (
                    <li key={index} className="text-gray-700">{factor}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="text-sm text-gray-500 mt-4">
            <p>Source: <a href={data.url} className="text-blue-500 hover:underline" target="_blank" rel="noopener noreferrer">{data.url}</a></p>
            <p>Last Updated: {new Date(data.timestamp).toLocaleString()}</p>
          </div>
        </div>
      )}
    </div>
  );
};