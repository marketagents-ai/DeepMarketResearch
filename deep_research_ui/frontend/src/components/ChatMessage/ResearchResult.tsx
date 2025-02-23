import React, { useState } from 'react';
import type { ResearchData } from '../../types/research';
import { Download } from 'lucide-react';

interface ResearchResultProps {
  data: ResearchData[];
}

export const ResearchResult: React.FC<ResearchResultProps> = ({ data }) => {
  const [expandedItems, setExpandedItems] = useState<Record<number, boolean>>({});
  const [isTableExpanded, setIsTableExpanded] = useState(true);

  if (!Array.isArray(data)) {
    return null;
  }

  const toggleExpand = (index: number) => {
    setExpandedItems(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const convertToCSV = (analysisData: any[]) => {
    // Get all field names including custom ones
    const allFields = Array.from(new Set(
      analysisData.flatMap(item => Object.keys(item))
    ));

    const rows = analysisData.map(item => 
      allFields.map(field => item[field] || '')
    );

    return [allFields, ...rows]
      .map(row => row.map(cell => 
        typeof cell === 'object' ? JSON.stringify(cell) : `"${cell}"`
      ).join(','))
      .join('\n');
  };

  const downloadCSV = (analysisData: any[]) => {
    const csv = convertToCSV(analysisData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', `market_analysis_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const renderSummaryTable = () => {
    const analysisData = data.map(item => {
      try {
        const summary = typeof item.summary === 'string' ? 
          JSON.parse(item.summary) : item.summary;
          
        if (summary.assets && summary.assets.length > 0) {
          const asset = summary.assets[0];
          
          // Extract custom fields
          const customFields = asset.custom_fields || {};
          
          return {
            ticker: asset.ticker,
            rating: asset.rating,
            target_price: asset.target_price,
            sentiment: asset.sentiment,
            action: asset.action,
            catalysts: asset.catalysts,
            kpis: asset.kpis,
            sources: asset.sources,
            ...customFields  // Include custom fields
          };
        }
        return null;
      } catch (e) {
        console.error('Error parsing summary:', e);
        return null;
      }
    }).filter(Boolean);

    if (analysisData.length === 0) return null;

    // Get all fields including custom ones
    const allFields = Array.from(new Set(
      analysisData.flatMap(item => Object.keys(item))
    ));

    return (
      <div className="bg-gray-800 rounded-lg p-4 mb-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium text-white">Market Analysis Summary</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setIsTableExpanded(!isTableExpanded)}
              className="text-gray-400 hover:text-white"
            >
              {isTableExpanded ? 'Collapse' : 'Expand'}
            </button>
            <button
              onClick={() => downloadCSV(analysisData)}
              className="flex items-center gap-1 text-blue-400 hover:text-blue-300"
            >
              <Download size={16} />
              Export CSV
            </button>
          </div>
        </div>

        {isTableExpanded && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-300">
              <thead className="text-xs uppercase bg-gray-700">
                <tr>
                  {allFields.map(field => (
                    <th key={field} className="px-4 py-3">
                      {field.replace(/_/g, ' ').toUpperCase()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {analysisData.map((item, index) => (
                  <tr key={index} className="border-b border-gray-700 bg-gray-800/50">
                    {allFields.map(field => (
                      <td key={field} className="px-4 py-3">
                        {renderTableCell(item[field])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  const renderTableCell = (value: any) => {
    if (Array.isArray(value)) {
      return (
        <ul className="list-disc list-inside">
          {value.map((item, i) => (
            <li key={i} className="truncate">{item}</li>
          ))}
        </ul>
      );
    }
    
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value);
    }

    return value;
  };

  // Main component return
  return (
    <div className="space-y-4 relative z-0"> 
      {renderSummaryTable()}
      
      {/* Source Analysis Details */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h4 className="text-lg font-medium text-white mb-4">Source Analysis Details</h4>
        <div className="space-y-3">
          {data.map((result, index) => (
            <div key={index} className="bg-gray-700/30 rounded-lg p-4">
              <div 
                className="cursor-pointer"
                onClick={() => toggleExpand(index)}
              >
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-medium text-white">
                    {result.title || new URL(result.url).hostname}
                  </h3>
                  <span>{expandedItems[index] ? '▼' : '▶'}</span>
                </div>
                
                <div className="text-sm text-gray-400 mt-2">
                  <a 
                    href={result.url} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="hover:text-blue-400"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {result.url}
                  </a>
                </div>
              </div>

              {expandedItems[index] && (
                <div className="mt-4 space-y-4">
                  <div className="bg-gray-700/50 p-4 rounded">
                    <h4 className="font-medium mb-2">Analysis Details</h4>
                    <pre className="text-sm text-gray-300 whitespace-pre-wrap">
                      {JSON.stringify(result.summary, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};