import React, { useState, useEffect, useRef } from 'react';
import { useResearch } from '../context/ResearchContext';
import { MatrixRain } from './MatrixRain';  

interface AIFinancialLoaderProps {
  logs?: string[];
}

export const AIFinancialLoader: React.FC<AIFinancialLoaderProps> = ({ logs: propLogs = [] }) => {
  const [frame, setFrame] = useState(0);
  const [marketData, setMarketData] = useState([50, 52, 54, 53, 56, 58, 57, 60]);
  const [startTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState('0s');
  const { logs: contextLogs = [] } = useResearch();
  const logContainerRef = useRef<HTMLDivElement>(null);
  const logs = propLogs.length > 0 ? propLogs : contextLogs;

  const generateMarketData = (prevMarketData: number[]) => {
    const lastValue = prevMarketData[prevMarketData.length - 1];
    const change = Math.random() > 0.5 ? 1 : -1;
    const newValue = Math.max(0, Math.min(100, lastValue + change * Math.random() * 5));
    return [...prevMarketData.slice(1), newValue];
  };

  const getMarketDataDisplay = () => marketData.map(v => v > 55 ? '▲' : '▼').join('');
  const getMarketChartDisplay = () => marketData.map(v => String.fromCharCode(9600 + Math.floor(v/12.5))).join('');
  const getMarketTrendDisplay = () => marketData.map(v => v > 55 ? '↗' : v < 45 ? '↘' : '→').join('');
  const getMarketEmojiDisplay = () => marketData.map(v => v > 55 ? '📈' : '📉').join('');

  const states = [
    {
      text: 'Initializing quantum processors...',
      ascii: `
    ┌──────── QUANTUM AI MARKETS V2.0 ────────┐
    │     🔮 QUANTUM INITIALIZATION 🔮        │
    │                                         │
    │        ┌─╴╶─┐  ┌─╴╶─┐  ┌─╴╶─┐         │
    │     ╺━━┥▓▓▓▓├━━┥▓▓▓▓├━━┥▓▓▓▓├━━╸      │
    │        └─╴╶─┘  └─╴╶─┘  └─╴╶─┘         │
    │     QUANTUM CORES: ▓▓▓▓▓▓░░ 75%        │
    │     ENTANGLEMENT: ▓▓▓▓▓▓▓░ 85%         │
    │     COHERENCE:    ▓▓▓▓▓░░░ 62%         │
    │                                         │
    │     [${getMarketDataDisplay()}]      │
    │     INITIALIZING QUANTUM STATES...     │
    │     Time elapsed: ${elapsedTime}           │
    └─────────────────────────────────────────┘`
    },
    {
      text: 'Processing market neural patterns...',
      ascii: `
    ┌──────── QUANTUM AI MARKETS V2.0 ────────┐
    │      🧠 NEURAL NETWORK ANALYSIS 🧠      │
    │                                         │
    │           ▄▄▄▄▄     ▄▄▄▄▄              │
    │          ▀●   ●▀   ▀●   ●▀             │
    │     ▄▄▄▄▄ ║   ║ ▄▄▄ ║   ║ ▄▄▄▄▄       │
    │    ▀●   ●▀║   ║▀●   ●▀  ║▀●   ●▀      │
    │     ║   ║ ║   ║ ║   ║   ║ ║   ║       │
    │     NEURONS ACTIVE: ${Math.floor(Math.random() * 20 + 80)}%             │
    │     SYNAPTIC STRENGTH: ${Math.floor(Math.random() * 15 + 85)}%          │
    │     [${getMarketChartDisplay()}]     │
    │     PROCESSING MARKET PATTERNS...       │
    │     Time elapsed: ${elapsedTime}           │
    └─────────────────────────────────────────┘`
    },
    {
      text: 'Analyzing market volatility...',
      ascii: `
    ┌──────── QUANTUM AI MARKETS V2.0 ────────┐
    │        📊 MARKET VOLATILITY 📊          │
    │                                         │
    │     ▗▄▄▄  ▄▖  ▄▄▄▖   ▗▄   ▄▄▄▖        │
    │    ▗█▀▀█▌ ▐▌ █▀▀▀▘  ▗█▀   █▀▀▘        │
    │    ▐▌  ▐▌ ▐▌ ▐▌     ▐▌    ▐▌          │
    │    VOLATILITY INDEX: ${Math.floor(Math.random() * 20 + 20)}             │
    │    MARKET SENTIMENT: ${Math.random() > 0.5 ? 'BULLISH' : 'BEARISH'}        │
    │    TRADE VOLUME: ${Math.floor(Math.random() * 1000 + 5000)}M           │
    │                                         │
    │    [${getMarketTrendDisplay()}]     │
    │    ANALYZING MARKET DYNAMICS...        │
    │    Time elapsed: ${elapsedTime}           │
    └─────────────────────────────────────────┘`
    },
    {
      text: 'Generating trade signals...',
      ascii: `
    ┌──────── QUANTUM AI MARKETS V2.0 ────────┐
    │         💹 TRADE SIGNAL ENGINE 💹       │
    │                                         │
    │    SIGNAL STRENGTH: ▓▓▓▓▓▓░░ ${Math.floor(Math.random() * 20 + 80)}%     │
    │    CONFIDENCE: ${Math.floor(Math.random() * 10 + 90)}%                    │
    │    POSITION: ${Math.random() > 0.5 ? 'LONG  ↗' : 'SHORT ↘'}              │
    │    ╔══════════════════════════╗        │
    │    ║ ${getMarketChartDisplay()} ║        │
    │    ╚══════════════════════════╝        │
    │    RISK RATIO: ${(Math.random() * 2 + 1).toFixed(2)}                     │
    │    PROFIT TARGET: ${Math.floor(Math.random() * 20 + 10)}%                │
    │    GENERATING SIGNALS...               │
    │    Time elapsed: ${elapsedTime}           │
    └─────────────────────────────────────────┘`
    }
  ];
  useEffect(() => {
    const frameTimer = setInterval(() => {
      setFrame(prev => (prev + 1) % states.length);
    }, 3000);

    const dataTimer = setInterval(() => {
      setMarketData(prevData => generateMarketData(prevData));
    }, 500);

    const timeTimer = setInterval(() => {
      setElapsedTime(formatElapsedTime(Date.now() - startTime));
    }, 1000);

    return () => {
      clearInterval(frameTimer);
      clearInterval(dataTimer);
      clearInterval(timeTimer);
    };
  }, []);

  const formatElapsedTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);



  return (
    <div className="w-full h-full flex flex-col items-center justify-center relative">
      {/* Add MatrixRain as background */}
      <div className="absolute inset-0 opacity-30">
        <MatrixRain />
      </div>

      {/* Keep existing content with increased z-index */}
      <div className="w-full flex flex-col items-center justify-center relative z-20">
        <div className="flex flex-col items-center space-y-4 max-w-2xl mx-auto w-full">
          {/* Quantum Display */}
          <div className="font-mono text-[#63e2ff] bg-[#1a1b1e]/90 p-4 rounded-lg shadow-lg w-full">
            <pre className="whitespace-pre overflow-x-auto">
              {states[frame].ascii}
            </pre>
          </div>

          {/* System Logs */}
          <div className="w-full bg-[#1a1b1e]/90 rounded-lg shadow-lg">
            <div className="text-[#63e2ff] font-medium text-sm p-2 border-b border-[#63e2ff]/20">
              System Logs:
            </div>
            <div 
              ref={logContainerRef}
              className="max-h-[150px] overflow-y-auto p-2"
            >
              {logs.map((log, index) => (
                <pre 
                  key={index} 
                  className="text-[#63e2ff] text-sm font-mono whitespace-pre-wrap break-words mb-1"
                >
                  {log}
                </pre>
              ))}
            </div>
          </div>

          {/* Loading Spinner */}
          <div className="flex flex-col items-center">
            <div className="w-12 h-12 border-4 border-[#63e2ff] border-t-transparent rounded-full animate-spin mb-2"></div>
            <div className="text-[#63e2ff] font-medium text-sm">
              {states[frame].text}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};