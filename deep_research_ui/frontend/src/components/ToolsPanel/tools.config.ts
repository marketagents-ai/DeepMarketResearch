import { BarChart3, Shield, Database, Brain, Bot } from 'lucide-react';
import type { SystemTool } from '../../types/tools';

export const AVAILABLE_TOOLS: SystemTool[] = [
  {
    id: 'market-analysis',
    name: 'Market Analysis',
    description: 'Technical and fundamental market analysis tools',
    icon: BarChart3,
    type: 'executable',
    category: 'analysis'
  },
  {
    id: 'risk-management',
    name: 'Risk Management',
    description: 'Risk assessment and portfolio management',
    icon: Shield,
    type: 'executable',
    category: 'risk'
  },
  {
    id: 'data-feeds',
    name: 'Data Feeds',
    description: 'Real-time market data and news feeds',
    icon: Database,
    type: 'typed',
    category: 'data'
  },
  {
    id: 'ai-assistant',
    name: 'AI Assistant',
    description: 'Advanced AI-powered research assistant',
    icon: Brain,
    type: 'typed',
    category: 'ai'
  },
  {
    id: 'trading-bot',
    name: 'Trading Bot',
    description: 'Automated trading system integration',
    icon: Bot,
    type: 'executable',
    category: 'automation'
  }
];