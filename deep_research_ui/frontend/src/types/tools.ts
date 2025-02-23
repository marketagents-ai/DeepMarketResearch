import type { LucideIcon } from 'lucide-react';

export interface SystemTool {
  id: string;
  name: string;
  description: string;
  icon: LucideIcon;
  type: 'executable' | 'typed';
  category: 'analysis' | 'risk' | 'data' | 'ai' | 'automation';
}
export interface SystemTool {
  id: string;
  name: string;
  description: string;
  icon: LucideIcon;
  type: 'executable' | 'typed';
  // category: string;
  enabled?: boolean;
}
export interface SchemaField {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array';
  required?: boolean;
}


export interface Tool {
  id: string;
  name: string;
  description: string;
  schema: any;
  enabled: boolean;
}
export interface CustomTool {
  id: string;
  name: string;
  description: string;
  schema: Record<string, any>;
  enabled: boolean;
}