export interface SystemMessage {
  id: string;
  name: string;
  content: string;
  mode: 'custom' | 'research';
  createdAt: Date;
}

export interface SystemState {
  messages: SystemMessage[];
  activeMessageId: string | null;
}