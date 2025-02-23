# Setting Up ChatSkeleton Locally

## Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (v18 or higher)
- npm (v9 or higher)

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ChatSkeleton.git
   cd ChatSkeleton
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory:
   ```bash
   VITE_API_URL=http://localhost:5000
   ```

4. **Start Development Server**
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`

## Project Structure

```
ChatSkeleton/
├── src/
│   ├── components/     # React components
│   ├── hooks/         # Custom React hooks
│   ├── services/      # API services
│   ├── types/         # TypeScript type definitions
│   └── utils/         # Utility functions
├── public/           # Static assets
└── package.json      # Project dependencies
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Development Guidelines

1. **Code Style**
   - Follow TypeScript best practices
   - Use functional components with hooks
   - Maintain consistent file naming (PascalCase for components)

2. **Component Structure**
   - Keep components small and focused
   - Use TypeScript interfaces for props
   - Implement proper error handling

3. **State Management**
   - Use React hooks for local state
   - Implement proper data persistence
   - Handle loading and error states

## Features Configuration

### Chat Modes
- Research Mode: Enable URL analysis and research queries
- Custom Mode: Configure file upload settings

### System Messages
Create custom system messages in the System tab:
1. Click the "+" button
2. Fill in the name and content
3. Select the mode (Research/Custom)
4. Save and activate as needed

### Tools
The tools system is extensible through:
- `src/components/ToolsPanel/tools.config.ts`
- Custom tool builder in the UI

## Troubleshooting

Common issues and solutions:

1. **Build Errors**
   ```bash
   npm clean-install
   ```

2. **Development Server Issues**
   ```bash
   # Clear Vite cache
   rm -rf node_modules/.vite
   ```

3. **Type Errors**
   ```bash
   # Regenerate TypeScript types
   npm run typecheck
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

## Support

For issues and questions:
1. Check the GitHub issues
2. Review the documentation
3. Open a new issue if needed