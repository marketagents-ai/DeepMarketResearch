# Market Oracle - Quick Start Guide

## 🚀 Run Project in 3 Simple Steps

### 1️⃣ Install Dependencies

```bash
# Install Backend Dependencies
cd MarketOracle/market_agents/research_agent_with_ui/backend
pip install -r requirements.txt

# Install Frontend Dependencies
cd ../frontend
npm install
```

### 2️⃣ Make Start Script Executable

```bash
# From project root directory
chmod +x start-app.sh
```

### 3️⃣ Run the Project

```bash
./start-app.sh
```

## 🌐 Access the Application
- 💻 Frontend: http://localhost:5173
- ⚙️ Backend: http://localhost:8000

## 🛑 Stop the Application
- Press `Ctrl + C` in the terminal to stop both services

## ❗ Common Issues
- If ports 5173 or 8000 are in use, stop other services using these ports
- If script doesn't run, ensure you've made it executable with `chmod +x start-app.sh`
- For dependency issues, try reinstalling with the commands in Step 1

## 📋 Features
- Real-time market analysis
- Multi-source data aggregation
- AI-powered insights
- CSV export functionality
- Interactive UI with expandable details

## 💡 Need Help?
Check terminal outputs for any error messages or contact the development team.