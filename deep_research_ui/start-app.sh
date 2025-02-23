#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}Starting Market Oracle Services...${NC}\n"

# Function to handle script termination
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $(jobs -p)
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Start Backend
echo -e "${GREEN}Starting Backend Service...${NC}"
cd backend && \
uvicorn app:app --reload &

# Start Frontend
echo -e "${GREEN}Starting Frontend Service...${NC}"
cd frontend && \
npm run dev &

# Wait for both processes
wait