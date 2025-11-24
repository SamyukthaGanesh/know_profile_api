#!/bin/bash

# TrustBank AI Governance Platform - Complete Startup Script
# Starts all backend services and frontend in one command

echo "════════════════════════════════════════════════════════════════"
echo "   🚀 Starting TrustBank AI Governance Platform"
echo "════════════════════════════════════════════════════════════════"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/Users/samganesh/Downloads/know_profile_api"

# Kill any existing processes on our ports
echo -e "\n${BLUE}🧹 Cleaning up existing processes...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:8002 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:3001 | xargs kill -9 2>/dev/null
sleep 2

# Clear logs
echo -e "${BLUE}📝 Clearing old logs...${NC}"
> "$BASE_DIR/logs/trustbank.log"
> "$BASE_DIR/logs/ai_governance_db.log"
> "$BASE_DIR/logs/chatbot.log"
> "$BASE_DIR/logs/auth.log"
> "$BASE_DIR/logs/frontend.log"

# Start TrustBank Backend (Port 8000)
echo -e "\n${GREEN}1️⃣  Starting TrustBank Backend (Port 8000)...${NC}"
cd "$BASE_DIR"
source .venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > logs/trustbank.log 2>&1 &
echo "   ✅ TrustBank Backend started"

# Start AI Governance Framework (Port 8001)
echo -e "\n${GREEN}2️⃣  Starting AI Governance Framework - GHCI (Port 8001)...${NC}"
cd "$BASE_DIR/backend"
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    source "$BASE_DIR/.venv/bin/activate"
fi
nohup uvicorn api:app --host 0.0.0.0 --port 8001 --reload > "$BASE_DIR/logs/ai_governance_db.log" 2>&1 &
echo "   ✅ AI Governance Framework started"

# Start AI Chatbot (Port 8002)
echo -e "\n${GREEN}3️⃣  Starting AI Chatbot (Port 8002)...${NC}"
cd "$BASE_DIR/chatbot/routed_agent_gemini"
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
else
    source "$BASE_DIR/.venv/bin/activate"
fi
nohup uvicorn api:app --host 0.0.0.0 --port 8002 --reload > "$BASE_DIR/logs/chatbot.log" 2>&1 &
echo "   ✅ AI Chatbot started"

# Start Auth Server (Port 3001)
echo -e "\n${GREEN}4️⃣  Starting Auth Server (Port 3001)...${NC}"
cd "$BASE_DIR/trust-platform-ui"
nohup node auth-server-simple.js > "$BASE_DIR/logs/auth.log" 2>&1 &
echo "   ✅ Auth Server started"

# Start React Frontend (Port 3000)
echo -e "\n${GREEN}5️⃣  Starting React Frontend (Port 3000)...${NC}"
cd "$BASE_DIR/trust-platform-ui"
nohup npm start > "$BASE_DIR/logs/frontend.log" 2>&1 &
echo "   ✅ React Frontend starting..."

echo -e "\n${BLUE}⏳ Waiting for all services to initialize...${NC}"
sleep 10

echo -e "\n════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ ALL SERVICES STARTED SUCCESSFULLY!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Service Status:"
echo "   1️⃣  TrustBank Backend:        http://localhost:8000"
echo "   2️⃣  AI Governance (GHCI):     http://localhost:8001"
echo "   3️⃣  AI Chatbot:               http://localhost:8002"
echo "   4️⃣  Auth Server:              http://localhost:3001"
echo "   5️⃣  React Frontend:           http://localhost:3000"
echo ""
echo "📖 API Documentation:"
echo "   • TrustBank API Docs:         http://localhost:8000/docs"
echo "   • GHCI API Docs:              http://localhost:8001/docs"
echo "   • Chatbot API Docs:           http://localhost:8002/docs"
echo ""
echo "🌐 Access the Platform:"
echo "   • User Portal:                http://localhost:3000"
echo "   • Admin Portal:               http://localhost:3000/admin"
echo ""
echo "📝 Logs are available in: $BASE_DIR/logs/"
echo ""
echo "🛑 To stop all services, run:"
echo "   pkill -f 'uvicorn'; pkill -f 'node.*auth-server'; pkill -f 'react-scripts'"
echo ""
echo "════════════════════════════════════════════════════════════════"
