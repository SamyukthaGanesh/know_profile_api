#!/bin/bash

# Complete startup script with AI Chatbot integration
# This script starts all 5 services for the TrustBank Platform

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                   🚀 TRUSTBANK COMPLETE PLATFORM STARTUP"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting 5 services:"
echo "  1️⃣  TrustBank Backend (port 8000)"
echo "  2️⃣  AI Governance Framework (port 8001)"
echo "  3️⃣  AI Governance Chatbot (port 8002)"
echo "  4️⃣  Auth Server (port 3001)"
echo "  5️⃣  React Frontend (port 3000)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Create logs directory
mkdir -p logs

# Stop any existing processes on these ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:8002 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:3001 | xargs kill -9 2>/dev/null
sleep 2

# 1. Start TrustBank Backend
echo "1️⃣  Starting TrustBank Backend..."
cd /Users/samganesh/Downloads/know_profile_api
source .venv/bin/activate 2>/dev/null || true
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > logs/trustbank.log 2>&1 &
TRUSTBANK_PID=$!
echo "   ✓ TrustBank Backend started (PID: $TRUSTBANK_PID)"
echo "   📊 http://localhost:8000/docs"

# 2. Start AI Governance Framework (Database version)
echo "2️⃣  Starting AI Governance Framework..."
cd "/Users/samganesh/Downloads/ai_governance_framework 2"
nohup python -m uvicorn api.endpoints:app --host 0.0.0.0 --port 8001 --reload > /Users/samganesh/Downloads/know_profile_api/logs/ai_governance_db.log 2>&1 &
GHCI_PID=$!
echo "   ✓ AI Governance Framework started (PID: $GHCI_PID)"
echo "   🔒 http://localhost:8001/docs"

# 3. Start AI Governance Chatbot
echo "3️⃣  Starting AI Governance Chatbot..."
cd /Users/samganesh/Downloads/ai_governance_chatbot
nohup python -m uvicorn routed_agent_gemini.api:app --host 0.0.0.0 --port 8002 --reload > /Users/samganesh/Downloads/know_profile_api/logs/chatbot.log 2>&1 &
CHATBOT_PID=$!
echo "   ✓ AI Chatbot started (PID: $CHATBOT_PID)"
echo "   🤖 http://localhost:8002/docs"

# 4. Start Auth Server
echo "4️⃣  Starting Auth Server..."
cd /Users/samganesh/Downloads/know_profile_api/trust-platform-ui
nohup node auth-server-simple.js > /Users/samganesh/Downloads/know_profile_api/logs/auth.log 2>&1 &
AUTH_PID=$!
echo "   ✓ Auth Server started (PID: $AUTH_PID)"
echo "   🔐 http://localhost:3001"

# Wait for backends to be ready
echo ""
echo "⏳ Waiting for backends to initialize..."
sleep 5

# 5. Start React Frontend
echo "5️⃣  Starting React Frontend..."
cd /Users/samganesh/Downloads/know_profile_api/trust-platform-ui
nohup npm start > /Users/samganesh/Downloads/know_profile_api/logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   ✓ React Frontend started (PID: $FRONTEND_PID)"
echo "   🌐 http://localhost:3000"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                          ✅ ALL SERVICES STARTED!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📍 Access Points:"
echo "   Frontend:           http://localhost:3000"
echo "   TrustBank API:      http://localhost:8000/docs"
echo "   AI Governance API:  http://localhost:8001/docs"
echo "   AI Chatbot API:     http://localhost:8002/docs"
echo "   Auth API:           http://localhost:3001"
echo ""
echo "📝 Process IDs:"
echo "   TrustBank:          $TRUSTBANK_PID"
echo "   AI Governance:      $GHCI_PID"
echo "   AI Chatbot:         $CHATBOT_PID"
echo "   Auth Server:        $AUTH_PID"
echo "   Frontend:           $FRONTEND_PID"
echo ""
echo "📋 Logs:"
echo "   tail -f logs/trustbank.log"
echo "   tail -f logs/ai_governance_db.log"
echo "   tail -f logs/chatbot.log"
echo "   tail -f logs/auth.log"
echo "   tail -f logs/frontend.log"
echo ""
echo "🛑 To stop all services:"
echo "   kill $TRUSTBANK_PID $GHCI_PID $CHATBOT_PID $AUTH_PID $FRONTEND_PID"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                    🎉 Platform ready for demo!"
echo "═══════════════════════════════════════════════════════════════════════════════"

