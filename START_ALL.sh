#!/bin/bash

# TrustBank AI Platform - Start All Services
# This script starts both the FastAPI backend and React frontend

echo "🚀 Starting TrustBank AI Platform..."
echo ""

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "uvicorn" 2>/dev/null
pkill -f "react-scripts" 2>/dev/null
sleep 2

# Start FastAPI Backend
echo "⚡ Starting FastAPI Backend (Port 8000)..."
cd /Users/samganesh/Downloads/know_profile_api
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

sleep 3

# Check if backend started
if lsof -i :8000 2>/dev/null | grep -q LISTEN; then
    echo "✅ Backend is running (PID: $BACKEND_PID)"
else
    echo "❌ Backend failed to start. Check /tmp/backend.log"
    exit 1
fi

# Start React Frontend
echo "🎨 Starting React Frontend (Port 3000)..."
cd /Users/samganesh/Downloads/know_profile_api/trust-platform-ui
NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ NPM_CONFIG_CACHE=/Users/samganesh/Downloads/know_profile_api/trust-platform-ui/.npm-cache npm start > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

echo "⏳ Waiting for React to compile (30 seconds)..."
sleep 30

# Check if frontend started
if lsof -i :3000 2>/dev/null | grep -q LISTEN; then
    echo "✅ Frontend is running (PID: $FRONTEND_PID)"
else
    echo "⏳ Frontend is still compiling... check /tmp/frontend.log"
fi

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  🎉 TrustBank AI Platform is LIVE!                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "⚡ Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "📱 Try these pages:"
echo "  • Your Profile:    http://localhost:3000/user/profile"
echo "  • Dashboard:       http://localhost:3000/user/dashboard"
echo "  • AI Explanations: http://localhost:3000/user/explanations"
echo "  • Consent Wallet:  http://localhost:3000/user/consent"
echo "  • Admin Dashboard: http://localhost:3000/admin/overview"
echo ""
echo "📋 Logs:"
echo "  • Backend:  tail -f /tmp/backend.log"
echo "  • Frontend: tail -f /tmp/frontend.log"
echo ""
echo "🛑 To stop all services:"
echo "  pkill -f uvicorn && pkill -f react-scripts"
echo ""

