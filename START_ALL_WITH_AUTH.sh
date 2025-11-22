#!/bin/bash

# TrustBank AI Platform - Complete Startup Script
# This script starts all 3 services needed for the platform

echo "🏦 Starting TrustBank AI Platform..."
echo ""
echo "This will start:"
echo "  1. FastAPI Backend (Port 8000)"
echo "  2. Auth Server (Port 3001)"
echo "  3. React Frontend (Port 3000)"
echo ""

# Check if we're in the right directory
if [ ! -d "trust-platform-ui" ]; then
    echo "❌ Error: trust-platform-ui directory not found!"
    echo "Please run this script from the know_profile_api directory"
    exit 1
fi

# Start FastAPI Backend
echo "📡 Starting FastAPI Backend..."
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000 > backend.log 2>&1 &
FASTAPI_PID=$!
echo "✅ FastAPI started (PID: $FASTAPI_PID) - Logs: backend.log"

# Wait a moment
sleep 2

# Start Auth Server
echo "🔐 Starting Auth Server..."
cd trust-platform-ui
node auth-server-simple.js > auth-server.log 2>&1 &
AUTH_PID=$!
echo "✅ Auth Server started (PID: $AUTH_PID) - Logs: trust-platform-ui/auth-server.log"

# Wait a moment
sleep 2

# Start React Frontend
echo "⚛️  Starting React Frontend..."
NPM_CONFIG_REGISTRY=https://registry.npmjs.org/ NPM_CONFIG_CACHE=.npm-cache npm start > react-app.log 2>&1 &
REACT_PID=$!
echo "✅ React App starting (PID: $REACT_PID) - Logs: trust-platform-ui/react-app.log"

echo ""
echo "═══════════════════════════════════════════════════════"
echo ""
echo "🎉 All services are starting!"
echo ""
echo "Access your application at:"
echo "  🌐 Landing Page:   http://localhost:3000"
echo "  👤 User Dashboard: http://localhost:3000/user/dashboard"
echo "  👔 Admin Panel:    http://localhost:3000/admin/overview"
echo "  📡 FastAPI Backend: http://localhost:8000"
echo "  🔐 Auth Server:    http://localhost:3001"
echo ""
echo "Default Login Credentials:"
echo "  Admin: userId='admin', password='password'"
echo "  User:  userId='user1', password='password'"
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Process IDs (for stopping):"
echo "  FastAPI: $FASTAPI_PID"
echo "  Auth Server: $AUTH_PID"
echo "  React App: $REACT_PID"
echo ""
echo "To stop all services, run:"
echo "  kill $FASTAPI_PID $AUTH_PID $REACT_PID"
echo ""
echo "Logs are being written to:"
echo "  backend.log"
echo "  trust-platform-ui/auth-server.log"
echo "  trust-platform-ui/react-app.log"
echo ""
echo "Wait ~20 seconds for React to compile..."
echo ""

