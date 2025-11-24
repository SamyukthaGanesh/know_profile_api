#!/usr/bin/env python3
"""
Database Setup and Migration Script
Sets up SQLite database for AI Governance Framework
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def install_dependencies():
    """Install required dependencies"""
    import subprocess
    
    dependencies = [
        'sqlalchemy==1.4.48',
        'alembic==1.12.1'
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True


def setup_database():
    """Initialize database with tables and sample data"""
    try:
        from core.database.models import init_database, reset_database
        from core.database.services import ModelService, HealthService
        from core.database.models import get_db_session
        from datetime import datetime
        
        print("🔄 Setting up database...")
        
        # Reset database (for development)
        reset_database()
        
        # Initialize with sample data
        with next(get_db_session()) as db:
            # Register a sample model
            ModelService.register_model(
                model_id="home_credit_v1",
                model_name="Home Credit Default Predictor",
                model_version="1.0",
                model_type="classification",
                feature_names=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_INCOME_TOTAL"],
                db=db
            )
            
            # Record initial health status
            services = ['framework', 'fairness', 'explainability', 'compliance']
            for service in services:
                HealthService.record_health_check(
                    service_name=service,
                    status="healthy",
                    db=db
                )
        
        print("✅ Database setup completed successfully")
        print(f"📁 Database location: {os.path.abspath('data/ai_governance.db')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 AI Governance Framework Database Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Setup database
    if not setup_database():
        print("❌ Failed to setup database")
        return False
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the API server: uvicorn api.endpoints:app --reload")
    print("2. Run tests: python home_credit_api_test.py")
    print("3. Check dashboard: http://localhost:8000/dashboard/overview")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)