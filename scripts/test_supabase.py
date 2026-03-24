#!/usr/bin/env python
"""
Test script to verify Supabase integration for FPF analytics.

This script tests the connection and basic operations with Supabase.
Run this after setting up the database schema.

Usage:
    python scripts/test_supabase.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import fpf_modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fpf_modules.supabase_manager import get_supabase_client, initialize_schema, read_table
    from supabase import create_client
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed supabase: pip install supabase")
    sys.exit(1)


def test_connection():
    """Test basic connection to Supabase."""
    print("🔗 Testing Supabase connection...")
    try:
        client = get_supabase_client()
        print("✅ Connected to Supabase successfully!")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None


def test_tables_exist(client):
    """Test if all required tables exist."""
    print("\n📋 Checking if tables exist...")
    required_tables = [
        'athletes', 'sessions', 'games', 'metrics',
        'performance_metrics', 'quality_metrics', 'samples', 'athlete_session'
    ]

    existing_tables = []
    for table in required_tables:
        try:
            # Try to select from table (will fail if table doesn't exist)
            response = client.table(table).select("*").limit(1).execute()
            existing_tables.append(table)
            print(f"✅ {table} - exists")
        except Exception as e:
            print(f"❌ {table} - missing or error: {str(e)[:100]}...")

    if len(existing_tables) == len(required_tables):
        print(f"\n✅ All {len(required_tables)} tables exist!")
        return True
    else:
        print(f"\n⚠️  Only {len(existing_tables)}/{len(required_tables)} tables found")
        return False


def test_basic_operations(client):
    """Test basic read/write operations."""
    print("\n🔄 Testing basic operations...")

    # Test 1: Insert a test athlete
    try:
        test_athlete = {
            'atleta_id': 'test_athlete_001',
            'genero': 'M',
            'ativo': True
        }

        response = client.table('athletes').insert(test_athlete).execute()
        print("✅ Insert athlete - success")

        # Get the athlete_sk for cleanup
        athlete_sk = response.data[0]['athlete_sk']

        # Test 2: Read the athlete back
        response = client.table('athletes').select('*').eq('atleta_id', 'test_athlete_001').execute()
        if response.data:
            print("✅ Read athlete - success")
        else:
            print("❌ Read athlete - failed")

        # Test 3: Insert a test session
        test_session = {
            'session_fingerprint': 'test_session_fingerprint_001',
            'started_at': '2024-01-01T10:00:00Z',
            'device': 'test_device'
        }

        response = client.table('sessions').insert(test_session).execute()
        print("✅ Insert session - success")

        session_sk = response.data[0]['session_sk']

        # Test 4: Insert performance metrics
        test_metrics = {
            'session_sk': session_sk,
            'athlete_sk': athlete_sk,
            'atleta_id': 'test_athlete_001',
            'phase_id': 1,
            'fase': '1P',
            'data': '2024-01-01',
            'selecao': 'U23 M',
            'genero': 'M',
            'contexto': 'Jogo',
            'jogo': 'Test Game',
            'duracao_min': 90.0,
            'dist_m': 10000.0,
            'm_min': 111.1,
            'vmax_mps': 8.5,
            'peak_1m_m_min': 300.0,
            'hsr_dist_m': 1500.0,
            'hsr_pct': 15.0,
            'sprint_dist_m': 500.0,
            'n_sprints': 10,
            'n_acc_2_5': 25,
            'n_dec_3_0': 15,
            'active_time_min': 85.0,
            'active_pct': 94.4
        }

        response = client.table('performance_metrics').insert(test_metrics).execute()
        print("✅ Insert performance metrics - success")

        # Test 5: Read performance metrics
        response = client.table('performance_metrics').select('*').eq('session_sk', session_sk).execute()
        if response.data:
            print("✅ Read performance metrics - success")
        else:
            print("❌ Read performance metrics - failed")

        # Cleanup: Delete test data
        print("\n🧹 Cleaning up test data...")
        client.table('performance_metrics').delete().eq('session_sk', session_sk).execute()
        client.table('sessions').delete().eq('session_sk', session_sk).execute()
        client.table('athletes').delete().eq('athlete_sk', athlete_sk).execute()
        print("✅ Test data cleaned up")

        return True

    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Supabase integration for FPF Analytics\n")

    # Test connection
    client = test_connection()
    if not client:
        sys.exit(1)

    # Test tables
    if not test_tables_exist(client):
        print("\n❌ Some tables are missing. Please run the setup script again.")
        sys.exit(1)

    # Test operations
    if test_basic_operations(client):
        print("\n🎉 All tests passed! Supabase integration is working correctly.")
        print("\n🚀 Your app is ready to use Supabase for data storage!")
        return True
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
