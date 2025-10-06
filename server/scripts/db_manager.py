#!/usr/bin/env python3
"""
Database Management Script for CodeAgent
Handles SQLite database operations, lock cleanup, and maintenance.
"""

import os
import sys
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

class DatabaseManager:
    def __init__(self, db_path="codeagent.db"):
        self.db_path = Path(db_path)
        self.backup_dir = Path("db_backups")
        self.backup_dir.mkdir(exist_ok=True)

    def check_connection(self):
        """Check if database is accessible"""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("SELECT 1")
            conn.close()
            return True, "Database is accessible"
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                return False, "Database is locked"
            else:
                return False, f"Database error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def unlock_database(self):
        """Attempt to unlock database by killing processes and cleaning up"""
        import subprocess

        print("🔍 Checking for processes holding database locks...")

        # Find processes that might be holding the database
        try:
            result = subprocess.run(
                ["lsof", str(self.db_path)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Header + at least one process
                    print("📋 Processes with database open:")
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 2:
                            pid = parts[1]
                            cmd = ' '.join(parts[8:]) if len(parts) > 8 else 'unknown'
                            print(f"  PID {pid}: {cmd}")

                            # Ask user if they want to kill the process
                            try:
                                response = input(f"Kill process {pid}? (y/N): ").strip().lower()
                                if response == 'y':
                                    subprocess.run(["kill", "-9", pid], check=True)
                                    print(f"✅ Killed process {pid}")
                            except KeyboardInterrupt:
                                print("\nOperation cancelled")
                                return False
                            except subprocess.CalledProcessError:
                                print(f"❌ Failed to kill process {pid}")
        except subprocess.TimeoutExpired:
            print("⚠️  lsof command timed out")
        except FileNotFoundError:
            print("⚠️  lsof command not found, cannot check processes")

        # Try to vacuum the database to clean up
        print("🧹 Attempting database cleanup...")
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.execute("VACUUM")
            conn.commit()
            conn.close()
            print("✅ Database vacuum completed")
        except Exception as e:
            print(f"❌ Database vacuum failed: {e}")

        # Check if unlock was successful
        accessible, message = self.check_connection()
        if accessible:
            print("✅ Database unlocked successfully")
            return True
        else:
            print(f"❌ Database still locked: {message}")
            return False

    def backup_database(self, suffix=None):
        """Create a backup of the database"""
        if not self.db_path.exists():
            print("❌ Database file does not exist")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            backup_name = f"{self.db_path.stem}_{suffix}_{timestamp}{self.db_path.suffix}"
        else:
            backup_name = f"{self.db_path.stem}_{timestamp}{self.db_path.suffix}"

        backup_path = self.backup_dir / backup_name

        try:
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ Database backed up to: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def reset_database(self):
        """Reset the database by removing it and recreating"""
        if not self.db_path.exists():
            print("❌ Database file does not exist")
            return False

        # Create backup first
        if not self.backup_database("before_reset"):
            print("⚠️  Continuing with reset despite backup failure")

        try:
            self.db_path.unlink()
            print("✅ Database file removed")

            # Initialize new database
            print("🔄 Initializing new database...")
            from orchestrator.database import init_db
            import asyncio

            async def init():
                await init_db()

            asyncio.run(init())
            print("✅ New database initialized")
            return True

        except Exception as e:
            print(f"❌ Database reset failed: {e}")
            return False

    def show_info(self):
        """Show database information"""
        if not self.db_path.exists():
            print("❌ Database file does not exist")
            return

        # File info
        stat = self.db_path.stat()
        size_mb = stat.st_size / (1024 * 1024)

        print(f"📊 Database Information:")
        print(f"  Path: {self.db_path.absolute()}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Modified: {datetime.fromtimestamp(stat.st_mtime)}")

        # Connection check
        accessible, message = self.check_connection()
        status = "✅ Accessible" if accessible else f"❌ {message}"
        print(f"  Status: {status}")

        if accessible:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Get table info
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()

                print(f"  Tables: {len(tables)}")
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"    - {table_name}: {count} rows")

                conn.close()

            except Exception as e:
                print(f"  Query error: {e}")

    def list_backups(self):
        """List available database backups"""
        backups = list(self.backup_dir.glob(f"{self.db_path.stem}_*.db"))
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if not backups:
            print("📁 No backups found")
            return

        print("📁 Database Backups:")
        for backup in backups:
            stat = backup.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  {backup.name} ({size_mb:.2f} MB) - {mtime}")

    def restore_backup(self, backup_name):
        """Restore database from backup"""
        backup_path = self.backup_dir / backup_name

        if not backup_path.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return False

        # Backup current database first
        if self.db_path.exists():
            self.backup_database("before_restore")

        try:
            shutil.copy2(backup_path, self.db_path)
            print(f"✅ Database restored from: {backup_path}")

            # Verify restoration
            accessible, message = self.check_connection()
            if accessible:
                print("✅ Restored database is accessible")
                return True
            else:
                print(f"❌ Restored database has issues: {message}")
                return False

        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="CodeAgent Database Manager")
    parser.add_argument("action", choices=[
        "info", "unlock", "backup", "reset", "list-backups", "restore"
    ], help="Action to perform")
    parser.add_argument("--backup-name", help="Backup name for restore action")
    parser.add_argument("--db-path", default="codeagent.db", help="Database file path")

    args = parser.parse_args()

    manager = DatabaseManager(args.db_path)

    if args.action == "info":
        manager.show_info()

    elif args.action == "unlock":
        success = manager.unlock_database()
        if success:
            print("🎉 Database unlock completed successfully")
        else:
            print("💥 Database unlock failed")
            sys.exit(1)

    elif args.action == "backup":
        success = manager.backup_database()
        if not success:
            sys.exit(1)

    elif args.action == "reset":
        print("⚠️  This will delete all data in the database!")
        response = input("Are you sure? Type 'YES' to confirm: ")
        if response == "YES":
            success = manager.reset_database()
            if success:
                print("🎉 Database reset completed successfully")
            else:
                print("💥 Database reset failed")
                sys.exit(1)
        else:
            print("Operation cancelled")

    elif args.action == "list-backups":
        manager.list_backups()

    elif args.action == "restore":
        if not args.backup_name:
            print("❌ --backup-name is required for restore action")
            sys.exit(1)

        success = manager.restore_backup(args.backup_name)
        if success:
            print("🎉 Database restore completed successfully")
        else:
            print("💥 Database restore failed")
            sys.exit(1)


if __name__ == "__main__":
    main()