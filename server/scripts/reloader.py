#!/usr/bin/env python3
"""
CodeAgent System Reloader
Monitors source code files for changes and automatically restarts the system.
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class CodeChangeHandler(FileSystemEventHandler):
    """Handles file system events for code changes"""

    def __init__(self, reloader):
        self.reloader = reloader
        self.last_reload = time.time()
        self.reload_cooldown = 2.0  # seconds

    def should_reload(self, event):
        """Check if we should trigger a reload"""
        # Skip if too soon since last reload
        if time.time() - self.last_reload < self.reload_cooldown:
            return False

        # Only monitor Python files
        if not event.src_path.endswith('.py'):
            return False

        # Skip certain directories
        skip_dirs = [
            'codeagent_venv',
            '__pycache__',
            '.pytest_cache',
            '.ruff_cache',
            'node_modules',
            '.git'
        ]

        for skip_dir in skip_dirs:
            if skip_dir in event.src_path:
                return False

        return True

    def on_modified(self, event):
        if self.should_reload(event):
            logger.info(f"Code change detected: {event.src_path}")
            self.reloader.trigger_reload()

    def on_created(self, event):
        if self.should_reload(event):
            logger.info(f"New file created: {event.src_path}")
            self.reloader.trigger_reload()

    def on_deleted(self, event):
        if self.should_reload(event):
            logger.info(f"File deleted: {event.src_path}")
            self.reloader.trigger_reload()

class SystemReloader:
    """Manages the reloading of the entire CodeAgent system"""

    def __init__(self):
        self.observer = None
        self.current_process = None
        self.watch_paths = [
            repo_root / 'orchestrator',
            repo_root / 'agents',
            repo_root / 'sandbox_executor',
            repo_root / 'tool_api_gateway',
            repo_root / 'comparator_service',
            repo_root / 'vector_store',
            repo_root / 'prompt_store',
            repo_root / 'transcript_store',
            repo_root / 'observability',
            repo_root / 'policy_engine',
            repo_root / 'providers',
            repo_root / 'scripts'
        ]

    def start_monitoring(self):
        """Start monitoring for file changes"""
        logger.info("Starting file monitoring for hot reloading...")

        event_handler = CodeChangeHandler(self)
        self.observer = Observer()

        for watch_path in self.watch_paths:
            if watch_path.exists():
                logger.info(f"Watching: {watch_path}")
                self.observer.schedule(event_handler, str(watch_path), recursive=True)
            else:
                logger.warning(f"Watch path does not exist: {watch_path}")

        self.observer.start()
        logger.info("File monitoring started successfully")

    def stop_monitoring(self):
        """Stop monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File monitoring stopped")

    def trigger_reload(self):
        """Trigger a system reload"""
        logger.info("Triggering system reload...")

        # Stop current system
        self.stop_system()

        # Wait a moment
        time.sleep(1)

        # Start system again
        self.start_system()

    def start_system(self):
        """Start the CodeAgent system"""
        logger.info("Starting CodeAgent system...")

        try:
            # Use the start_local.sh script
            start_script = repo_root / 'start_local.sh'
            if start_script.exists():
                logger.info("Using start_local.sh script")
                self.current_process = subprocess.Popen(
                    [str(start_script)],
                    cwd=str(repo_root),
                    preexec_fn=os.setsid  # Create new process group
                )
            else:
                logger.error("start_local.sh not found")
                return

            logger.info(f"System started with PID: {self.current_process.pid}")

        except Exception as e:
            logger.error(f"Failed to start system: {e}")

    def stop_system(self):
        """Stop the current system"""
        if self.current_process:
            logger.info("Stopping current system...")

            try:
                # Try graceful shutdown first
                if repo_root / 'stop_local.sh' in [f for f in repo_root.iterdir() if f.is_file()]:
                    stop_script = repo_root / 'stop_local.sh'
                    subprocess.run([str(stop_script)], cwd=str(repo_root), timeout=10)
                else:
                    # Kill the process group
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                    self.current_process.wait(timeout=10)

            except subprocess.TimeoutExpired:
                logger.warning("Graceful shutdown timed out, force killing...")
                try:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already dead

            except Exception as e:
                logger.error(f"Error stopping system: {e}")

            self.current_process = None
            logger.info("System stopped")

    def run(self):
        """Main run loop"""
        logger.info("CodeAgent System Reloader starting...")

        # Start monitoring
        self.start_monitoring()

        # Start initial system
        self.start_system()

        try:
            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        finally:
            # Cleanup
            self.stop_monitoring()
            self.stop_system()
            logger.info("Reloader shutdown complete")

def main():
    """Main entry point"""
    reloader = SystemReloader()
    reloader.run()

if __name__ == "__main__":
    main()