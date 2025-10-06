"""Common utility functions shared across the codebase."""

def is_running_in_container() -> bool:
    """Check if we're running inside a Docker container"""
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read().lower()
    except:
        return False