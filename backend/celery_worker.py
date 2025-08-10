#!/usr/bin/env python3
"""
Celery worker entry point for the Citation Network Explorer.

This script starts a Celery worker that processes background tasks.

Usage:
    python celery_worker.py

Or with specific options:
    celery -A celery_worker worker --loglevel=info --concurrency=4 --queues=papers,zotero
"""

import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.tasks import get_celery_app
from app.utils.logger import setup_logging

# Setup logging
setup_logging()

# Get the Celery app instance
celery_app = get_celery_app()

if __name__ == '__main__':
    # Start worker programmatically
    celery_app.start(['worker', '--loglevel=info', '--concurrency=4'])