"""
Centralized logging configuration for the RAG system.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOG_FILE, LOG_LEVEL

# Create logs directory if needed
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Formatter
fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

# File handler with rotation
fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
fh.setFormatter(fmt)
logger.addHandler(fh)

# Console handler for INFO+
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(ch)
