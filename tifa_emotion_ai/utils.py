"""
TIFA Emotion AI - Shared Utilities
==================================
Common utility functions used across modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Global console for rich output
console = Console()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging with rich formatting.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("tifa_emotion_ai")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "tifa_emotion_ai") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


def print_header(title: str, subtitle: str = ""):
    """Print styled header"""
    console.print()
    console.print("=" * 60, style="bold blue")
    console.print(f"  {title}", style="bold white")
    if subtitle:
        console.print(f"  {subtitle}", style="dim")
    console.print("=" * 60, style="bold blue")
    console.print()


def print_status(icon: str, label: str, value: str, style: str = ""):
    """Print status line with icon"""
    console.print(f"[{icon}] {label}: {value}", style=style)


def print_emotion(emotion: str, confidence: float):
    """Print emotion with emoji and color"""
    emoji_map = {
        "neutral": "😐",
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fear": "😨",
        "surprise": "😲",
        "disgust": "🤢"
    }
    
    color_map = {
        "neutral": "white",
        "happy": "yellow",
        "sad": "blue",
        "angry": "red",
        "fear": "magenta",
        "surprise": "cyan",
        "disgust": "green"
    }
    
    emoji = emoji_map.get(emotion.lower(), "❓")
    color = color_map.get(emotion.lower(), "white")
    
    console.print(
        f"{emoji} Emosi: [bold {color}]{emotion.upper()}[/] "
        f"(confidence: {confidence:.1%})"
    )


def timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
