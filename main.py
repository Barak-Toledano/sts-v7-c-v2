#!/usr/bin/env python3
"""
OpenAI Realtime Assistant - Main Entry Point

This module provides the command-line interface for the OpenAI Realtime Assistant.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from src.config.logging_config import get_logger
from src.application import Application
from src.domain.audio.manager import AudioMode
from src.system_instructions import APPOINTMENT_SCHEDULER, APPOINTMENT_TOOLS

# Initialize logger
logger = get_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="OpenAI Realtime Assistant")
    
    parser.add_argument(
        "--assistant-id",
        type=str,
        default=os.environ.get("OPENAI_ASSISTANT_ID", ""),
        help="ID of the OpenAI assistant to use"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Set logging level"
    )
    
    parser.add_argument(
        "--audio-mode",
        choices=["conversation", "dictation", "playback_only", "off"],
        default="conversation",
        help="Audio processing mode"
    )
    
    parser.add_argument(
        "--select-devices",
        action="store_true",
        help="Prompt for audio device selection"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Process an audio file instead of running in interactive mode"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()

def map_audio_mode(mode_str: str) -> AudioMode:
    """Map string audio mode to enum value."""
    mode_map = {
        "conversation": AudioMode.CONVERSATION,
        "dictation": AudioMode.DICTATION,
        "playback_only": AudioMode.PLAYBACK_ONLY,
        "off": AudioMode.OFF
    }
    return mode_map.get(mode_str.lower(), AudioMode.CONVERSATION)

async def process_audio_file(file_path: str, app: Application) -> int:
    """
    Process an audio file and get a response.
    
    Args:
        file_path: Path to the audio file to process
        app: Initialized application instance
        
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        logger.info(f"Processing audio file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 1
            
        # Start the application
        if not await app.start():
            logger.error("Failed to start application")
            return 1
            
        # TODO: Implement file processing
        logger.info(f"File processing completed: {file_path}")
        
        # Clean shutdown
        await app.stop()
        return 0
        
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return 1

async def run_async_main(args: argparse.Namespace) -> int:
    """
    Run the main application logic asynchronously.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize application
        app = Application(
            assistant_id=args.assistant_id,
            instructions=APPOINTMENT_SCHEDULER,
            tools=APPOINTMENT_TOOLS,
            audio_mode=map_audio_mode(args.audio_mode),
            select_devices=args.select_devices,
            debug_mode=args.debug
        )
        
        # Handle file processing or interactive mode
        if args.file:
            return await process_audio_file(args.file, app)
        else:
            await app.run()
            return 0
            
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run the async main function
    try:
        return asyncio.run(run_async_main(args))
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())