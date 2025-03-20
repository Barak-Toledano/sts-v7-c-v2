"""
Main application for the OpenAI Realtime Assistant.

This module brings together all components of the application architecture,
coordinating services, domain logic, and user interaction.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable

from src.config import settings
from src.config.logging_config import get_logger
from src.domain.audio.manager import AudioManager, AudioMode
from src.domain.conversation.manager import ConversationManager, ConversationState
from src.events.event_interface import Event, EventType, event_bus
from src.services.api_client import RealtimeClient
from src.services.audio_service import AudioService
from src.presentation.cli import CliInterface, print_cli_header
from src.utils.async_helpers import TaskManager, wait_for_event
from src.utils.error_handling import AppError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class Application:
    """
    Main application class for the OpenAI Realtime Assistant.
    
    This class brings together all components of the application and 
    provides high-level functionality for managing the assistant.
    """
    
    def __init__(
        self,
        assistant_id: str,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
        audio_mode: AudioMode = AudioMode.CONVERSATION,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        select_devices: bool = False,
        save_recordings: bool = False,
        debug_mode: Optional[bool] = None,
    ):
        """
        Initialize the application.
        
        Args:
            assistant_id: ID of the OpenAI assistant to use
            instructions: Optional custom instructions for the assistant
            tools: Optional tools configuration for the assistant
            temperature: Temperature parameter for generation (0.0-2.0)
            audio_mode: Mode for audio processing
            input_device: Optional audio input device index
            output_device: Optional audio output device index
            select_devices: Whether to prompt for device selection
            save_recordings: Whether to save audio recordings to disk
            debug_mode: Whether to enable debug mode
        """
        # Set debug mode if provided
        if debug_mode is not None:
            settings.debug_mode = debug_mode
        
        # Configure logging based on debug mode
        log_level = "DEBUG" if settings.debug_mode else settings.logging.level
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Store initialization parameters
        self.assistant_id = assistant_id
        self.instructions = instructions
        self.tools = tools
        self.temperature = temperature
        self.audio_mode = audio_mode
        self.select_devices = select_devices
        
        # Select audio devices if requested
        if select_devices:
            input_device, output_device = self._select_audio_devices()
        
        # Initialize API client
        self.api_client = RealtimeClient()
        
        # Initialize audio service
        self.audio_service = AudioService(
            input_device_index=input_device,
            output_device_index=output_device
        )
        
        # Initialize audio manager
        self.audio_manager = AudioManager(
            input_device=input_device,
            output_device=output_device,
            mode=audio_mode,
            save_recordings=save_recordings
        )
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            assistant_id=assistant_id,
            instructions=instructions,
            temperature=temperature,
            input_device=input_device,
            output_device=output_device
        )
        
        # Initialize CLI interface
        self.cli = CliInterface(
            show_timestamps=True,
            compact_mode=False,
            color_output=True,
            show_status=True
        )
        
        # Task management
        self.task_manager = TaskManager("application")
        self.shutdown_event = asyncio.Event()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        logger.info("Application initialized")
    
    async def start(self) -> bool:
        """
        Start the application.
        
        This method starts the conversation session and initializes
        all required components.
        
        Returns:
            bool: True if the application started successfully
        """
        logger.info("Starting application")
        
        try:
            # Connect to the API
            if not await self.api_client.connect(
                assistant_id=self.assistant_id,
                session_id=None,  # Generate new session
                instructions=self.instructions,
                temperature=self.temperature,
                enable_transcription=True
            ):
                logger.error("Failed to connect to OpenAI Realtime API")
                return False
            
            # Start the conversation manager
            if not await self.conversation_manager.start():
                logger.error("Failed to start conversation manager")
                return False
            
            # Start audio recording if enabled
            if self.audio_mode != AudioMode.OFF and self.audio_mode != AudioMode.PLAYBACK_ONLY:
                if not await self.audio_manager.start_recording():
                    logger.error("Failed to start audio recording")
                    return False
            
            # Configure tools if provided
            if self.tools:
                session_config = {"tools": self.tools}
                if not await self.api_client.update_session(session_config):
                    logger.warning("Failed to configure tools, continuing anyway")
            
            logger.info("Application started successfully")
            return True
            
        except Exception as e:
            error = AppError(
                f"Failed to start application: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                cause=e
            )
            logger.error(str(error))
            return False
    
    async def stop(self) -> None:
        """Stop the application and clean up resources."""
        logger.info("Stopping application")
        
        try:
            # Stop the conversation manager
            await self.conversation_manager.stop()
            
            # Clean up audio manager
            await self.audio_manager.cleanup()
            
            # Clean up audio service
            await self.audio_service.cleanup()
            
            # Disconnect from API
            await self.api_client.disconnect()
            
            # Cancel all tasks
            await self.task_manager.cancel_all()
            
            logger.info("Application stopped")
            
        except Exception as e:
            error = AppError(
                f"Error during application shutdown: {str(e)}",
                severity=ErrorSeverity.ERROR,
                cause=e
            )
            logger.error(str(error))
    
    async def run(self) -> None:
        """
        Run the application until shutdown is requested.
        
        This method starts the application and waits for a shutdown signal.
        """
        print_cli_header()
        
        success = await self.start()
        
        if not success:
            logger.error("Application failed to start")
            return
        
        # Display welcome message
        self.cli.display_welcome_message()
        
        # Start CLI input loop
        cli_task = self.task_manager.create_task(
            self.cli.input_loop(self._get_command_handlers()),
            "cli_input_loop"
        )
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Clean up
        await self.stop()
    
    def shutdown(self) -> None:
        """Request application shutdown."""
        logger.info("Shutdown requested")
        self.shutdown_event.set()
    
    async def request_response(
        self, 
        instructions: Optional[str] = None,
        modalities: Optional[List[str]] = None
    ) -> bool:
        """
        Request a response from the assistant.
        
        Args:
            instructions: Optional instructions for this response
            modalities: Optional response modalities
            
        Returns:
            bool: True if the request was successful
        """
        return await self.conversation_manager.request_response(instructions)
    
    async def interrupt(self) -> bool:
        """
        Interrupt the assistant's current response.
        
        Returns:
            bool: True if the interrupt was successful
        """
        return await self.conversation_manager.interrupt()
    
    def _get_command_handlers(self) -> Dict[str, Callable]:
        """
        Get handlers for CLI commands.
        
        Returns:
            Dict of command handlers
        """
        return {
            "quit": lambda _: self.shutdown(),
            "exit": lambda _: self.shutdown(),
            "restart": lambda _: self.task_manager.create_task(self._restart(), "restart"),
            "interrupt": lambda _: self.task_manager.create_task(self.interrupt(), "interrupt"),
            "status": lambda _: logger.info(f"State: {self.conversation_manager.state.name}"),
            "text_input": lambda text: self.task_manager.create_task(
                self.conversation_manager.send_text_message(text), 
                "process_text_input"
            )
        }
    
    async def _restart(self) -> None:
        """Restart the application."""
        logger.info("Restarting application")
        
        # Stop the current session
        await self.stop()
        
        # Wait a moment to ensure clean shutdown
        await asyncio.sleep(1)
        
        # Start a new session
        await self.start()
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for the application."""
        # Error handling
        event_bus.on(EventType.ERROR, self._handle_error)
        
        # Register for shutdown events
        event_bus.on(EventType.SHUTDOWN, self._handle_shutdown_event)
        
        # Register for conversation state changes
        event_bus.on(EventType.CONVERSATION_STATE_CHANGED, self._handle_conversation_state)
    
    def _handle_error(self, event: Event) -> None:
        """
        Handle error events.
        
        Args:
            event: Error event
        """
        error_data = event.data.get("error", {})
        error_type = error_data.get("type", "unknown")
        error_message = error_data.get("message", "Unknown error")
        
        # For critical errors, initiate shutdown
        if error_type in ("authentication_error", "authorization_error"):
            logger.critical(f"Critical error: {error_type} - {error_message}")
            self.task_manager.create_task(self._delayed_shutdown(), "shutdown_after_error")
    
    async def _delayed_shutdown(self, delay: float = 2.0) -> None:
        """
        Shutdown the application after a delay.
        
        Args:
            delay: Delay in seconds before shutdown
        """
        logger.info(f"Application will shut down in {delay} seconds due to critical error")
        await asyncio.sleep(delay)
        self.shutdown()
    
    def _handle_shutdown_event(self, event: Event) -> None:
        """
        Handle shutdown events from the application.
        
        Args:
            event: The shutdown event with reason details
        """
        reason = event.data.get("reason", "unknown")
        command = event.data.get("command", "")
        transcript = event.data.get("transcript", "")
        
        logger.info(f"Processing shutdown event. Reason: {reason}, Command: {command}")
        if transcript:
            logger.info(f"Shutdown triggered by transcript: '{transcript}'")
        
        # Trigger application shutdown
        self.shutdown()
    
    def _handle_conversation_state(self, event: Event) -> None:
        """
        Handle conversation state change events.
        
        Args:
            event: State change event
        """
        state = event.data.get("state")
        if state:
            logger.debug(f"Conversation state changed: {state}")
    
    def _select_audio_devices(self) -> tuple:
        """
        Prompt user to select audio devices.
        
        Returns:
            Tuple of (input_device_index, output_device_index)
        """
        # Use audio service to list devices
        devices = self.audio_service.list_audio_devices()
        
        # Show available devices
        print("\nAvailable Audio Devices:")
        print("------------------------")
        
        input_devices = []
        output_devices = []
        
        for device in devices:
            if device.get("maxInputChannels", 0) > 0:
                input_devices.append(device)
                print(f"Input  {device['index']}: {device['name']}")
            
            if device.get("maxOutputChannels", 0) > 0:
                output_devices.append(device)
                print(f"Output {device['index']}: {device['name']}")
        
        # Prompt for input device
        input_device_index = None
        if input_devices:
            try:
                choice = input("\nSelect input device # (press Enter for default): ")
                if choice.strip():
                    input_choice = int(choice)
                    if any(d["index"] == input_choice for d in input_devices):
                        input_device_index = input_choice
                    else:
                        print("Invalid selection, using default input device.")
            except ValueError:
                print("Invalid input, using default input device.")
        
        # Prompt for output device
        output_device_index = None
        if output_devices:
            try:
                choice = input("Select output device # (press Enter for default): ")
                if choice.strip():
                    output_choice = int(choice)
                    if any(d["index"] == output_choice for d in output_devices):
                        output_device_index = output_choice
                    else:
                        print("Invalid selection, using default output device.")
            except ValueError:
                print("Invalid input, using default output device.")
        
        return input_device_index, output_device_index
    
    def _register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        # Only register signal handlers if running in the main thread
        if not self._is_main_thread():
            return
        
        # Register SIGINT and SIGTERM handlers
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
                logger.debug(f"Registered signal handler for {sig}")
            except NotImplementedError:
                # Windows does not support add_signal_handler
                if sys.platform == 'win32':
                    signal.signal(sig, lambda s, f: self._signal_handler())
                    logger.debug(f"Registered Windows signal handler for {sig}")
    
    def _signal_handler(self) -> None:
        """Handle OS signals for graceful shutdown."""
        logger.info("Received shutdown signal")
        self.shutdown()
    
    def _is_main_thread(self) -> bool:
        """Check if current thread is the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()


async def run_application(
    assistant_id: str,
    instructions: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 1.0,
    audio_mode: AudioMode = AudioMode.CONVERSATION,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    select_devices: bool = False,
    save_recordings: bool = False,
    debug_mode: Optional[bool] = None,
) -> None:
    """
    Run the application with the given parameters.
    
    Args:
        assistant_id: ID of the OpenAI assistant to use
        instructions: Optional custom instructions for the assistant
        tools: Optional tools configuration for the assistant
        temperature: Temperature parameter for generation (0.0-2.0)
        audio_mode: Mode for audio processing
        input_device: Optional audio input device index
        output_device: Optional audio output device index
        select_devices: Whether to prompt for device selection
        save_recordings: Whether to save audio recordings to disk
        debug_mode: Whether to enable debug mode
    """
    app = Application(
        assistant_id=assistant_id,
        instructions=instructions,
        tools=tools,
        temperature=temperature,
        audio_mode=audio_mode,
        input_device=input_device,
        output_device=output_device,
        select_devices=select_devices,
        save_recordings=save_recordings,
        debug_mode=debug_mode
    )
    
    await app.run()


def main() -> int:
    """Entry point for running the application from the command line."""
    import argparse
    from dotenv import load_dotenv
    from src.system_instructions import APPOINTMENT_SCHEDULER, APPOINTMENT_TOOLS
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="OpenAI Realtime Assistant")
    
    parser.add_argument(
        "--assistant-id",
        type=str,
        default=os.environ.get("OPENAI_ASSISTANT_ID", ""),
        help="ID of the OpenAI assistant to use"
    )
    
    parser.add_argument(
        "--instructions",
        type=str,
        default=APPOINTMENT_SCHEDULER,
        help="Custom instructions for the assistant"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for generation (0.0-2.0)"
    )
    
    parser.add_argument(
        "--audio-mode",
        choices=["conversation", "dictation", "playback_only", "off"],
        default="conversation",
        help="Audio processing mode"
    )
    
    parser.add_argument(
        "--input-device",
        type=int,
        help="Input device index for audio"
    )
    
    parser.add_argument(
        "--output-device",
        type=int,
        help="Output device index for audio"
    )
    
    parser.add_argument(
        "--select-devices",
        action="store_true",
        help="Prompt for audio device selection"
    )
    
    parser.add_argument(
        "--save-recordings",
        action="store_true",
        help="Save audio recordings to disk"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Map string audio mode to enum
    audio_mode_map = {
        "conversation": AudioMode.CONVERSATION,
        "dictation": AudioMode.DICTATION,
        "playback_only": AudioMode.PLAYBACK_ONLY,
        "off": AudioMode.OFF
    }
    audio_mode = audio_mode_map.get(args.audio_mode, AudioMode.CONVERSATION)
    
    # Run the application
    try:
        asyncio.run(
            run_application(
                assistant_id=args.assistant_id,
                instructions=args.instructions,
                tools=APPOINTMENT_TOOLS,
                temperature=args.temperature,
                audio_mode=audio_mode,
                input_device=args.input_device,
                output_device=args.output_device,
                select_devices=args.select_devices,
                save_recordings=args.save_recordings,
                debug_mode=args.debug
            )
        )
        return 0
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        return 0
    except Exception as e:
        print(f"\nApplication error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
