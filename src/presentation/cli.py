"""
Command-line interface for the OpenAI Realtime Assistant.

This module provides a CLI interface for interacting with the assistant,
displaying status information, and controlling the conversation.
"""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.config import settings
from src.config.logging_config import get_logger
from src.domain.conversation.manager import ConversationState
from src.events.event_interface import Event, EventType, event_bus
from src.utils.error_handling import AppError, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class CliInterface:
    """
    Command-line interface for the OpenAI Realtime Assistant.
    
    This class provides a terminal-based interface for interacting with
    the assistant, including command processing, status display, and
    conversation visualization.
    """
    
    def __init__(
        self,
        show_timestamps: bool = False,
        compact_mode: bool = False,
        color_output: bool = True,
        show_status: bool = True,
    ):
        """
        Initialize the CLI interface.
        
        Args:
            show_timestamps: Whether to show timestamps for messages
            compact_mode: Whether to use a compact display mode
            color_output: Whether to use colored output
            show_status: Whether to show status information
        """
        self.show_timestamps = show_timestamps
        self.compact_mode = compact_mode
        self.color_output = color_output and self._supports_color()
        self.show_status = show_status
        
        # Display state
        self.conversation_state = ConversationState.IDLE
        self.last_status_update = 0.0
        self.status_update_interval = 0.5  # seconds
        self.transcript_cache = []
        self.current_user_input = ""
        
        # Signal handling
        self.shutdown_event = asyncio.Event()
        self._shutdown_in_progress = False
        
        # Initialize terminal colors
        self._init_terminal_colors()
        
        # Event handlers tracking for cleanup
        self._registered_handlers = []
        
        # Register event handlers
        self._register_event_handlers()
    
    def _init_terminal_colors(self) -> None:
        """Initialize terminal color codes based on terminal capabilities."""
        if self.color_output:
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.RED = "\033[31m"
            self.GREEN = "\033[32m"
            self.YELLOW = "\033[33m"
            self.BLUE = "\033[34m"
            self.MAGENTA = "\033[35m"
            self.CYAN = "\033[36m"
            self.GRAY = "\033[90m"
        else:
            self.RESET = ""
            self.BOLD = ""
            self.RED = ""
            self.GREEN = ""
            self.YELLOW = ""
            self.BLUE = ""
            self.MAGENTA = ""
            self.CYAN = ""
            self.GRAY = ""
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for conversation events."""
        self._register_handler(EventType.CONVERSATION_STATE_CHANGED, self._handle_state_changed)
        
        # Speech events
        self._register_handler(EventType.USER_SPEECH_STARTED, self._handle_user_speech_started)
        self._register_handler(EventType.USER_SPEECH_ONGOING, self._handle_user_speech_content)
        self._register_handler(EventType.USER_SPEECH_FINISHED, self._handle_user_speech_finished)
        
        # Assistant events
        self._register_handler(EventType.ASSISTANT_MESSAGE_STARTED, self._handle_assistant_message_started)
        self._register_handler(EventType.ASSISTANT_MESSAGE_CONTENT, self._handle_assistant_message_content)
        self._register_handler(EventType.ASSISTANT_MESSAGE_COMPLETED, self._handle_assistant_message_completed)
        
        # Error events
        self._register_handler(EventType.ERROR, self._handle_error)
        
        # Shutdown events
        self._register_handler(EventType.SHUTDOWN, self._handle_shutdown)
    
    def _register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register an event handler and track it for cleanup.
        
        Args:
            event_type: The event type to listen for
            handler: The handler function
        """
        event_bus.on(event_type, handler)
        self._registered_handlers.append((event_type, handler))
    
    def _handle_state_changed(self, event: Event) -> None:
        """
        Handle conversation state change events.
        
        Args:
            event: State change event
        """
        try:
            new_state = event.data.get("state")
            if isinstance(new_state, str):
                # Convert string to enum if needed
                self.conversation_state = getattr(ConversationState, new_state, ConversationState.IDLE)
            else:
                self.conversation_state = new_state or ConversationState.IDLE
                
            self._update_status_display()
        except Exception as e:
            logger.error(f"Error handling state change: {str(e)}")
    
    def _handle_user_speech_started(self, event: Event) -> None:
        """
        Handle user speech started events.
        
        Args:
            event: User speech started event
        """
        if not self.compact_mode:
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            self._print_safe(f"\n{timestamp}{self.BOLD}{self.BLUE}You: {self.RESET}", end="", flush=True)
    
    def _handle_user_speech_content(self, event: Event) -> None:
        """
        Handle user speech content events.
        
        Args:
            event: User speech content event
        """
        transcript = event.data.get("text", "")
        if transcript and transcript != self.current_user_input:
            if self.compact_mode:
                # In compact mode, wait until the speech is finished
                self.current_user_input = transcript
            else:
                # Clear the line and print the updated transcript
                # ANSI escape for clearing line and returning to start
                self._print_safe(f"\r\033[K{self.BOLD}{self.BLUE}You: {self.RESET}{transcript}", end="", flush=True)
                self.current_user_input = transcript
    
    def _handle_user_speech_finished(self, event: Event) -> None:
        """
        Handle user speech finished events.
        
        Args:
            event: User speech finished event
        """
        final_transcript = event.data.get("text", "")
        
        if self.compact_mode:
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            self._print_safe(f"{timestamp}{self.BOLD}{self.BLUE}You: {self.RESET}{final_transcript}")
        else:
            # Add a newline to complete the transcript
            self._print_safe("")
        
        # Reset the current input
        self.current_user_input = ""
    
    def _handle_assistant_message_started(self, event: Event) -> None:
        """
        Handle assistant message started events.
        
        Args:
            event: Assistant message started event
        """
        if not self.compact_mode:
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            self._print_safe(f"{timestamp}{self.BOLD}{self.GREEN}Assistant: {self.RESET}", end="", flush=True)
        
        # Clear transcript cache
        self.transcript_cache = []
    
    def _handle_assistant_message_content(self, event: Event) -> None:
        """
        Handle assistant message content events.
        
        Args:
            event: Assistant message content event
        """
        content = event.data.get("text", "")
        
        if content:
            if self.compact_mode:
                # In compact mode, accumulate content for later display
                self.transcript_cache.append(content)
            else:
                # In normal mode, print incremental updates
                self._print_safe(content, end="", flush=True)
    
    def _handle_assistant_message_completed(self, event: Event) -> None:
        """
        Handle assistant message completed events.
        
        Args:
            event: Assistant message completed event
        """
        if self.compact_mode and self.transcript_cache:
            # In compact mode, print the full message at once
            timestamp = self._format_timestamp() if self.show_timestamps else ""
            message = "".join(self.transcript_cache)
            self._print_safe(f"{timestamp}{self.BOLD}{self.GREEN}Assistant: {self.RESET}{message}")
            self.transcript_cache = []
        else:
            # Add a newline to complete the output
            self._print_safe("")
    
    def _handle_error(self, event: Event) -> None:
        """
        Handle error events.
        
        Args:
            event: Error event
        """
        error_data = event.data.get("error", {})
        error_message = error_data.get("message", "Unknown error")
        error_type = error_data.get("type", "unknown")
        
        timestamp = self._format_timestamp() if self.show_timestamps else ""
        self._print_safe(f"{timestamp}{self.BOLD}{self.RED}Error ({error_type}): {self.RESET}{error_message}")
    
    def _handle_shutdown(self, event: Event) -> None:
        """
        Handle shutdown events.
        
        Args:
            event: Shutdown event
        """
        if not self._shutdown_in_progress:
            self._shutdown_in_progress = True
            reason = event.data.get("reason", "unknown")
            logger.info(f"CLI received shutdown event. Reason: {reason}")
            
            # Set the shutdown event to stop input loop
            self.shutdown_event.set()
    
    def _update_status_display(self) -> None:
        """Update the status display with current state information."""
        if not self.show_status:
            return
        
        # Rate limit status updates
        current_time = time.time()
        if current_time - self.last_status_update < self.status_update_interval:
            return
        
        self.last_status_update = current_time
        
        # Map states to display strings
        state_display = {
            ConversationState.IDLE: f"{self.GRAY}Idle{self.RESET}",
            ConversationState.CONNECTING: f"{self.YELLOW}Connecting...{self.RESET}",
            ConversationState.READY: f"{self.GREEN}Ready{self.RESET}",
            ConversationState.USER_SPEAKING: f"{self.BLUE}Listening...{self.RESET}",
            ConversationState.THINKING: f"{self.YELLOW}Thinking...{self.RESET}",
            ConversationState.ASSISTANT_SPEAKING: f"{self.GREEN}Speaking...{self.RESET}",
            ConversationState.ERROR: f"{self.RED}Error{self.RESET}",
            ConversationState.DISCONNECTED: f"{self.GRAY}Disconnected{self.RESET}",
        }
        
        # Get display string for current state
        status = state_display.get(self.conversation_state, f"{self.GRAY}Unknown{self.RESET}")
        
        # Print status (replacing the current line)
        try:
            # ANSI escape for clearing line and returning to start
            sys.stdout.write(f"\r\033[K{status}")
            sys.stdout.flush()
        except (IOError, BrokenPipeError) as e:
            logger.debug(f"Failed to update status display: {str(e)}")
    
    def _format_timestamp(self) -> str:
        """
        Format a timestamp for display.
        
        Returns:
            Formatted timestamp string
        """
        current_time = time.strftime("%H:%M:%S")
        return f"{self.GRAY}[{current_time}] {self.RESET}"
    
    def _supports_color(self) -> bool:
        """
        Check if the terminal supports colored output.
        
        Returns:
            True if color is supported
        """
        # Check for NO_COLOR environment variable (https://no-color.org/)
        if os.environ.get("NO_COLOR"):
            return False
        
        # Check for forced color mode in settings
        if hasattr(settings, 'force_color') and settings.force_color:
            return True
        
        # Check for color support in various terminals
        plat = sys.platform
        supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
        
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        return supported_platform and is_a_tty
    
    def _print_safe(self, *args, **kwargs) -> None:
        """
        Print to stdout with error handling.
        
        Args:
            *args: Arguments to pass to print
            **kwargs: Keyword arguments to pass to print
        """
        try:
            print(*args, **kwargs)
        except (IOError, BrokenPipeError) as e:
            logger.debug(f"Failed to print to stdout: {str(e)}")
    
    def display_welcome_message(self) -> None:
        """Display a welcome message when the application starts."""
        app_version = getattr(settings, 'app_version', '0.1.0')
        
        self._print_safe(f"\n{self.BOLD}{self.CYAN}=== OpenAI Realtime Assistant ==={self.RESET}")
        self._print_safe(f"{self.GRAY}Version: {app_version}{self.RESET}")
        self._print_safe(f"{self.GRAY}Speak naturally or type commands prefixed with '/'{self.RESET}")
        self._print_safe(f"{self.GRAY}Type /help for a list of available commands{self.RESET}\n")
    
    def display_help(self) -> None:
        """Display help information."""
        self._print_safe(f"\n{self.BOLD}Available Commands:{self.RESET}")
        self._print_safe(f"  {self.BOLD}/quit{self.RESET} - Exit the application")
        self._print_safe(f"  {self.BOLD}/help{self.RESET} - Display this help message")
        self._print_safe(f"  {self.BOLD}/restart{self.RESET} - Restart the conversation")
        self._print_safe(f"  {self.BOLD}/pause{self.RESET} - Pause listening")
        self._print_safe(f"  {self.BOLD}/resume{self.RESET} - Resume listening")
        self._print_safe(f"  {self.BOLD}/interrupt{self.RESET} - Interrupt the assistant")
        self._print_safe(f"  {self.BOLD}/status{self.RESET} - Toggle status display")
        self._print_safe(f"  {self.BOLD}/timestamps{self.RESET} - Toggle timestamps")
        self._print_safe(f"  {self.BOLD}/compact{self.RESET} - Toggle compact mode")
        self._print_safe("")
    
    def parse_command(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Parse user input for commands.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_command, command, args)
        """
        if not text:
            return False, None, None
            
        text = text.strip()
        
        # Check if this is a command (starts with /)
        if not text.startswith('/'):
            return False, None, None
        
        # Split into command and arguments
        parts = text.split(' ', 1)
        command = parts[0][1:].lower()  # Remove the leading /
        args = parts[1] if len(parts) > 1 else None
        
        return True, command, args
    
    def process_command(
        self, 
        command: str, 
        args: Optional[str] = None, 
        command_callbacks: Optional[Dict[str, Callable]] = None
    ) -> bool:
        """
        Process a command from the user.
        
        Args:
            command: Command name (without the leading /)
            args: Command arguments
            command_callbacks: Dictionary of command callbacks
            
        Returns:
            True if application should continue, False if it should exit
        """
        # Handle internal commands
        if command == "help":
            self.display_help()
            return True
        
        elif command == "status":
            self.show_status = not self.show_status
            self._print_safe(f"Status display: {'on' if self.show_status else 'off'}")
            return True
        
        elif command == "timestamps":
            self.show_timestamps = not self.show_timestamps
            self._print_safe(f"Timestamps: {'on' if self.show_timestamps else 'off'}")
            return True
        
        elif command == "compact":
            self.compact_mode = not self.compact_mode
            self._print_safe(f"Compact mode: {'on' if self.compact_mode else 'off'}")
            return True
        
        # Handle external commands via callbacks
        if command_callbacks and command in command_callbacks:
            try:
                command_callbacks[command](args)
            except Exception as e:
                logger.error(f"Error executing command '{command}': {str(e)}")
                self._print_safe(f"{self.RED}Error executing command: {str(e)}{self.RESET}")
            return True
        
        # Handle quit command
        if command in ["quit", "exit", "bye"]:
            self._print_safe("Exiting application...")
            # Emit shutdown event
            event_bus.emit(
                EventType.SHUTDOWN, 
                {"reason": "user_command", "command": command}
            )
            return False
        
        # Unknown command
        self._print_safe(f"{self.YELLOW}Unknown command: /{command}{self.RESET}")
        self._print_safe(f"Type {self.BOLD}/help{self.RESET} for a list of commands")
        return True
    
    async def run_input_loop(
        self, 
        command_callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Run the input loop to process user commands from the terminal.
        
        Args:
            command_callbacks: Dictionary of command callbacks for external commands
        """
        # Display welcome message
        self.display_welcome_message()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Input task tracking
        input_task = None
        
        try:
            # Function to handle terminal input
            async def handle_input():
                try:
                    while not self.shutdown_event.is_set():
                        # Get input in a non-blocking way
                        try:
                            line = await asyncio.to_thread(input)
                        except EOFError:
                            # Handle CTRL+D
                            logger.info("Received EOF (CTRL+D), shutting down")
                            self.shutdown_event.set()
                            break
                        
                        if self.shutdown_event.is_set():
                            break
                            
                        if not line:
                            continue
                            
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check if this is a command
                        is_command, command, args = self.parse_command(line)
                        
                        if is_command:
                            # Process command
                            should_continue = self.process_command(command, args, command_callbacks)
                            if not should_continue:
                                break
                        else:
                            # Regular text input - emit as text command
                            if command_callbacks and "text_input" in command_callbacks:
                                try:
                                    command_callbacks["text_input"](line)
                                except Exception as e:
                                    logger.error(f"Error processing text input: {str(e)}")
                                    self._print_safe(f"{self.RED}Error processing input: {str(e)}{self.RESET}")
                except Exception as e:
                    logger.error(f"Error in input loop: {str(e)}")
                    # Try to emit a shutdown event in case of error
                    try:
                        event_bus.emit(
                            EventType.SHUTDOWN, 
                            {"reason": "input_loop_error", "error": str(e)}
                        )
                    except:
                        pass
                    self.shutdown_event.set()
            
            # Start the input handling task
            input_task = asyncio.create_task(handle_input())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        finally:
            # Clean up
            if input_task and not input_task.done():
                input_task.cancel()
                try:
                    await input_task
                except asyncio.CancelledError:
                    pass
                    
            # Unregister event handlers
            self._cleanup_event_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        # This is safe to call from the main thread only
        if not self._is_main_thread():
            return
            
        # Only try to set these up if we're on a platform that supports them
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, 
                    lambda s=sig: asyncio.create_task(self._handle_signal(s))
                )
        except (NotImplementedError, AttributeError, RuntimeError):
            # Windows doesn't support add_signal_handler, or we're not in an event loop
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self._handle_signal_sync)
    
    async def _handle_signal(self, sig: int) -> None:
        """
        Handle a received signal asynchronously.
        
        Args:
            sig: Signal number
        """
        if self._shutdown_in_progress:
            return
            
        logger.info(f"Received signal {sig}, initiating shutdown")
        self._shutdown_in_progress = True
        
        # Emit a shutdown event
        event_bus.emit(
            EventType.SHUTDOWN, 
            {"reason": "signal", "signal": sig}
        )
        
        # Set the shutdown event
        self.shutdown_event.set()
    
    def _handle_signal_sync(self, sig: int, frame) -> None:
        """
        Handle a received signal synchronously.
        
        Args:
            sig: Signal number
            frame: Frame object
        """
        if self._shutdown_in_progress:
            return
            
        logger.info(f"Received signal {sig}, initiating shutdown")
        self._shutdown_in_progress = True
        
        # This is a synchronous handler, so we need to set the event directly
        self.shutdown_event.set()
    
    def _is_main_thread(self) -> bool:
        """Check if the current thread is the main thread."""
        import threading
        return threading.current_thread() is threading.main_thread()
    
    def _cleanup_event_handlers(self) -> None:
        """Clean up registered event handlers."""
        for event_type, handler in self._registered_handlers:
            try:
                event_bus.off(event_type, handler)
            except Exception as e:
                logger.debug(f"Error unregistering handler: {str(e)}")
        
        self._registered_handlers = []


def print_cli_header(version: str = "0.1.0") -> None:
    """
    Print application header with version information.
    
    Args:
        version: Application version
    """
    header = f"""
╭──────────────────────────────────────────────╮
│ OpenAI Realtime Assistant                     │
│ Version: {version.ljust(35)}│
│ Realtime API - Voice Conversation Interface   │
╰──────────────────────────────────────────────╯
"""
    print(header)