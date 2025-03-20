"""
Async utility functions for the OpenAI Realtime Assistant.

This module provides helper functions for working with asyncio,
including task management, cancellation, and timeouts.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Type, TypeVar, Union

from src.config.logging_config import get_logger
from src.utils.error_handling import safe_execute, AppError, ErrorSeverity

logger = get_logger(__name__)

T = TypeVar('T')


class TaskManager:
    """
    Manager for tracking and cleaning up async tasks.
    
    This class helps track running tasks and provides methods to
    safely cancel them when they're no longer needed.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize the task manager.
        
        Args:
            name: Name for this task manager (for logging)
        """
        self.name = name
        self.tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        logger.debug(f"TaskManager '{name}' initialized")
    
    def create_task(self, coro: Coroutine, name: Optional[str] = None) -> asyncio.Task:
        """
        Create and track a new asyncio task.
        
        Args:
            coro: Coroutine to run as a task
            name: Optional name for the task
            
        Returns:
            asyncio.Task: The created task
        """
        task = asyncio.create_task(coro)
        
        if name:
            # Set task name if supported by Python version
            if hasattr(task, "set_name"):
                task.set_name(name)
        
        # Add done callback to automatically remove task when completed
        task.add_done_callback(self._task_done_callback)
        
        # Track the task
        self.tasks.add(task)
        logger.debug(f"Task {name or id(task)} created")
        
        return task
    
    def _task_done_callback(self, task: asyncio.Task) -> None:
        """
        Callback for when a task is completed.
        
        Args:
            task: The completed task
        """
        # Remove the task from our set
        self.tasks.discard(task)
        
        # Check for and log exceptions
        if not task.cancelled():
            exception = task.exception()
            if exception:
                task_name = getattr(task, "get_name", lambda: id(task))()
                logger.error(f"Task {task_name} raised an exception: {exception}")
    
    async def cancel_all(self, wait: bool = True, timeout: Optional[float] = 5.0) -> None:
        """
        Cancel all tracked tasks.
        
        Args:
            wait: Whether to wait for tasks to complete
            timeout: Timeout in seconds if waiting, or None for no timeout
        """
        if not self.tasks:
            return
        
        # Cancel all tasks
        for task in list(self.tasks):
            if not task.done():
                task_name = getattr(task, "get_name", lambda: id(task))()
                logger.debug(f"Cancelling task {task_name}")
                task.cancel()
        
        if wait and self.tasks:
            # Wait for all tasks to complete or timeout
            try:
                done, pending = await asyncio.wait(
                    self.tasks, 
                    timeout=timeout, 
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Log any pending tasks
                if pending:
                    pending_names = [getattr(t, "get_name", lambda: id(t))() for t in pending]
                    logger.warning(f"Some tasks didn't complete within timeout: {pending_names}")
            except asyncio.CancelledError:
                logger.warning("Task cancellation was interrupted")
                raise
    
    async def shutdown(self, timeout: Optional[float] = 5.0) -> None:
        """
        Shut down the task manager and cancel all tasks.
        
        Args:
            timeout: Timeout in seconds for waiting for tasks to complete
        """
        # Set shutdown event
        self._shutdown_event.set()
        
        # Cancel all tasks
        await self.cancel_all(wait=True, timeout=timeout)
        
        logger.debug(f"TaskManager '{self.name}' shut down")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for the shutdown event to be set."""
        await self._shutdown_event.wait()


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    timeout_message: str = "Operation timed out",
    cancel_on_timeout: bool = True
) -> T:
    """
    Run a coroutine with a timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        timeout_message: Message for timeout error
        cancel_on_timeout: Whether to cancel the task on timeout
        
    Returns:
        T: Result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If the operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if cancel_on_timeout:
            # The task is automatically cancelled by wait_for on timeout
            logger.warning(f"Operation timed out after {timeout}s: {timeout_message}")
        raise


async def retry_async(
    coro_func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    retry_count: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    retry_on_result: Optional[Callable[[Any], bool]] = None,
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        coro_func: Async function to retry
        *args: Arguments for the function
        retry_count: Maximum number of retries
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Backoff factor for delay
        exceptions: Exception types to retry on
        retry_on_result: Function to check if result warrants a retry
        **kwargs: Keyword arguments for the function
        
    Returns:
        T: Result of the function
        
    Raises:
        Exception: If all retries fail
    """
    last_exception: Optional[Exception] = None
    delay = initial_delay
    
    # Try initial execution plus retries
    for attempt in range(retry_count + 1):
        try:
            result = await coro_func(*args, **kwargs)
            
            # Check if result warrants a retry
            if retry_on_result and retry_on_result(result):
                last_exception = ValueError(f"Retry condition met on result: {result}")
                raise last_exception
            
            # Successful execution
            return result
            
        except exceptions as e:
            last_exception = e
            
            # If this was the last attempt, re-raise
            if attempt >= retry_count:
                logger.warning(f"Max retries ({retry_count}) exceeded")
                break
            
            # Calculate next delay with exponential backoff
            delay = min(delay * backoff_factor, max_delay)
            
            # Log the retry
            logger.info(f"Attempt {attempt+1}/{retry_count+1} failed: {str(e)}. Retrying in {delay:.2f}s")
            
            # Wait before next attempt
            await asyncio.sleep(delay)
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    
    # This should never happen, but just in case
    raise ValueError("All retries failed without an exception")


def debounce(delay: float) -> Callable:
    """
    Create a decorator that debounces a coroutine.
    
    A debounced coroutine will only execute after a specified delay,
    and if called again during the delay, the previous call is cancelled.
    
    Args:
        delay: Delay in seconds
        
    Returns:
        Callable: Decorator function
    """
    def decorator(coro_func: Callable) -> Callable:
        # Store the last task for cancellation
        pending_task: Optional[asyncio.Task] = None
        
        @functools.wraps(coro_func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal pending_task
            
            # Cancel any pending task
            if pending_task and not pending_task.done():
                pending_task.cancel()
                
            # Create a new task that waits, then calls the original function
            async def delayed_call():
                try:
                    await asyncio.sleep(delay)
                    return await coro_func(*args, **kwargs)
                except asyncio.CancelledError:
                    # Task was cancelled, no need to propagate
                    pass
            
            # Schedule the delayed call
            pending_task = asyncio.create_task(delayed_call())
            return await pending_task
        
        return wrapper
    
    return decorator


def throttle(rate_limit: float) -> Callable:
    """
    Create a decorator that throttles a coroutine.
    
    A throttled coroutine will execute no more frequently than the rate limit.
    
    Args:
        rate_limit: Minimum time between executions in seconds
        
    Returns:
        Callable: Decorator function
    """
    def decorator(coro_func: Callable) -> Callable:
        # Track last execution time
        last_execution: float = 0
        
        @functools.wraps(coro_func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal last_execution
            
            # Get current time
            now = asyncio.get_event_loop().time()
            
            # Calculate time since last execution
            time_since_last = now - last_execution
            
            # If we need to wait, do so
            if time_since_last < rate_limit:
                wait_time = rate_limit - time_since_last
                await asyncio.sleep(wait_time)
            
            # Update last execution time and call function
            last_execution = asyncio.get_event_loop().time()
            return await coro_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


async def wait_for_event(event: asyncio.Event, timeout: Optional[float] = None) -> bool:
    """
    Wait for an event with timeout.
    
    Args:
        event: Event to wait for
        timeout: Timeout in seconds or None for no timeout
        
    Returns:
        bool: True if event was set, False if timeout occurred
    """
    if timeout is None:
        await event.wait()
        return True
    
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
