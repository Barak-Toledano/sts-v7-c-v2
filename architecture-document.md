# OpenAI Realtime Assistant Architecture

## Overview
This document outlines the architecture of the OpenAI Realtime Assistant project, reflecting our refactored implementation with a domain-oriented approach.

## Core Components

### Configuration Management
The application uses a centralized configuration management approach:

1. **Environment Variables**:
   - Environment variables are loaded from a `.env` file using python-dotenv
   - Sensitive information (API keys) is stored only in environment variables
   - Default values are provided for optional configuration

2. **Configuration Classes**:
   - Pydantic models validate and enforce type constraints
   - Hierarchical configuration with nested settings classes (ApiSettings, AudioSettings, LoggingSettings)
   - Single point of access through `settings` singleton
   - Validators ensure configuration values are appropriate

3. **Directory Structure**:
   - `src/config/__init__.py` - Exports the settings singleton
   - `src/config/settings.py` - Defines the configuration classes and loads environment variables

4. **Key Components**:
   - `ApiSettings`: Manages OpenAI API credentials and model selection
   - `AudioSettings`: Controls audio parameters, device selection, and VAD settings
   - `LoggingSettings`: Configures logging behavior and formats
   - `Settings`: Main class that integrates all settings and handles directory creation


### Logging System
The application implements a robust, session-based logging system:

1. **Logger Management**:
   - Centralized configuration through the `LoggingManager` class
   - Session-based logging with unique identifiers
   - Prevention of duplicate logger configuration

2. **Output Destinations**:
   - Console logging with simplified format (configurable)
   - File logging with detailed format
   - Session-specific log files for easier troubleshooting

3. **Features**:
   - Automatic log directory creation
   - Log file rotation to manage disk space
   - Different formatting for console vs. file output
   - Global log level configuration with per-logger overrides

4. **Directory Structure**:
   - `src/config/logging_config.py` - Main logging configuration
   - `logs/` - Base directory for log files
   - `logs/sessions/` - Session-specific log files
   - `logs/app_[session_id].log` - Application-wide log for each session

### Event System
The application uses an event-driven architecture for communication between components:

1. **Event Types**:
   - Standardized event types enumerated in `EventType` 
   - Clear categorization of events by domain (user, system, audio, etc.)
   - Typed event definitions for better IDE support

2. **Event Bus**:
   - Centralized event bus for publishing and subscribing to events
   - Support for wildcard event handlers
   - Error isolation between event handlers

3. **Event Handlers**:
   - Components register handlers for events they care about
   - Events propagate through the system asynchronously
   - Clear event flow for tracking system behavior

4. **Directory Structure**:
   - `src/events/event_interface.py` - Defines event types, bus, and base classes

### Utility Functions
The application provides various utility modules to support common functionality:

1. **Error Handling** (`src/utils/error_handling.py`):
   - Hierarchical error classes with severity levels
   - Standardized error format with optional error codes
   - Safe execution patterns with proper exception handling
   - Protection of sensitive information from logs
   - Contextualized error reporting

2. **Async Utilities** (`src/utils/async_helpers.py`):
   - Task management for tracking and cancelling async tasks
   - Timeout handling for async operations
   - Retry mechanisms with exponential backoff
   - Debouncing and throttling for rate-limited operations
   - Event waiting with timeout support
   
3. **Audio Utilities** (`src/utils/audio_utilities.py`):
   - Audio format conversion and validation
   - WAV file reading and writing
   - Audio duration calculation
   - Silence detection and audio chunking
   - Format compatibility checks for OpenAI API
   
4. **Transcription Utilities** (`src/utils/transcription.py`):
   - Extraction and processing of transcription events from the OpenAI Realtime API
   - Session configuration generation for the API's built-in Whisper transcription
   - Text cleaning and formatting of transcription results
   - Segmentation of long transcriptions into manageable chunks
   - Saving and loading transcriptions with metadata

Each utility module follows these principles:
- Consistent logging approach using the configured logger
- No direct dependencies on application-specific modules
- Thorough error handling and reporting
- Comprehensive type hints for better IDE support

### Services
The application uses a service-oriented approach to interact with external systems:

1. **API Client** (`src/services/api_client.py`):
   - Manages WebSocket connections to the OpenAI Realtime API
   - Handles authentication, connectivity, and reconnection logic
   - Sends client events and processes server events
   - Provides a clean interface for the domain layer to interact with the API

2. **Audio Service** (`src/services/audio_service.py`):
   - Manages audio recording and playback
   - Handles device selection and audio format conversion
   - Implements Voice Activity Detection (VAD) integration
   - Buffers and processes audio streams in real-time

3. **Realtime Event Handler** (`src/services/realtime_event_handler.py`):
   - Centralizes processing of all events from the OpenAI Realtime API
   - Maps API events to application events
   - Validates event types and formats
   - Provides targeted error handling for API-specific issues

4. **Transcription Service** (`src/services/transcription_service.py`):
   - Integrates with the OpenAI Realtime API's Whisper transcription
   - Processes and formats transcription results
   - Emits application events for transcription updates
   - Manages transcription configuration and settings

5. **OpenAI Service** (`src/services/openai_service.py`):
   - Provides a higher-level interface to OpenAI's capabilities
   - Encapsulates API-specific details behind a clean facade
   - Manages authentication and session state
   - Handles synchronization between different API features

All services follow these design principles:
- Clear separation from domain logic
- Consistent error handling and reporting
- Proper resource cleanup and lifecycle management
- Event-based communication when appropriate

### Domain Logic
The application implements a domain-driven design with clear separation of concerns:

1. **Audio Domain** (`src/domain/audio/`):
   - `manager.py`: Coordinates audio recording, processing, and playback
   - Implements high-level audio operations independent of specific services
   - Manages audio state and lifecycle
   - Handles audio-specific events and commands

2. **Conversation Domain** (`src/domain/conversation/`):
   - `manager.py`: Orchestrates the conversation lifecycle and state transitions
   - `state.py`: Manages conversation state, history, and message flow
   - Implements the core business logic for conversational interaction
   - Coordinates between user inputs, model responses, and function calls

The domain layer follows these principles:
- Independence from specific implementation details
- Event-driven communication with other components
- State management through clear, well-defined state machines
- High-level business logic that encapsulates use cases

## Directory Structure

```
openai-realtime-assistant/
│
├── main.py                # Main application entry point
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── .env                   # Environment variables (gitignored)
├── architecture.md        # Architecture documentation
│
├── src/
│   ├── config/            # Configuration management
│   │   ├── __init__.py    # Exports settings singleton
│   │   ├── settings.py    # Configuration classes with Pydantic
│   │   └── logging_config.py # Logging configuration
│   │
│   ├── services/          # External service integration
│   │   ├── __init__.py
│   │   ├── api_client.py          # OpenAI Realtime API client
│   │   ├── audio_service.py       # Audio recording and playback
│   │   ├── base_service.py        # Abstract service interface
│   │   ├── openai_service.py      # OpenAI API integration
│   │   ├── transcription_service.py # Transcription processing
│   │   └── realtime_event_handler.py # API event handling
│   │
│   ├── domain/            # Domain logic with business rules
│   │   ├── __init__.py
│   │   ├── audio/         # Audio domain
│   │   │   ├── __init__.py
│   │   │   └── manager.py # Audio orchestration
│   │   │
│   │   └── conversation/  # Conversation domain
│   │       ├── __init__.py
│   │       ├── manager.py # Conversation orchestration
│   │       └── state.py   # Conversation state management
│   │
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── async_helpers.py   # Async utility functions
│   │   ├── audio_utilities.py # Audio processing utilities
│   │   ├── error_handling.py  # Error handling utilities
│   │   ├── logging_utils.py   # Logging utilities
│   │   └── transcription.py   # Transcription utilities
│   │
│   ├── presentation/      # User interface components
│   │   ├── __init__.py
│   │   └── cli.py         # Command-line interface
│   │
│   ├── events/            # Event system
│   │   ├── __init__.py
│   │   └── event_interface.py # Event definitions and bus
│   │
│   ├── application.py     # Main application class
│   ├── system_instructions.py # System instructions for the assistant
│   ├── __init__.py
│   └── __main__.py        # Entry point when run as module
│
└── tests/                 # Test directory
    ├── __init__.py
    ├── test_api_client.py
    ├── test_audio_service.py
    ├── test_config.py
    ├── test_conversation_manager.py
    ├── test_event_system.py
    ├── test_integration.py
    ├── test_security.py
    └── test_transcription.py
```

## Dependencies and Data Flow

The application follows a layered architecture with clear dependencies:

1. **Dependency Direction**:
   - Domain layer depends on events and utils only
   - Services depend on events and utils only
   - Presentation depends on domain, events, and utils
   - Utils have no internal dependencies

2. **Input Flow**:
   - Audio Input → Audio Service → Audio Manager → API Client → OpenAI API
   - CLI Command → CLI Interface → Conversation Manager → API Client → OpenAI API

3. **Output Flow**:
   - OpenAI API → API Client → Event Handler → Event Bus → Event Subscribers
   - Audio Response → Audio Service → Audio Output Device

4. **Event Flow**:
   - API events → Event Handler → Event Bus → Domain components
   - Domain events → Event Bus → Presentation components

5. **Configuration Flow**:
   - Environment Variables → Settings Classes → Application Components

This ensures a clean separation of concerns where:
- The domain layer contains pure business logic
- Services encapsulate external systems
- Utils provide shared functionality
- Presentation components focus on user interaction
- Components communicate via well-defined events

## Notes on Implementation

1. **Error Handling**:
   - Each layer has its own error handling approach
   - Domain errors represent business rule violations
   - Service errors represent external system failures
   - All errors are logged with appropriate context
   - Critical errors trigger application shutdown

2. **Resource Management**:
   - Services implement cleanup methods to release resources
   - The application ensures proper shutdown sequence
   - Async tasks are tracked and cancelled appropriately
   - External connections (WebSockets, audio devices) are managed carefully

3. **Configuration**:
   - Settings are validated at startup
   - Invalid configurations trigger early failure
   - Sensible defaults are provided for optional settings
   - Environment variables override defaults

4. **Testing**:
   - Each component has dedicated unit tests
   - Integration tests verify component interaction
   - Mocks are used for external dependencies
   - Test coverage focuses on critical paths

5. **Performance Considerations**:
   - Audio processing is optimized for real-time use
   - WebSocket communication is handled efficiently
   - Memory usage is carefully managed for long conversations
   - Background tasks are properly managed to avoid resource leaks

6. **Security Considerations**:
   - API keys are handled securely and never logged
   - Input validation prevents injection attacks
   - Error messages don't expose sensitive information
   - Secure default configurations are provided
