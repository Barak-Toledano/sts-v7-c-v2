"""
Transcription utilities for processing OpenAI Realtime API transcriptions.

This module primarily processes and formats transcriptions received from 
the OpenAI Realtime API, and provides utilities for handling transcription events.

Note: The actual transcription is performed by the OpenAI Realtime API using Whisper.
This module only handles configuration, event processing, and text formatting.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config.logging_config import get_logger
from src.utils.error_handling import AppError, ErrorSeverity

logger = get_logger(__name__)


def extract_transcription_from_realtime_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract transcription data from a Realtime API event.
    
    This function parses the 'conversation.item.input_audio_transcription.completed'
    event data from the OpenAI Realtime API.
    
    Args:
        event_data: The event data from the API
        
    Returns:
        Dictionary containing:
        - text: The transcribed text
        - is_final: Whether this is a final transcription
        - language: Detected language code if available
        - confidence: Confidence score (0-1) if available
        - timestamp: When the transcription was created
    """
    result = {
        "text": "",
        "is_final": True,
        "language": "en",  # Default language
        "confidence": 1.0,  # Default confidence
        "timestamp": time.time()
    }
    
    try:
        content = event_data.get("content", {})
        
        # Extract text
        if "text" in content:
            result["text"] = content["text"]
        
        # Extract language if available
        if "language" in content:
            result["language"] = content["language"]
            
        # Extract additional metadata
        if "metadata" in content:
            metadata = content["metadata"]
            if "is_final" in metadata:
                result["is_final"] = metadata["is_final"]
            if "confidence" in metadata:
                result["confidence"] = metadata["confidence"]
        
        return result
    except Exception as e:
        logger.error(f"Error extracting transcription data: {str(e)}")
        return result


def handle_realtime_transcription_event(event_data: Dict[str, Any]) -> Optional[str]:
    """
    Process a transcription event from the OpenAI Realtime API.
    
    Args:
        event_data: The event data received from the Realtime API
        
    Returns:
        Formatted transcript text, or None if no transcript was found
    """
    try:
        # Extract transcript from event data
        content = event_data.get("content", {})
        transcript = content.get("text", "")
        
        if not transcript:
            logger.warning("No transcript found in transcription event")
            return None
        
        # Clean and format the transcript
        formatted_transcript = format_transcription(transcript)
        
        logger.debug(f"Processed realtime transcription: '{formatted_transcript}'")
        return formatted_transcript
        
    except Exception as e:
        error = AppError(
            f"Failed to process realtime transcription event: {str(e)}",
            severity=ErrorSeverity.WARNING,
            cause=e,
        )
        logger.error(str(error))
        return None


def clean_transcription(text: str) -> str:
    """
    Clean a transcription by removing artifacts and normalizing text.
    
    Args:
        text: The raw transcription text
        
    Returns:
        Cleaned transcription text
    """
    if not text:
        return ""
    
    # Remove speaker labels like "[Speaker 1]:" or "(Speaker 2):"
    text = re.sub(r'\[Speaker\s*\d+\]:\s*', '', text)
    text = re.sub(r'\(Speaker\s*\d+\):\s*', '', text)
    
    # Remove timestamps like [00:01:23]
    text = re.sub(r'\[\d{1,2}:\d{2}(:\d{2})?\]', '', text)
    
    # Remove sound indicators like [laughter], [applause], etc.
    text = re.sub(r'\[\w+\]', '', text)
    
    # Remove duplicate spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper capitalization of sentences
    sentences = re.split(r'([.!?])\s+', text.strip())
    result = ""
    
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i].strip()
            if sentence:
                # Capitalize first letter of sentence
                sentence = sentence[0].upper() + sentence[1:]
                result += sentence
                
            # Add punctuation if available
            if i + 1 < len(sentences):
                result += sentences[i + 1]
                
            # Add space after punctuation
            if i + 1 < len(sentences) and i + 2 < len(sentences):
                result += " "
    
    # If no sentences were processed, just return the capitalized original
    if not result and text.strip():
        result = text.strip()
        result = result[0].upper() + result[1:]
    
    # Clean up potential issues from the sentence splitting
    result = re.sub(r'\s+([.!?,;:])', r'\1', result)
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


def format_transcription(text: str, add_punctuation: bool = True) -> str:
    """
    Format transcription text for better readability.
    
    Args:
        text: The cleaned transcription text
        add_punctuation: Whether to try to add missing punctuation
        
    Returns:
        Formatted transcription text
    """
    if not text:
        return ""
    
    # Clean the text first
    cleaned_text = clean_transcription(text)
    
    # Add punctuation if requested and text doesn't end with punctuation
    if add_punctuation and cleaned_text and not re.search(r'[.!?]$', cleaned_text):
        cleaned_text += "."
    
    return cleaned_text


def save_transcription(
    text: str, 
    file_path: Union[str, Path], 
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save transcription to a file.
    
    Args:
        text: The transcription text
        file_path: Path to save the transcription
        metadata: Optional metadata to include
        
    Returns:
        True if saving was successful
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to metadata
        if metadata is None:
            metadata = {}
        
        metadata["timestamp"] = time.time()
        metadata["text_length"] = len(text)
        
        # Prepare data to save
        data = {
            "text": text,
            "metadata": metadata
        }
        
        # Save as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved transcription to {file_path}")
        return True
        
    except Exception as e:
        error = AppError(
            f"Failed to save transcription: {str(e)}",
            severity=ErrorSeverity.WARNING,
            cause=e,
        )
        logger.error(str(error))
        return False


def load_transcription(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load transcription from a file.
    
    Args:
        file_path: Path to the transcription file
        
    Returns:
        Dictionary containing the transcription text and metadata,
        or None if loading fails
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    try:
        if not file_path.exists():
            logger.warning(f"Transcription file not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded transcription from {file_path}")
        return data
        
    except Exception as e:
        error = AppError(
            f"Failed to load transcription: {str(e)}",
            severity=ErrorSeverity.WARNING,
            cause=e,
        )
        logger.error(str(error))
        return None


def split_transcription_into_segments(
    text: str, 
    max_length: int = 100, 
    split_on: str = ".,;:!? "
) -> List[str]:
    """
    Split a transcription into segments based on punctuation and maximum length.
    
    Args:
        text: The transcription text to split
        max_length: Maximum length of each segment
        split_on: Characters to split on (ordered by priority)
        
    Returns:
        List of text segments
    """
    if not text:
        return []
    
    if len(text) <= max_length:
        return [text]
    
    segments = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # If the remaining text is shorter than max_length, add it and break
        if current_pos + max_length >= text_length:
            segments.append(text[current_pos:])
            break
        
        # Look for a split point
        split_pos = current_pos + max_length
        
        # Try to find split points based on punctuation
        for char in split_on:
            # Find the last occurrence of the character before max_length
            last_char_pos = text.rfind(char, current_pos, split_pos)
            
            if last_char_pos != -1:
                # If character is a space, don't include it in the segment
                if char == " ":
                    split_pos = last_char_pos
                else:
                    split_pos = last_char_pos + 1
                break
        
        # If no suitable split point was found, force a split at max_length
        if split_pos == current_pos + max_length and text[split_pos-1] not in split_on:
            # Try to find the next space after max_length to avoid splitting words
            next_space = text.find(" ", split_pos)
            if next_space != -1 and next_space < split_pos + 20:  # Only look ahead 20 chars
                split_pos = next_space
        
        # Add the segment
        segments.append(text[current_pos:split_pos].strip())
        current_pos = split_pos
    
    return segments


def generate_realtime_session_config(
    model: str = "whisper-1",
    vad_enabled: bool = True,
    auto_response: bool = True,
    vad_threshold: float = 0.82,
    silence_duration_ms: int = 1200,
    prefix_padding_ms: int = 500
) -> Dict[str, Any]:
    """
    Generate configuration for enabling transcription and VAD in a Realtime API session.
    
    Args:
        model: The model to use for transcription (e.g., "whisper-1")
        vad_enabled: Whether to enable voice activity detection
        auto_response: Whether to automatically generate responses when user stops speaking
        vad_threshold: Threshold for voice detection (higher is less sensitive, 0-1 range)
        silence_duration_ms: Silence duration in ms to consider speech ended
        prefix_padding_ms: Padding before speech in ms
        
    Returns:
        Dictionary containing session configuration
    """
    config = {
        "input_audio_transcription": {
            "model": model
        }
    }
    
    # Add turn detection (VAD) configuration if enabled
    if vad_enabled:
        config["turn_detection"] = {
            "type": "server_vad",
            "threshold": vad_threshold,
            "prefix_padding_ms": prefix_padding_ms,
            "silence_duration_ms": silence_duration_ms,
            "create_response": auto_response,
            "interrupt_response": True  # Enable interruption of assistant by default
        }
    else:
        # Explicitly disable VAD
        config["turn_detection"] = None
    
    return config