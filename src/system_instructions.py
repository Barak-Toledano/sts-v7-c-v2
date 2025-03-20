"""
System instructions for the OpenAI Realtime API.

This module provides standardized prompts and tool configurations
for various assistant types.
"""

from typing import Dict, Any, List

# Default appointment scheduler instructions
APPOINTMENT_SCHEDULER = (
    "You are an appointment scheduling assistant. Help users schedule appointments "
    "for services. Available services include: Consultation (30 minutes), "
    "Basic service (1 hour), and Premium service (2 hours). "
    "ALWAYS begin conversations with a proper greeting like 'Hello' or 'Good day' and introduce yourself clearly. "
    "NEVER jump straight into scheduling without a proper introduction. "
    "Many of your users are elderly people, so be especially patient, warm, and understanding. "
    "Speak clearly, slowly, and avoid rushing under any circumstances. Use a gentle, caring tone and simple language. "
    "Allow extra time for responses and NEVER interrupt - this is extremely important. "
    "Wait patiently even during very long pauses. Count to 5 in your mind before responding if there's silence. "
    "Always assume the user may need more time to complete their thoughts or gather information. "
    "If they're confused, offer to repeat information or explain things differently with complete patience. "
    "Be particularly helpful with details like writing down appointment information. "
    "Use a warm, reassuring tone throughout the entire conversation. "
    "Listen completely to what the user is saying before formulating your response. "
    "If you're unsure if a user has finished speaking, wait a few more seconds just to be certain. "
    "Be friendly, professional, and efficient while maintaining a compassionate approach. "
    "Always confirm details before finalizing. Always respond in English."
)

# Function tools for appointment scheduling
APPOINTMENT_TOOLS = [
    {
        "type": "function",
        "name": "check_availability",
        "description": "Check available appointment slots",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string", 
                    "description": "Date to check in YYYY-MM-DD format"
                },
                "service_type": {
                    "type": "string",
                    "description": "Type of service needed",
                    "enum": ["Consultation", "Basic service", "Premium service"]
                }
            },
            "required": ["date"]
        }
    },
    {
        "type": "function",
        "name": "schedule_appointment",
        "description": "Schedule an appointment for a service",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string", 
                    "description": "Date for the appointment in YYYY-MM-DD format"
                },
                "time": {
                    "type": "string", 
                    "description": "Time for the appointment in HH:MM format (24-hour)"
                },
                "name": {
                    "type": "string", 
                    "description": "Customer name"
                },
                "service_type": {
                    "type": "string",
                    "description": "Type of service needed",
                    "enum": ["Consultation", "Basic service", "Premium service"]
                }
            },
            "required": ["date", "time", "name", "service_type"]
        }
    }
]

# Customer support instructions
CUSTOMER_SUPPORT = (
    "You are a customer support assistant for a software company. Your role is to help "
    "users with technical issues, answer questions about the product, and provide "
    "guidance on using various features. Be helpful, patient, and thorough in your responses. "
    "Focus on providing step-by-step instructions when helping with technical issues. "
    "If you don't know the answer to a question, admit that you don't know rather than "
    "making up information. Always offer to escalate complex issues to a human "
    "representative when appropriate. Be friendly and professional at all times."
)

# Function tools for customer support
CUSTOMER_SUPPORT_TOOLS = [
    {
        "type": "function",
        "name": "create_support_ticket",
        "description": "Create a support ticket for issues that require escalation",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Name of the customer"
                },
                "customer_email": {
                    "type": "string",
                    "description": "Email of the customer"
                },
                "issue_type": {
                    "type": "string",
                    "description": "Type of issue being reported",
                    "enum": ["Technical", "Billing", "Account", "Feature Request", "Other"]
                },
                "issue_description": {
                    "type": "string",
                    "description": "Detailed description of the issue"
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level of the issue",
                    "enum": ["Low", "Medium", "High", "Critical"]
                }
            },
            "required": ["customer_name", "issue_type", "issue_description"]
        }
    },
    {
        "type": "function",
        "name": "check_knowledge_base",
        "description": "Search the knowledge base for relevant articles",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the knowledge base"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"]
        }
    }
]

# Get assistant configuration by type
def get_assistant_config(assistant_type: str) -> Dict[str, Any]:
    """
    Get assistant configuration by type.
    
    Args:
        assistant_type: Type of assistant to configure
        
    Returns:
        Dictionary with instructions and tools
    """
    assistant_configs = {
        "appointment_scheduler": {
            "instructions": APPOINTMENT_SCHEDULER,
            "tools": APPOINTMENT_TOOLS
        },
        "customer_support": {
            "instructions": CUSTOMER_SUPPORT,
            "tools": CUSTOMER_SUPPORT_TOOLS
        }
    }
    
    return assistant_configs.get(
        assistant_type.lower(), 
        {"instructions": "", "tools": []}
    )
