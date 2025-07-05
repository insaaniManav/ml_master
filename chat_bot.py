#!/usr/bin/env python3
"""
ML-Master Conversational Bot

An interactive conversational interface for the ML-Master AI-for-AI framework.
This bot allows users to have natural conversations and get AI assistance
for various tasks including code generation, documentation, and problem solving.
"""

import os
import sys
import json
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.integration import MLMaster
from core.integration.content_types import ContentType, ContentTypeDetector


class MLMasterBot:
    """
    Conversational bot interface for ML-Master framework
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the conversational bot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_master = None
        self.conversation_history = []
        self.session_start_time = time.time()
        self.user_name = "User"
        self.bot_name = "ML-Master"
        
        # Bot personality and capabilities
        self.capabilities = {
            "code_generation": "Generate code in Python, JavaScript, Java, C++, SQL, HTML, CSS, and more",
            "documentation": "Create README files, documentation, tutorials, and guides",
            "problem_solving": "Solve complex problems using AI reasoning and exploration",
            "data_analysis": "Help with data analysis, visualization, and machine learning",
            "explanation": "Provide detailed explanations of concepts, algorithms, and processes",
            "structured_data": "Generate JSON, YAML, XML, and other structured data formats"
        }
        
        # Initialize ML-Master framework
        self._initialize_framework()
    
    def _initialize_framework(self):
        """Initialize the ML-Master framework"""
        try:
            logger.info("Initializing ML-Master framework for conversational bot...")
            self.ml_master = MLMaster(self.config)
            logger.info("Framework initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize framework: {e}")
            raise
    
    def start_conversation(self):
        """Start the conversational interface"""
        self._print_welcome_message()
        
        try:
            while True:
                # Get user input
                user_input = self._get_user_input()
                
                # Check for exit commands
                if self._is_exit_command(user_input):
                    break
                
                # Process the input
                response = self._process_user_input(user_input)
                
                # Display response
                self._display_response(response)
                
                # Add to conversation history
                self._add_to_history(user_input, response)
                
        except KeyboardInterrupt:
            print(f"\n\n{self.bot_name}: Goodbye! Thanks for chatting with me!")
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            print(f"\n{self.bot_name}: Sorry, I encountered an error. Please try again.")
        finally:
            self._cleanup()
    
    def _print_welcome_message(self):
        """Print the welcome message"""
        print(f"\n{'='*80}")
        print(f"🤖 Welcome to {self.bot_name} - Your AI Assistant!")
        print(f"{'='*80}")
        print(f"\nI'm an AI-powered assistant that can help you with:")
        
        for capability, description in self.capabilities.items():
            print(f"  • {capability.replace('_', ' ').title()}: {description}")
        
        print(f"\n💡 Tips:")
        print(f"  • Be specific about what you want me to help you with")
        print(f"  • I can generate code, documentation, solve problems, and more")
        print(f"  • Type 'help' for more information about my capabilities")
        print(f"  • Type 'exit', 'quit', or 'bye' to end our conversation")
        print(f"  • Type 'history' to see our conversation history")
        print(f"  • Type 'stats' to see framework statistics")
        
        print(f"\n{self.bot_name}: Hi! I'm ready to help you. What would you like to work on today?")
    
    def _get_user_input(self) -> str:
        """Get user input with a nice prompt"""
        try:
            user_input = input(f"\n{self.user_name}: ").strip()
            return user_input
        except EOFError:
            return "exit"
    
    def _is_exit_command(self, user_input: str) -> bool:
        """Check if user wants to exit"""
        exit_commands = ['exit', 'quit', 'bye', 'goodbye', 'stop']
        return user_input.lower() in exit_commands
    
    def _process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response"""
        try:
            # Handle special commands
            if user_input.lower() == 'help':
                return self._handle_help_command()
            elif user_input.lower() == 'history':
                return self._handle_history_command()
            elif user_input.lower() == 'stats':
                return self._handle_stats_command()
            elif user_input.lower() == 'capabilities':
                return self._handle_capabilities_command()
            elif user_input.lower().startswith('generate'):
                return self._handle_generate_command(user_input)
            elif user_input.lower().startswith('solve'):
                return self._handle_solve_command(user_input)
            else:
                # Treat as a general task
                return self._handle_general_task(user_input)
                
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return {
                'type': 'error',
                'content': f"Sorry, I encountered an error while processing your request: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def _handle_help_command(self) -> Dict[str, Any]:
        """Handle help command"""
        help_text = f"""
🤖 {self.bot_name} Help Guide

I can help you with various tasks. Here are some ways to interact with me:

📝 **General Tasks:**
  • Just ask me anything! I'll try to help you with explanations, code, or solutions
  • "Explain how neural networks work"
  • "Help me understand machine learning algorithms"

💻 **Code Generation:**
  • "Write a Python function to sort a list"
  • "Create a React component for a todo list"
  • "Generate SQL queries for a database"

📚 **Documentation:**
  • "Write a README for my Python project"
  • "Create documentation for an API"
  • "Write a tutorial on web development"

🔧 **Problem Solving:**
  • "Solve this algorithm problem: [describe the problem]"
  • "Help me optimize this code"
  • "Debug this Python script"

📊 **Data & Analysis:**
  • "Create a data visualization script"
  • "Generate a machine learning pipeline"
  • "Help me analyze this dataset"

🎯 **Special Commands:**
  • 'help' - Show this help message
  • 'history' - Show conversation history
  • 'stats' - Show framework statistics
  • 'capabilities' - Show my capabilities
  • 'exit' - End the conversation

💡 **Tips:**
  • Be specific about what you want
  • I can handle multiple content types (code, markdown, structured data)
  • I remember our conversation context
  • I can explore multiple solutions to complex problems
        """
        
        return {
            'type': 'help',
            'content': help_text,
            'metadata': {'command': 'help'}
        }
    
    def _handle_history_command(self) -> Dict[str, Any]:
        """Handle history command"""
        if not self.conversation_history:
            return {
                'type': 'history',
                'content': "No conversation history yet. Let's start chatting!",
                'metadata': {'command': 'history', 'count': 0}
            }
        
        history_text = f"📚 Conversation History ({len(self.conversation_history)} exchanges):\n\n"
        
        for i, (user_msg, bot_response) in enumerate(self.conversation_history[-10:], 1):
            history_text += f"{i}. {self.user_name}: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}\n"
            history_text += f"   {self.bot_name}: {bot_response['content'][:100]}{'...' if len(bot_response['content']) > 100 else ''}\n\n"
        
        if len(self.conversation_history) > 10:
            history_text += f"... and {len(self.conversation_history) - 10} more exchanges"
        
        return {
            'type': 'history',
            'content': history_text,
            'metadata': {'command': 'history', 'count': len(self.conversation_history)}
        }
    
    def _handle_stats_command(self) -> Dict[str, Any]:
        """Handle stats command"""
        try:
            stats = self.ml_master.get_framework_stats()
            
            session_duration = time.time() - self.session_start_time
            
            stats_text = f"""
📊 Framework Statistics

⏱️  Session Duration: {session_duration:.1f} seconds
💬 Conversation Exchanges: {len(self.conversation_history)}
🧠 Memory Usage: {stats.get('memory_usage_percent', 0):.1f}%
🌳 Total Nodes Explored: {stats.get('total_nodes', 0)}
⚡ Framework Version: {stats.get('version', 'Unknown')}
🤖 Model Used: {stats.get('model_used', 'Unknown')}
            """
            
            return {
                'type': 'stats',
                'content': stats_text,
                'metadata': {'command': 'stats', 'stats': stats}
            }
        except Exception as e:
            return {
                'type': 'error',
                'content': f"Sorry, I couldn't retrieve statistics: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def _handle_capabilities_command(self) -> Dict[str, Any]:
        """Handle capabilities command"""
        capabilities_text = f"🎯 {self.bot_name} Capabilities:\n\n"
        
        for capability, description in self.capabilities.items():
            capabilities_text += f"• **{capability.replace('_', ' ').title()}**: {description}\n"
        
        capabilities_text += f"\n💡 I can also handle multiple content types:"
        capabilities_text += f"\n  • Code (Python, JavaScript, Java, C++, SQL, HTML, CSS, etc.)"
        capabilities_text += f"\n  • Documentation (Markdown, README, guides, tutorials)"
        capabilities_text += f"\n  • Structured Data (JSON, YAML, XML, CSV)"
        capabilities_text += f"\n  • Plain Text (explanations, descriptions, analysis)"
        
        return {
            'type': 'capabilities',
            'content': capabilities_text,
            'metadata': {'command': 'capabilities'}
        }
    
    def _handle_generate_command(self, user_input: str) -> Dict[str, Any]:
        """Handle generate command"""
        # Extract the task from the command
        task = user_input[9:].strip()  # Remove "generate " prefix
        
        if not task:
            return {
                'type': 'error',
                'content': "Please specify what you'd like me to generate. For example: 'generate a Python function to calculate fibonacci numbers'",
                'metadata': {'error': 'No task specified'}
            }
        
        try:
            # Use content generation mode (faster)
            content_output = self.ml_master.generate_content(
                task_description=task,
                content_type=None,  # Auto-detect
                format_output=True
            )
            
            return {
                'type': 'generated_content',
                'content': content_output.content,
                'metadata': {
                    'task': task,
                    'content_type': content_output.content_type.value,
                    'generation_time': content_output.metadata.get('generation_time', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return {
                'type': 'error',
                'content': f"Sorry, I encountered an error while generating content: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def _handle_solve_command(self, user_input: str) -> Dict[str, Any]:
        """Handle solve command"""
        # Extract the task from the command
        task = user_input[6:].strip()  # Remove "solve " prefix
        
        if not task:
            return {
                'type': 'error',
                'content': "Please specify what problem you'd like me to solve. For example: 'solve this algorithm problem: find the longest common subsequence'",
                'metadata': {'error': 'No task specified'}
            }
        
        try:
            # Use full task solving mode (with exploration)
            results = self.ml_master.solve_task(
                task_description=task,
                content_type=None,  # Auto-detect
                max_time=300,  # 5 minutes
                max_iterations=50
            )
            
            return {
                'type': 'solved_task',
                'content': results['solution']['content'],
                'metadata': {
                    'task': task,
                    'performance_score': results['solution']['performance_score'],
                    'iterations_completed': results['performance']['iterations_completed'],
                    'total_time': results['performance']['total_time']
                }
            }
            
        except Exception as e:
            logger.error(f"Error solving task: {e}")
            return {
                'type': 'error',
                'content': f"Sorry, I encountered an error while solving the task: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def _handle_general_task(self, user_input: str) -> Dict[str, Any]:
        """Handle general task input"""
        try:
            # Auto-detect content type
            content_type = ContentTypeDetector.detect_from_task(user_input)
            
            # Use content generation for general tasks (faster response)
            content_output = self.ml_master.generate_content(
                task_description=user_input,
                content_type=content_type,
                format_output=True
            )
            
            return {
                'type': 'general_response',
                'content': content_output.content,
                'metadata': {
                    'task': user_input,
                    'content_type': content_output.content_type.value,
                    'generation_time': content_output.metadata.get('generation_time', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error handling general task: {e}")
            return {
                'type': 'error',
                'content': f"Sorry, I encountered an error while processing your request: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def _display_response(self, response: Dict[str, Any]):
        """Display the bot's response"""
        response_type = response.get('type', 'general')
        content = response.get('content', '')
        metadata = response.get('metadata', {})
        
        # Add some formatting based on response type
        if response_type == 'error':
            print(f"\n❌ {self.bot_name}: {content}")
        elif response_type == 'help':
            print(f"\n{content}")
        elif response_type == 'history':
            print(f"\n{content}")
        elif response_type == 'stats':
            print(f"\n{content}")
        elif response_type == 'capabilities':
            print(f"\n{content}")
        elif response_type in ['generated_content', 'solved_task', 'general_response']:
            print(f"\n🤖 {self.bot_name}:")
            print(f"{content}")
            
            # Add metadata if available
            if metadata.get('generation_time'):
                print(f"\n⏱️  Generated in {metadata['generation_time']:.2f} seconds")
            if metadata.get('performance_score'):
                print(f"📊 Performance Score: {metadata['performance_score']:.4f}")
        else:
            print(f"\n{self.bot_name}: {content}")
    
    def _add_to_history(self, user_input: str, response: Dict[str, Any]):
        """Add exchange to conversation history"""
        self.conversation_history.append((user_input, response))
        
        # Keep only last 50 exchanges to manage memory
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.ml_master:
                self.ml_master.cleanup()
            logger.info("Bot cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or create default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Load default config
        default_config_path = Path(__file__).parent / "configs" / "config.yaml"
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with environment variables
    config = override_config_with_env(config)
    
    return config


def override_config_with_env(config: dict) -> dict:
    """Override configuration with environment variables"""
    # Azure OpenAI settings
    if os.getenv('AZURE_OPENAI_ENDPOINT'):
        config['reasoning']['azure_endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if os.getenv('AZURE_OPENAI_API_KEY'):
        config['reasoning']['subscription_key'] = os.getenv('AZURE_OPENAI_API_KEY')
    
    if os.getenv('AZURE_OPENAI_MODEL'):
        config['reasoning']['model_name'] = os.getenv('AZURE_OPENAI_MODEL')
        config['reasoning']['deployment'] = os.getenv('AZURE_OPENAI_MODEL')
    
    if os.getenv('AZURE_OPENAI_DEPLOYMENT'):
        config['reasoning']['deployment'] = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    
    # Framework settings
    if os.getenv('ML_MASTER_LOG_LEVEL'):
        config['framework']['log_level'] = os.getenv('ML_MASTER_LOG_LEVEL')
    
    if os.getenv('ML_MASTER_MAX_ITERATIONS'):
        config['exploration']['max_iterations'] = int(os.getenv('ML_MASTER_MAX_ITERATIONS'))
    
    if os.getenv('ML_MASTER_MAX_TIME'):
        config['exploration']['max_time'] = int(os.getenv('ML_MASTER_MAX_TIME'))
    
    return config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "ml_master_chat.log",
        level=log_level,
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def main():
    """Main entry point for the conversational bot"""
    parser = argparse.ArgumentParser(
        description="ML-Master Conversational Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the conversational bot
  python chat_bot.py
  
  # Start with custom config
  python chat_bot.py --config my_config.yaml
  
  # Start in debug mode
  python chat_bot.py --debug
  
  # Start with custom user name
  python chat_bot.py --user-name "Alice"
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--user-name',
        type=str,
        default='User',
        help='Name to use for the user in conversation (default: User)'
    )
    
    parser.add_argument(
        '--bot-name',
        type=str,
        default='ML-Master',
        help='Name to use for the bot in conversation (default: ML-Master)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='ML-Master Chat Bot 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Check for required environment variables
    if not config['reasoning'].get('subscription_key'):
        logger.error("AZURE_OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Create and start the bot
    try:
        bot = MLMasterBot(config)
        bot.user_name = args.user_name
        bot.bot_name = args.bot_name
        bot.start_conversation()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 