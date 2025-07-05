#!/usr/bin/env python3
"""
ML-Master Chat Bot Demo

This script demonstrates how to use the ML-Master conversational bot
with some example interactions.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat_bot import MLMasterBot, load_config, setup_logging


def demo_conversation():
    """Run a demo conversation with the bot"""
    
    # Setup basic logging
    setup_logging("INFO")
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Create bot instance
    bot = MLMasterBot(config)
    bot.user_name = "Demo User"
    bot.bot_name = "ML-Master"
    
    # Demo conversation examples
    demo_exchanges = [
        "Hello! Can you help me with programming?",
        "Write a Python function to calculate fibonacci numbers",
        "Can you explain how machine learning works?",
        "Generate a simple React component for a todo list",
        "Create a JSON configuration for a web server",
        "Write a README for a Python project called 'my-awesome-app'",
        "Help me understand neural networks",
        "Solve this problem: find the longest common subsequence of two strings",
        "stats",
        "capabilities",
        "exit"
    ]
    
    print("🤖 ML-Master Chat Bot Demo")
    print("=" * 50)
    print("This demo will show various interactions with the bot.")
    print("You can also run 'python chat_bot.py' for interactive mode.\n")
    
    # Simulate conversation
    for i, user_input in enumerate(demo_exchanges, 1):
        print(f"\n--- Exchange {i} ---")
        print(f"Demo User: {user_input}")
        
        # Process the input
        response = bot._process_user_input(user_input)
        
        # Display response
        bot._display_response(response)
        
        # Add to history
        bot._add_to_history(user_input, response)
        
        # Small delay for readability
        time.sleep(1)
        
        # Check if we should exit
        if user_input.lower() == 'exit':
            break
    
    print(f"\n{'='*50}")
    print("Demo completed! Run 'python chat_bot.py' for interactive mode.")
    
    # Cleanup
    bot._cleanup()


def demo_quick_start():
    """Show quick start instructions"""
    print("🚀 ML-Master Chat Bot Quick Start")
    print("=" * 50)
    
    print("\n1. Set up your environment variables:")
    print("   export AZURE_OPENAI_API_KEY='your-api-key'")
    print("   export AZURE_OPENAI_ENDPOINT='your-endpoint'")
    print("   export AZURE_OPENAI_MODEL='your-model-name'")
    
    print("\n2. Start the interactive bot:")
    print("   python chat_bot.py")
    
    print("\n3. Or run with custom settings:")
    print("   python chat_bot.py --user-name 'Alice' --debug")
    
    print("\n4. Example interactions:")
    print("   • 'Write a Python function to sort a list'")
    print("   • 'Explain how neural networks work'")
    print("   • 'Generate a React component for a todo list'")
    print("   • 'Create a README for my project'")
    print("   • 'help' - Show all capabilities")
    print("   • 'stats' - Show framework statistics")
    print("   • 'exit' - End conversation")
    
    print("\n5. Special commands:")
    print("   • 'generate [task]' - Fast content generation")
    print("   • 'solve [problem]' - Full problem solving with exploration")
    print("   • 'history' - Show conversation history")
    print("   • 'capabilities' - Show bot capabilities")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML-Master Chat Bot Demo")
    parser.add_argument(
        '--quick-start',
        action='store_true',
        help='Show quick start instructions instead of running demo'
    )
    
    args = parser.parse_args()
    
    if args.quick_start:
        demo_quick_start()
    else:
        demo_conversation() 