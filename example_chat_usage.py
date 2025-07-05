#!/usr/bin/env python3
"""
Example: Programmatic Usage of ML-Master Chat Bot

This example shows how to use the ML-Master conversational bot
programmatically for integration into other applications.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chat_bot import MLMasterBot, load_config, setup_logging


def example_basic_usage():
    """Example of basic bot usage"""
    print("=== Basic Bot Usage Example ===")
    
    # Setup
    setup_logging("INFO")
    config = load_config()
    
    # Create bot instance
    bot = MLMasterBot(config)
    bot.user_name = "Example User"
    bot.bot_name = "ML-Master"
    
    # Example interactions
    tasks = [
        "Write a Python function to calculate the factorial of a number",
        "Explain how binary search works",
        "Create a simple HTML form for user registration"
    ]
    
    for task in tasks:
        print(f"\nUser: {task}")
        
        # Process the task
        response = bot._process_user_input(task)
        
        # Display response
        bot._display_response(response)
        
        # Add to history
        bot._add_to_history(task, response)
    
    # Cleanup
    bot._cleanup()


def example_special_commands():
    """Example of using special commands"""
    print("\n=== Special Commands Example ===")
    
    # Setup
    setup_logging("INFO")
    config = load_config()
    
    # Create bot instance
    bot = MLMasterBot(config)
    
    # Special commands
    commands = [
        "help",
        "capabilities", 
        "stats",
        "history"
    ]
    
    for command in commands:
        print(f"\nUser: {command}")
        
        # Process the command
        response = bot._process_user_input(command)
        
        # Display response
        bot._display_response(response)
        
        # Add to history
        bot._add_to_history(command, response)
    
    # Cleanup
    bot._cleanup()


def example_generate_and_solve():
    """Example of using generate and solve commands"""
    print("\n=== Generate and Solve Commands Example ===")
    
    # Setup
    setup_logging("INFO")
    config = load_config()
    
    # Create bot instance
    bot = MLMasterBot(config)
    
    # Generate command (fast content generation)
    generate_task = "generate a Python function to sort a list of dictionaries by a specific key"
    print(f"\nUser: {generate_task}")
    response = bot._process_user_input(generate_task)
    bot._display_response(response)
    bot._add_to_history(generate_task, response)
    
    # Solve command (full exploration)
    solve_task = "solve this algorithm problem: implement a function to find all pairs of numbers in an array that sum to a target value"
    print(f"\nUser: {solve_task}")
    response = bot._process_user_input(solve_task)
    bot._display_response(response)
    bot._add_to_history(solve_task, response)
    
    # Cleanup
    bot._cleanup()


def example_custom_bot():
    """Example of creating a custom bot with different personality"""
    print("\n=== Custom Bot Example ===")
    
    # Setup
    setup_logging("INFO")
    config = load_config()
    
    # Create custom bot
    bot = MLMasterBot(config)
    bot.user_name = "Developer"
    bot.bot_name = "CodeMaster"
    
    # Customize capabilities
    bot.capabilities = {
        "code_review": "Review and improve code quality",
        "debugging": "Help debug and fix code issues",
        "optimization": "Optimize code performance",
        "architecture": "Design software architecture",
        "testing": "Create test cases and testing strategies"
    }
    
    # Test custom bot
    task = "Review this Python code and suggest improvements"
    print(f"\n{bot.user_name}: {task}")
    
    response = bot._process_user_input(task)
    bot._display_response(response)
    
    # Cleanup
    bot._cleanup()


def example_error_handling():
    """Example of error handling"""
    print("\n=== Error Handling Example ===")
    
    # Setup
    setup_logging("INFO")
    config = load_config()
    
    # Create bot instance
    bot = MLMasterBot(config)
    
    # Test error cases
    error_cases = [
        "",  # Empty input
        "generate",  # Incomplete command
        "solve",  # Incomplete command
        "invalid_command_that_does_not_exist"
    ]
    
    for case in error_cases:
        print(f"\nUser: '{case}'")
        
        response = bot._process_user_input(case)
        bot._display_response(response)
        bot._add_to_history(case, response)
    
    # Cleanup
    bot._cleanup()


def main():
    """Run all examples"""
    print("🤖 ML-Master Chat Bot - Programmatic Usage Examples")
    print("=" * 60)
    
    try:
        # Check if environment variables are set
        if not os.getenv('AZURE_OPENAI_API_KEY'):
            print("❌ Error: AZURE_OPENAI_API_KEY environment variable is required")
            print("Please set it before running the examples:")
            print("export AZURE_OPENAI_API_KEY='your-api-key'")
            return
        
        # Run examples
        example_basic_usage()
        example_special_commands()
        example_generate_and_solve()
        example_custom_bot()
        example_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n💡 Tips for integration:")
        print("  • Use bot._process_user_input() for single interactions")
        print("  • Use bot.start_conversation() for interactive mode")
        print("  • Customize bot.capabilities for different use cases")
        print("  • Handle responses based on response['type']")
        print("  • Always call bot._cleanup() when done")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure your environment variables are set correctly.")


if __name__ == "__main__":
    main() 