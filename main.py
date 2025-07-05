#!/usr/bin/env python3
"""
ML-Master: AI-for-AI Framework

Main entry point for the ML-Master framework that provides
a command-line interface for solving various AI tasks and generating
different types of content (code, markdown, plain text, structured data).
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from loguru import logger

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.integration import MLMaster
from core.integration.content_types import ContentType, ContentTypeDetector


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
        log_dir / "ml_master.log",
        level=log_level,
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


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


def solve_task(config: dict, task_description: str, content_type: str = None, max_time: int = None, max_iterations: int = None):
    """Solve a task using ML-Master"""
    logger.info("Initializing ML-Master framework...")
    
    # Initialize ML-Master
    ml_master = MLMaster(config)
    
    try:
        # Convert content type string to enum
        content_type_enum = None
        if content_type:
            try:
                content_type_enum = ContentType(content_type.lower())
            except ValueError:
                logger.warning(f"Invalid content type '{content_type}', auto-detecting...")
                content_type_enum = None
        
        # Solve the task
        logger.info("Starting task solution...")
        results = ml_master.solve_task(
            task_description=task_description,
            content_type=content_type_enum,
            max_time=max_time,
            max_iterations=max_iterations
        )
        
        # Display results
        display_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during task execution: {e}")
        raise
    
    finally:
        # Cleanup
        ml_master.cleanup()


def generate_content(config: dict, task_description: str, content_type: str = None, format_output: bool = True):
    """Generate content using ML-Master"""
    logger.info("Initializing ML-Master framework...")
    
    # Initialize ML-Master
    ml_master = MLMaster(config)
    
    try:
        # Convert content type string to enum
        content_type_enum = None
        if content_type:
            try:
                content_type_enum = ContentType(content_type.lower())
            except ValueError:
                logger.warning(f"Invalid content type '{content_type}', auto-detecting...")
                content_type_enum = None
        
        # Generate content
        logger.info("Generating content...")
        content_output = ml_master.generate_content(
            task_description=task_description,
            content_type=content_type_enum,
            format_output=format_output
        )
        
        # Display content
        display_content(content_output)
        
        return content_output
        
    except Exception as e:
        logger.error(f"Error during content generation: {e}")
        raise
    
    finally:
        # Cleanup
        ml_master.cleanup()


def display_results(results: dict):
    """Display task results in a formatted way"""
    print("\n" + "="*80)
    print("ML-MASTER TASK RESULTS")
    print("="*80)
    
    # Task info
    print(f"\n📋 Task Description:")
    print(f"   {results['task'][:200]}...")
    
    # Content type
    print(f"\n📝 Content Type:")
    print(f"   {results['content_type']}")
    
    # Solution info
    print(f"\n🎯 Solution:")
    print(f"   Performance Score: {results['solution']['performance_score']:.4f}")
    print(f"   Node ID: {results['solution']['node_id']}")
    
    # Performance metrics
    print(f"\n⚡ Performance Metrics:")
    print(f"   Iterations Completed: {results['performance']['iterations_completed']}")
    print(f"   Total Time: {results['performance']['total_time']:.2f}s")
    print(f"   Iterations/Second: {results['performance']['iterations_per_second']:.2f}")
    
    # Exploration stats
    print(f"\n🌳 Exploration Statistics:")
    print(f"   Total Nodes: {results['exploration']['total_nodes']}")
    print(f"   Expanded Nodes: {results['exploration']['expanded_nodes']}")
    print(f"   Terminal Nodes: {results['exploration']['terminal_nodes']}")
    print(f"   Average Depth: {results['exploration']['average_depth']:.2f}")
    print(f"   Tree Height: {results['exploration']['tree_height']}")
    
    # Memory usage
    print(f"\n🧠 Memory Usage:")
    print(f"   Total Entries: {results['memory']['total_entries']}")
    print(f"   Memory Usage: {results['memory']['memory_usage_percent']:.1f}%")
    
    # Framework info
    print(f"\n🔧 Framework Info:")
    print(f"   Version: {results['framework_info']['version']}")
    print(f"   Model Used: {results['framework_info']['model_used']}")
    print(f"   Parallel Workers: {results['framework_info']['parallel_workers']}")
    print(f"   Content Type: {results['framework_info']['content_type']}")
    
    # Generated content
    if results['solution']['content']:
        print(f"\n💻 Generated Content:")
        print("-" * 60)
        print(results['solution']['content'])
        print("-" * 60)
    else:
        print(f"\n💻 Generated Content: No content generated")


def display_content(content_output):
    """Display generated content in a formatted way"""
    print("\n" + "="*80)
    print("ML-MASTER CONTENT GENERATION")
    print("="*80)
    
    # Task info
    print(f"\n📋 Task Description:")
    print(f"   {content_output.metadata['task_description'][:200]}...")
    
    # Content type
    print(f"\n📝 Content Type:")
    print(f"   {content_output.content_type.value}")
    
    # Generation time
    if 'generation_time' in content_output.metadata:
        print(f"\n⏱️  Generation Time:")
        print(f"   {content_output.metadata['generation_time']:.2f}s")
    
    # Generated content
    print(f"\n💻 Generated Content:")
    print("-" * 60)
    print(content_output.content)
    print("-" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ML-Master: AI-for-AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a task with auto-detected content type
  python main.py --task "Create a binary classification model for iris dataset"
  
  # Generate markdown documentation
  python main.py --generate --task "Write a README for a Python project" --content-type markdown
  
  # Generate structured data
  python main.py --generate --task "Create a JSON configuration for a web server" --content-type structured_data
  
  # Solve with custom time and iteration limits
  python main.py --task "Implement a neural network for MNIST" --max-time 600 --max-iterations 100
  
  # Use custom config file
  python main.py --config my_config.yaml --task "Your task description"
  
  # Run in debug mode
  python main.py --debug --task "Your task description"
  
Content Types:
  - code: Generate executable code (default)
  - markdown: Generate markdown documentation
  - plain_text: Generate plain text content
  - structured_data: Generate JSON/YAML data
        """
    )
    
    parser.add_argument(
        '--task', '-t',
        type=str,
        required=True,
        help='Description of the task to solve or content to generate'
    )
    
    parser.add_argument(
        '--generate', '-g',
        action='store_true',
        help='Generate content directly (faster, no exploration)'
    )
    
    parser.add_argument(
        '--content-type', '-c',
        type=str,
        choices=['code', 'markdown', 'plain_text', 'structured_data'],
        help='Type of content to generate (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--max-time', '-T',
        type=int,
        help='Maximum time to spend on the task (in seconds)'
    )
    
    parser.add_argument(
        '--max-iterations', '-i',
        type=int,
        help='Maximum number of MCTS iterations'
    )
    
    parser.add_argument(
        '--no-format',
        action='store_true',
        help='Disable output formatting'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='ML-Master 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.max_time:
        config['exploration']['max_time'] = args.max_time
    
    if args.max_iterations:
        config['exploration']['max_iterations'] = args.max_iterations
    
    # Check for required environment variables
    if not config['reasoning'].get('subscription_key'):
        logger.error("AZURE_OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Execute based on mode
    try:
        if args.generate:
            # Content generation mode
            results = generate_content(
                config=config,
                task_description=args.task,
                content_type=args.content_type,
                format_output=not args.no_format
            )
        else:
            # Full task solving mode
            results = solve_task(
                config=config,
                task_description=args.task,
                content_type=args.content_type,
                max_time=args.max_time,
                max_iterations=args.max_iterations
            )
        
        logger.info("Task completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Task interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Task failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 