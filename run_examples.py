#!/usr/bin/env python3
"""
ML-Master Examples Runner

This script provides easy access to run different ML-Master examples.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def show_menu():
    """Show the examples menu"""
    print("🚀 ML-MASTER EXAMPLES")
    print("="*50)
    print("Choose an example to run:")
    print()
    print("1. 📖 Flow Explanation")
    print("   - Detailed explanation of ML-Master flow")
    print("   - Component interaction demonstration")
    print("   - Step-by-step process walkthrough")
    print()
    print("2. 🎯 Simple Example")
    print("   - Basic ML-Master usage")
    print("   - Quick demonstration of capabilities")
    print("   - Minimal configuration")
    print()
    print("3. 🔍 Comprehensive Example")
    print("   - Full framework demonstration")
    print("   - All components showcased")
    print("   - Detailed statistics and analysis")
    print()
    print("4. 📊 All Examples")
    print("   - Run all examples in sequence")
    print("   - Complete framework overview")
    print()
    print("0. ❌ Exit")
    print()


def run_flow_example():
    """Run the flow explanation example"""
    print("\n" + "="*60)
    print("📖 RUNNING FLOW EXPLANATION EXAMPLE")
    print("="*60)
    
    try:
        from examples.flow_example import explain_ml_master_flow, demonstrate_component_interaction, run_flow_example
        
        # Run the flow explanation
        explain_ml_master_flow()
        demonstrate_component_interaction()
        run_flow_example()
        
    except ImportError as e:
        print(f"❌ Error importing flow example: {e}")
    except Exception as e:
        print(f"❌ Error running flow example: {e}")


def run_simple_example():
    """Run the simple example"""
    print("\n" + "="*60)
    print("🎯 RUNNING SIMPLE EXAMPLE")
    print("="*60)
    
    try:
        from examples.simple_example import run_simple_example as run_simple
        
        run_simple()
        
    except ImportError as e:
        print(f"❌ Error importing simple example: {e}")
    except Exception as e:
        print(f"❌ Error running simple example: {e}")


def run_comprehensive_example():
    """Run the comprehensive example"""
    print("\n" + "="*60)
    print("🔍 RUNNING COMPREHENSIVE EXAMPLE")
    print("="*60)
    
    try:
        from examples.comprehensive_example import run_comprehensive_example as run_comprehensive
        
        run_comprehensive()
        
    except ImportError as e:
        print(f"❌ Error importing comprehensive example: {e}")
    except Exception as e:
        print(f"❌ Error running comprehensive example: {e}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "="*60)
    print("📊 RUNNING ALL EXAMPLES")
    print("="*60)
    
    examples = [
        ("Flow Explanation", run_flow_example),
        ("Simple Example", run_simple_example),
        ("Comprehensive Example", run_comprehensive_example)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*20} Example {i}: {name} {'='*20}")
        try:
            func()
            print(f"✅ {name} completed successfully!")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        
        if i < len(examples):
            print("\n" + "="*60)
            input("Press Enter to continue to the next example...")


def main():
    """Main function"""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                run_flow_example()
            elif choice == "2":
                run_simple_example()
            elif choice == "3":
                run_comprehensive_example()
            elif choice == "4":
                run_all_examples()
            else:
                print("\n❌ Invalid choice. Please enter a number between 0 and 4.")
                continue
            
            if choice != "4":  # Don't ask for continuation if running all examples
                print("\n" + "="*60)
                input("Press Enter to return to menu...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main() 