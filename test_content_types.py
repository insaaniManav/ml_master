#!/usr/bin/env python3
"""
Test script for content types functionality
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.integration.content_types import ContentType, ContentTypeDetector, ContentFormatter, ContentValidator


def test_content_type_detection():
    """Test content type detection from task descriptions"""
    print("Testing Content Type Detection...")
    
    test_cases = [
        ("Create a Python function to process data", ContentType.CODE),
        ("Write a README for the project", ContentType.MARKDOWN),
        ("Explain how machine learning works", ContentType.PLAIN_TEXT),
        ("Generate a JSON configuration file", ContentType.STRUCTURED_DATA),
        ("Implement a neural network", ContentType.CODE),
        ("Document the API endpoints", ContentType.MARKDOWN),
        ("Write a detailed explanation of the algorithm", ContentType.PLAIN_TEXT),
        ("Create a YAML config", ContentType.STRUCTURED_DATA),
    ]
    
    for task, expected in test_cases:
        detected = ContentTypeDetector.detect_from_task(task)
        status = "✅" if detected == expected else "❌"
        print(f"{status} Task: '{task}' -> Detected: {detected.value}, Expected: {expected.value}")


def test_content_formatting():
    """Test content formatting"""
    print("\nTesting Content Formatting...")
    
    # Test code formatting
    code = "def hello(): print('world')"
    formatted_code = ContentFormatter.format_code(code, "python")
    print(f"Code formatting: {formatted_code[:50]}...")
    
    # Test markdown formatting
    markdown = "# Title\nThis is content"
    formatted_markdown = ContentFormatter.format_markdown(markdown)
    print(f"Markdown formatting: {formatted_markdown}")
    
    # Test plain text formatting
    text = "  This is some text  "
    formatted_text = ContentFormatter.format_plain_text(text)
    print(f"Plain text formatting: '{formatted_text}'")
    
    # Test structured data formatting
    data = {"key": "value", "number": 42}
    formatted_data = ContentFormatter.format_structured_data(data)
    print(f"Structured data formatting: {formatted_data[:50]}...")


def test_content_validation():
    """Test content validation"""
    print("\nTesting Content Validation...")
    
    # Test valid Python code
    valid_code = "def test(): return True"
    validation = ContentValidator.validate_code(valid_code, "python")
    print(f"Valid Python code: {validation['is_valid']}")
    
    # Test invalid Python code
    invalid_code = "def test(: return True"  # Missing closing parenthesis
    validation = ContentValidator.validate_code(invalid_code, "python")
    print(f"Invalid Python code: {validation['is_valid']} - {validation['errors']}")
    
    # Test markdown validation
    markdown = "# Title\nContent here"
    validation = ContentValidator.validate_markdown(markdown)
    print(f"Markdown validation: {validation['is_valid']}")
    
    # Test structured data validation
    valid_json = '{"key": "value"}'
    try:
        data = eval(valid_json)  # Simple way to parse JSON-like string
        validation = ContentValidator.validate_structured_data(data)
        print(f"Valid structured data: {validation['is_valid']}")
    except:
        print("Valid structured data: Error in test")


def test_content_type_enum():
    """Test content type enum"""
    print("\nTesting Content Type Enum...")
    
    # Test enum values
    print(f"Available content types: {[ct.value for ct in ContentType]}")
    
    # Test enum creation
    code_type = ContentType("code")
    markdown_type = ContentType("markdown")
    print(f"Code type: {code_type.value}")
    print(f"Markdown type: {markdown_type.value}")


def main():
    """Main test function"""
    print("ML-Master Content Types Test")
    print("="*50)
    
    try:
        test_content_type_enum()
        test_content_type_detection()
        test_content_formatting()
        test_content_validation()
        
        print("\n" + "="*50)
        print("All tests completed!")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 