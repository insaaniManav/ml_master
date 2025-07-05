"""
Content Types for ML-Master Framework

This module defines the different content types that the ML-Master framework
can generate, including code, markdown, plain text, and structured data.
"""

from enum import Enum
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import json


class ContentType(Enum):
    """Supported content types for the framework"""
    CODE = "code"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED_DATA = "structured_data"
    MIXED = "mixed"


@dataclass
class ContentOutput:
    """Container for generated content with metadata"""
    content_type: ContentType
    content: str
    metadata: Dict[str, Any]
    format_hints: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'content_type': self.content_type.value,
            'content': self.content,
            'metadata': self.metadata,
            'format_hints': self.format_hints or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentOutput':
        """Create from dictionary representation"""
        return cls(
            content_type=ContentType(data['content_type']),
            content=data['content'],
            metadata=data['metadata'],
            format_hints=data.get('format_hints', {})
        )


class ContentFormatter:
    """Handles formatting of different content types"""
    
    @staticmethod
    def format_code(code: str, language: str = "python") -> str:
        """Format code with language-specific syntax highlighting"""
        return f"```{language}\n{code}\n```"
    
    @staticmethod
    def format_markdown(content: str) -> str:
        """Format markdown content"""
        return content
    
    @staticmethod
    def format_plain_text(content: str) -> str:
        """Format plain text content"""
        return content.strip()
    
    @staticmethod
    def format_structured_data(data: Dict[str, Any], format_type: str = "json") -> str:
        """Format structured data"""
        if format_type == "json":
            return json.dumps(data, indent=2)
        elif format_type == "yaml":
            import yaml
            return yaml.dump(data, default_flow_style=False)
        else:
            return str(data)
    
    @staticmethod
    def format_mixed_content(content_parts: list) -> str:
        """Format mixed content with multiple types"""
        formatted_parts = []
        
        for part in content_parts:
            if isinstance(part, dict):
                content_type = part.get('type', 'plain_text')
                content = part.get('content', '')
                
                if content_type == 'code':
                    language = part.get('language', 'python')
                    formatted_parts.append(ContentFormatter.format_code(content, language))
                elif content_type == 'markdown':
                    formatted_parts.append(ContentFormatter.format_markdown(content))
                elif content_type == 'structured_data':
                    format_type = part.get('format', 'json')
                    formatted_parts.append(ContentFormatter.format_structured_data(content, format_type))
                else:
                    formatted_parts.append(ContentFormatter.format_plain_text(content))
            else:
                formatted_parts.append(str(part))
        
        return "\n\n".join(formatted_parts)


class ContentValidator:
    """Validates different content types"""
    
    @staticmethod
    def validate_code(code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code content"""
        import ast
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Syntax error: {e}")
        
        return validation_result
    
    @staticmethod
    def validate_markdown(content: str) -> Dict[str, Any]:
        """Validate markdown content"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic markdown validation
        if not content.strip():
            validation_result['warnings'].append("Empty markdown content")
        
        return validation_result
    
    @staticmethod
    def validate_structured_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate structured data"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Invalid JSON: {e}")
        
        return validation_result


class ContentTypeDetector:
    """Detects content type from task description or user input"""
    
    @staticmethod
    def detect_from_task(task_description: str) -> ContentType:
        """Detect content type from task description"""
        task_lower = task_description.lower()
        
        # Code-related keywords
        code_keywords = [
            'code', 'program', 'script', 'function', 'class', 'algorithm',
            'implement', 'develop', 'create a program', 'write code',
            'python', 'javascript', 'java', 'c++', 'sql', 'html', 'css',
            'def ', 'import ', 'return ', 'if __name__'
        ]
        
        # Markdown-related keywords
        markdown_keywords = [
            'documentation', 'readme', 'guide', 'tutorial', 'explanation',
            'document', 'write up', 'report', 'analysis', 'summary',
            'markdown', 'md', 'documentation'
        ]
        
        # Structured data keywords
        structured_keywords = [
            'json', 'yaml', 'xml', 'csv', 'data structure', 'schema',
            'configuration', 'config', 'settings', 'parameters',
            'table', 'database', 'api response'
        ]
        
        # Plain text keywords (explanations, descriptions)
        plain_text_keywords = [
            'explain', 'describe', 'write about', 'tell me about',
            'what is', 'how does', 'why', 'detailed explanation',
            'comprehensive overview', 'introduction to'
        ]
        
        # Check for plain text keywords first (to avoid conflicts with code)
        if any(keyword in task_lower for keyword in plain_text_keywords):
            return ContentType.PLAIN_TEXT
        
        # Check for code keywords
        if any(keyword in task_lower for keyword in code_keywords):
            return ContentType.CODE
        
        # Check for markdown keywords
        if any(keyword in task_lower for keyword in markdown_keywords):
            return ContentType.MARKDOWN
        
        # Check for structured data keywords
        if any(keyword in task_lower for keyword in structured_keywords):
            return ContentType.STRUCTURED_DATA
        
        # Default to plain text
        return ContentType.PLAIN_TEXT
    
    @staticmethod
    def detect_from_content(content: str) -> ContentType:
        """Detect content type from actual content"""
        content_lower = content.lower()
        
        # Check for code patterns
        if any(pattern in content_lower for pattern in ['def ', 'class ', 'import ', 'function ', 'var ', 'const ']):
            return ContentType.CODE
        
        # Check for markdown patterns
        if any(pattern in content_lower for pattern in ['# ', '## ', '**', '*', '```', '[', '](']):
            return ContentType.MARKDOWN
        
        # Check for JSON/structured data patterns
        if content.strip().startswith('{') and content.strip().endswith('}'):
            try:
                json.loads(content)
                return ContentType.STRUCTURED_DATA
            except:
                pass
        
        # Default to plain text
        return ContentType.PLAIN_TEXT 