"""
Steerable Reasoning Implementation

This module implements the steerable reasoning system that enhances
LLM reasoning capabilities with adaptive memory integration.
"""

import os
import time
import json
import httpx
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from openai import AzureOpenAI
from loguru import logger

from ..memory import AdaptiveMemory, MemoryType
from ..integration.content_types import ContentType, ContentOutput, ContentFormatter, ContentValidator


@dataclass
class ReasoningContext:
    """Context for reasoning process"""
    task_description: str
    current_action: str
    content_type: ContentType
    parent_insights: List[Dict[str, Any]]
    sibling_insights: List[Dict[str, Any]]
    execution_feedback: List[Dict[str, Any]]
    performance_metrics: List[Dict[str, Any]]
    current_content: str = ""
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class SteerableReasoner:
    """
    Steerable Reasoning System
    
    Implements the steerable reasoning component of ML-Master that enhances
    LLM reasoning capabilities by embedding adaptive memory directly into
    the reasoning process.
    """
    
    def __init__(self, config: Dict[str, Any], memory: AdaptiveMemory):
        """
        Initialize the steerable reasoner
        
        Args:
            config: Configuration dictionary
            memory: Adaptive memory instance
        """
        self.config = config
        self.memory = memory
        
        # Azure OpenAI configuration
        self.endpoint = config.get('azure_endpoint', "")
        self.model_name = config.get('model_name', '')
        self.deployment = config.get('deployment', '')
        self.subscription_key = config.get('subscription_key', os.getenv('AZURE_OPENAI_API_KEY'))
        self.api_version = config.get('api_version', '')
        
        # Reasoning settings
        self.max_context_length = config.get('max_context_length', 8000)
        self.context_compression = config.get('context_compression', True)
        self.sibling_node_inclusion = config.get('sibling_node_inclusion', True)
        self.enable_chain_of_thought = config.get('enable_chain_of_thought', True)
        self.enable_self_verification = config.get('enable_self_verification', True)
        self.reasoning_depth = config.get('reasoning_depth', 3)
        
        # Content type settings
        self.default_content_type = ContentType.CODE  # Default for backward compatibility
        self.enable_content_type_detection = config.get('enable_content_type_detection', True)
        
        # Create httpx client with SSL verification disabled
        self.http_client = httpx.Client(
            verify=False,  # Disable SSL verification
            timeout=60.0
        )
        
        # Initialize Azure OpenAI client with custom httpx client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            http_client=self.http_client
        )
        
        logger.info(f"Initialized SteerableReasoner with Azure OpenAI {self.model_name} (SSL verification disabled)")
    
    def reason(self, 
               task_description: str,
               action_type: str,
               content_type: Optional[ContentType] = None,
               current_content: str = "",
               error_messages: List[str] = None,
               contextual_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform steerable reasoning with adaptive memory integration
        
        Args:
            task_description: Description of the task
            action_type: Type of action (draft, debug, improve)
            content_type: Type of content to generate
            current_content: Current content if any
            error_messages: Error messages if any
            contextual_memory: Contextual memory from exploration
            
        Returns:
            Reasoning results including plan, content, and insights
        """
        # Detect content type if not specified
        if content_type is None and self.enable_content_type_detection:
            from ..integration.content_types import ContentTypeDetector
            content_type = ContentTypeDetector.detect_from_task(task_description)
        
        # Use default if still None
        if content_type is None:
            content_type = self.default_content_type
        
        # Build reasoning context
        context = self._build_reasoning_context(
            task_description=task_description,
            action_type=action_type,
            content_type=content_type,
            current_content=current_content,
            error_messages=error_messages or [],
            contextual_memory=contextual_memory or {}
        )
        
        # Generate reasoning prompt
        prompt = self._generate_reasoning_prompt(context)
        
        # Perform reasoning with LLM
        reasoning_result = self._perform_llm_reasoning(prompt, context)
        
        # Extract insights and generate content
        insights = self._extract_insights(reasoning_result)
        content = self._generate_content(reasoning_result, context)
        
        # Self-verification if enabled
        if self.enable_self_verification:
            verification_result = self._self_verify(content, context)
            insights['verification'] = verification_result
        
        return {
            'reasoning_process': reasoning_result,
            'insights': insights,
            'content': content,
            'content_type': content_type.value,
            'action_type': action_type,
            'context_used': self._summarize_context_usage(context)
        }
    
    def _build_reasoning_context(self,
                                task_description: str,
                                action_type: str,
                                content_type: ContentType,
                                current_content: str,
                                error_messages: List[str],
                                contextual_memory: Dict[str, Any]) -> ReasoningContext:
        """Build comprehensive reasoning context"""
        
        # Extract memory components
        parent_insights = contextual_memory.get('parent_insights', [])
        sibling_insights = contextual_memory.get('sibling_insights', [])
        execution_feedback = contextual_memory.get('execution_feedback', [])
        performance_metrics = contextual_memory.get('performance_metrics', [])
        
        # Apply context compression if enabled
        if self.context_compression:
            parent_insights = self._compress_insights(parent_insights)
            sibling_insights = self._compress_insights(sibling_insights)
            execution_feedback = self._compress_feedback(execution_feedback)
            performance_metrics = self._compress_metrics(performance_metrics)
        
        return ReasoningContext(
            task_description=task_description,
            current_action=action_type,
            content_type=content_type,
            parent_insights=parent_insights,
            sibling_insights=sibling_insights,
            execution_feedback=execution_feedback,
            performance_metrics=performance_metrics,
            current_content=current_content,
            error_messages=error_messages
        )
    
    def _generate_reasoning_prompt(self, context: ReasoningContext) -> str:
        """Generate comprehensive reasoning prompt with memory integration"""
        
        prompt_parts = []
        
        # System instruction
        system_instruction = self._get_system_instruction(context.current_action, context.content_type)
        prompt_parts.append(f"System: {system_instruction}")
        
        # Task description
        prompt_parts.append(f"Task: {context.task_description}")
        prompt_parts.append(f"Content Type: {context.content_type.value}")
        
        # Current state
        if context.current_content:
            if context.content_type == ContentType.CODE:
                prompt_parts.append(f"Current Code:\n```python\n{context.current_content}\n```")
            else:
                prompt_parts.append(f"Current Content:\n{context.current_content}")
        
        if context.error_messages:
            prompt_parts.append(f"Errors to Fix:\n" + "\n".join([f"- {error}" for error in context.error_messages]))
        
        # Memory integration - Parent insights for continuity
        if context.parent_insights:
            parent_summary = self._summarize_insights(context.parent_insights, "parent")
            prompt_parts.append(f"Previous Reasoning Insights:\n{parent_summary}")
        
        # Memory integration - Sibling insights for diversity
        if context.sibling_insights and self.sibling_node_inclusion:
            sibling_summary = self._summarize_insights(context.sibling_insights, "sibling")
            prompt_parts.append(f"Alternative Approaches:\n{sibling_summary}")
        
        # Memory integration - Execution feedback
        if context.execution_feedback:
            feedback_summary = self._summarize_feedback(context.execution_feedback)
            prompt_parts.append(f"Execution Feedback:\n{feedback_summary}")
        
        # Memory integration - Performance metrics
        if context.performance_metrics:
            metrics_summary = self._summarize_metrics(context.performance_metrics)
            prompt_parts.append(f"Performance Metrics:\n{metrics_summary}")
        
        # Reasoning instruction
        reasoning_instruction = self._get_reasoning_instruction(context.current_action, context.content_type)
        prompt_parts.append(f"Instructions: {reasoning_instruction}")
        
        return "\n\n".join(prompt_parts)
    
    def _get_system_instruction(self, action_type: str, content_type: ContentType) -> str:
        """Get system instruction based on action type and content type"""
        base_instruction = f"You are an AI assistant specialized in {content_type.value} generation. "
        
        if content_type == ContentType.CODE:
            base_instruction += "You generate high-quality, executable code with proper documentation and error handling."
        elif content_type == ContentType.MARKDOWN:
            base_instruction += "You generate well-structured markdown documents with clear formatting and organization."
        elif content_type == ContentType.PLAIN_TEXT:
            base_instruction += "You generate clear, concise, and well-organized plain text content."
        elif content_type == ContentType.STRUCTURED_DATA:
            base_instruction += "You generate properly formatted structured data (JSON, YAML, etc.) with clear schemas."
        
        if action_type == "draft":
            base_instruction += " Focus on creating a complete initial solution."
        elif action_type == "debug":
            base_instruction += " Focus on identifying and fixing issues in the current content."
        elif action_type == "improve":
            base_instruction += " Focus on enhancing and optimizing the current content."
        
        return base_instruction
    
    def _get_reasoning_instruction(self, action_type: str, content_type: ContentType) -> str:
        """Get reasoning instruction based on action type and content type"""
        if content_type == ContentType.CODE:
            if action_type == "draft":
                return "Generate complete, executable code that solves the task. Include proper imports, error handling, and documentation."
            elif action_type == "debug":
                return "Analyze the code for errors and issues. Provide corrected code with explanations of the fixes."
            elif action_type == "improve":
                return "Enhance the code for better performance, readability, and maintainability. Add optimizations and improvements."
        elif content_type == ContentType.MARKDOWN:
            if action_type == "draft":
                return "Create a comprehensive markdown document with proper headings, formatting, and structure."
            elif action_type == "debug":
                return "Review the markdown for formatting issues, broken links, or unclear content. Provide corrected version."
            elif action_type == "improve":
                return "Enhance the markdown with better organization, additional details, and improved formatting."
        elif content_type == ContentType.PLAIN_TEXT:
            if action_type == "draft":
                return "Write clear, well-structured plain text content that addresses the task requirements."
            elif action_type == "debug":
                return "Review the text for clarity, grammar, and logical flow. Provide corrected version."
            elif action_type == "improve":
                return "Enhance the text with better organization, additional details, and improved clarity."
        elif content_type == ContentType.STRUCTURED_DATA:
            if action_type == "draft":
                return "Create properly structured data (JSON/YAML) with clear schema and appropriate data types."
            elif action_type == "debug":
                return "Validate and fix any structural or data type issues in the structured data."
            elif action_type == "improve":
                return "Enhance the structured data with additional fields, better organization, and improved schema."
        
        return f"Generate {content_type.value} content for the {action_type} action."
    
    def _perform_llm_reasoning(self, prompt: str, context: ReasoningContext) -> str:
        """Perform reasoning using Azure OpenAI"""
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert AI-for-AI agent. Provide detailed step-by-step reasoning before generating code."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                max_completion_tokens=100000,
                model=self.deployment
            )
            
            reasoning_result = response.choices[0].message.content
            logger.debug(f"Generated reasoning for {context.current_action} action")
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            return f"Error in reasoning: {str(e)}"
    
    def _extract_insights(self, reasoning_result: str) -> Dict[str, Any]:
        """Extract key insights from reasoning process"""
        insights = {
            'reasoning_steps': [],
            'key_decisions': [],
            'assumptions': [],
            'potential_improvements': []
        }
        
        # Simple insight extraction (can be enhanced with more sophisticated parsing)
        lines = reasoning_result.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify reasoning sections
            if 'step' in line.lower() or 'reasoning' in line.lower():
                insights['reasoning_steps'].append(line)
            elif 'decision' in line.lower() or 'choose' in line.lower():
                insights['key_decisions'].append(line)
            elif 'assume' in line.lower() or 'assumption' in line.lower():
                insights['assumptions'].append(line)
            elif 'improve' in line.lower() or 'optimize' in line.lower():
                insights['potential_improvements'].append(line)
        
        return insights
    
    def _generate_content(self, reasoning_result: str, context: ReasoningContext) -> str:
        """Extract or generate content from reasoning result"""
        
        # Look for content blocks in reasoning result
        content_blocks = []
        lines = reasoning_result.split('\n')
        in_content_block = False
        current_block = []
        
        for line in lines:
            if '```' in line:
                if in_content_block:
                    # End of content block
                    content_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_content_block = False
                else:
                    # Start of content block
                    in_content_block = True
            elif in_content_block:
                current_block.append(line)
        
        # If no content blocks found, extract content based on type
        if not content_blocks:
            if context.content_type == ContentType.CODE:
                # For code, look for code-like content
                content_lines = []
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['import ', 'def ', 'class ', 'return ', 'if __name__', 'function ', 'var ', 'const ']):
                        content_lines.append(line)
                
                if content_lines:
                    content_blocks.append('\n'.join(content_lines))
            
            elif context.content_type == ContentType.MARKDOWN:
                # For markdown, look for markdown-like content
                content_lines = []
                for line in lines:
                    if any(keyword in line for keyword in ['# ', '## ', '**', '*', '[', '](']):
                        content_lines.append(line)
                    elif line.strip() and not line.startswith('System:') and not line.startswith('Task:') and not line.startswith('Instructions:'):
                        content_lines.append(line)
                
                if content_lines:
                    content_blocks.append('\n'.join(content_lines))
            
            elif context.content_type == ContentType.PLAIN_TEXT:
                # For plain text, extract all non-system content
                content_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('System:') and not line.startswith('Task:') and not line.startswith('Instructions:') and not line.startswith('Content Type:'):
                        content_lines.append(line)
                
                if content_lines:
                    content_blocks.append('\n'.join(content_lines))
            
            elif context.content_type == ContentType.STRUCTURED_DATA:
                # For structured data, look for JSON/YAML content
                content_lines = []
                for line in lines:
                    if any(keyword in line for keyword in ['{', '}', '[', ']', ':', 'yaml', 'json']):
                        content_lines.append(line)
                    elif line.strip() and not line.startswith('System:') and not line.startswith('Task:') and not line.startswith('Instructions:'):
                        content_lines.append(line)
                
                if content_lines:
                    content_blocks.append('\n'.join(content_lines))
        
        # If still no content blocks, return the full reasoning result (cleaned)
        if not content_blocks:
            # Remove system instructions and return the rest
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('System:') and not line.startswith('Task:') and not line.startswith('Instructions:') and not line.startswith('Content Type:'):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                content_blocks.append('\n'.join(cleaned_lines))
        
        return '\n\n'.join(content_blocks) if content_blocks else ""
    
    def _self_verify(self, content: str, context: ReasoningContext) -> Dict[str, Any]:
        """Perform self-verification of generated content"""
        
        verification_prompt = f"""
Verify the following content for the task: {context.task_description}

Content:
```
{content}
```

Check for:
1. Syntax errors
2. Logical errors
3. Missing imports
4. Incomplete implementations
5. Potential improvements

Provide a detailed verification report.
"""
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a content verification expert. Analyze content for errors and improvements."
                },
                {
                    "role": "user",
                    "content": verification_prompt
                }
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                max_completion_tokens=5000,
                model=self.deployment
            )
            
            verification_result = response.choices[0].message.content
            
            return {
                'verification_report': verification_result,
                'has_errors': 'error' in verification_result.lower(),
                'suggestions': verification_result
            }
            
        except Exception as e:
            logger.error(f"Error in self-verification: {e}")
            return {
                'verification_report': f"Verification failed: {str(e)}",
                'has_errors': True,
                'suggestions': []
            }
    
    def _compress_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress insights to reduce context length"""
        if len(insights) <= 5:  # Keep all if small
            return insights
        
        # Sort by relevance and keep top insights
        sorted_insights = sorted(insights, key=lambda x: x.get('relevance_score', 0), reverse=True)
        return sorted_insights[:5]
    
    def _compress_feedback(self, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress execution feedback"""
        if len(feedback) <= 3:
            return feedback
        
        # Keep most recent and most relevant feedback
        sorted_feedback = sorted(feedback, key=lambda x: x.get('timestamp', 0), reverse=True)
        return sorted_feedback[:3]
    
    def _compress_metrics(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress performance metrics"""
        if len(metrics) <= 3:
            return metrics
        
        # Keep best performing metrics
        sorted_metrics = sorted(metrics, key=lambda x: x.get('content', {}).get('performance_score', 0), reverse=True)
        return sorted_metrics[:3]
    
    def _summarize_insights(self, insights: List[Dict[str, Any]], source: str) -> str:
        """Summarize insights for prompt inclusion"""
        if not insights:
            return ""
        
        summary_parts = []
        for i, insight in enumerate(insights[:3]):  # Limit to top 3
            content = insight.get('content', {})
            relevance = insight.get('relevance_score', 0)
            
            if 'reasoning_process' in content:
                summary_parts.append(f"{source.capitalize()} Insight {i+1} (relevance: {relevance:.2f}): {content['reasoning_process'][:200]}...")
        
        return "\n".join(summary_parts)
    
    def _summarize_feedback(self, feedback: List[Dict[str, Any]]) -> str:
        """Summarize execution feedback"""
        if not feedback:
            return ""
        
        summary_parts = []
        for i, item in enumerate(feedback[:2]):  # Limit to top 2
            content = item.get('content', {})
            logs = content.get('execution_logs', [])
            errors = content.get('error_messages', [])
            
            if errors:
                summary_parts.append(f"Recent Error {i+1}: {errors[0] if errors else 'Unknown error'}")
            elif logs:
                summary_parts.append(f"Recent Log {i+1}: {logs[-1] if logs else 'No logs'}")
        
        return "\n".join(summary_parts)
    
    def _summarize_metrics(self, metrics: List[Dict[str, Any]]) -> str:
        """Summarize performance metrics"""
        if not metrics:
            return ""
        
        summary_parts = []
        for i, item in enumerate(metrics[:2]):  # Limit to top 2
            content = item.get('content', {})
            score = content.get('performance_score', 0)
            improvement = content.get('improvement', 0)
            
            summary_parts.append(f"Performance {i+1}: Score={score:.4f}, Improvement={improvement:.4f}")
        
        return "\n".join(summary_parts)
    
    def _summarize_context_usage(self, context: ReasoningContext) -> Dict[str, Any]:
        """Summarize how context was used in reasoning"""
        return {
            'parent_insights_used': len(context.parent_insights),
            'sibling_insights_used': len(context.sibling_insights),
            'execution_feedback_used': len(context.execution_feedback),
            'performance_metrics_used': len(context.performance_metrics),
            'context_compression_applied': self.context_compression,
            'sibling_inclusion_enabled': self.sibling_node_inclusion
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'http_client'):
            self.http_client.close()
            logger.debug("Closed httpx client") 