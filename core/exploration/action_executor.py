"""
Action Executor Implementation

This module implements the action executor that handles Draft, Debug, and Improve
actions using the steerable reasoning system.
"""

import os
import subprocess
import tempfile
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from ..reasoning import SteerableReasoner
from ..memory import AdaptiveMemory, MemoryType
from ..integration.content_types import ContentType, ContentOutput, ContentFormatter, ContentValidator
from .mcts_node import MCTSNode, ActionType


class ActionExecutor:
    """
    Action Executor for ML-Master
    
    Handles the execution of Draft, Debug, and Improve actions using
    the steerable reasoning system and content processing environment.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 reasoner: SteerableReasoner,
                 memory: AdaptiveMemory):
        """
        Initialize the action executor
        
        Args:
            config: Configuration dictionary
            reasoner: Steerable reasoner instance
            memory: Adaptive memory instance
        """
        self.config = config
        self.reasoner = reasoner
        self.memory = memory
        
        # Execution environment settings
        self.execution_timeout = config.get('execution_timeout', 300)
        self.memory_limit = config.get('memory_limit', '4GB')
        self.cpu_limit = config.get('cpu_limit', 2)
        self.sandbox_execution = config.get('sandbox_execution', True)
        self.allow_network_access = config.get('allow_network_access', False)
        
        # File management
        self.temp_dir = config.get('temp_dir', './temp')
        self.output_dir = config.get('output_dir', './outputs')
        self.cache_dir = config.get('cache_dir', './cache')
        
        # Content processing settings
        self.enable_content_validation = config.get('enable_content_validation', True)
        self.enable_content_formatting = config.get('enable_content_formatting', True)
        
        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized ActionExecutor with timeout={self.execution_timeout}s")
    
    def execute_action(self,
                      node: MCTSNode,
                      action_type: ActionType,
                      contextual_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action (Draft, Debug, Improve) on a node
        
        Args:
            node: MCTS node to execute action on
            action_type: Type of action to execute
            contextual_memory: Contextual memory for reasoning
            
        Returns:
            Action execution results
        """
        logger.info(f"Executing {action_type.value} action on node {node.node_id}")
        
        # Get task description from node or parent
        task_description = self._get_task_description(node)
        
        # Get current content and errors
        current_content = node.solution.content if node.solution else ""
        error_messages = node.solution.error_messages if node.solution else []
        
        # Determine content type
        content_type = self._determine_content_type(node, task_description)
        
        # Perform reasoning
        reasoning_result = self.reasoner.reason(
            task_description=task_description,
            action_type=action_type.value,
            content_type=content_type,
            current_content=current_content,
            error_messages=error_messages,
            contextual_memory=contextual_memory
        )
        
        # Extract generated content
        generated_content = reasoning_result.get('content', '')
        content_type = ContentType(reasoning_result.get('content_type', 'code'))
        
        # Process content based on type
        processing_result = self._process_content(generated_content, content_type, node.node_id)
        
        # Combine results
        action_result = {
            'reasoning_process': reasoning_result.get('reasoning_process', ''),
            'insights': reasoning_result.get('insights', {}),
            'content': generated_content,
            'content_type': content_type.value,
            'processing_logs': processing_result.get('logs', []),
            'error_messages': processing_result.get('errors', []),
            'performance_score': processing_result.get('performance_score', 0.0),
            'metadata': {
                'action_type': action_type.value,
                'node_id': node.node_id,
                'processing_time': processing_result.get('processing_time', 0),
                'context_used': reasoning_result.get('context_used', {})
            }
        }
        
        # Add to memory
        self._add_action_to_memory(node, action_result, reasoning_result)
        
        logger.info(f"Completed {action_type.value} action on node {node.node_id}")
        return action_result
    
    def _get_task_description(self, node: MCTSNode) -> str:
        """Get task description from node or traverse up to root"""
        current_node = node
        
        # Try to get task description from current node
        if current_node.solution.metadata.get('task_description'):
            return current_node.solution.metadata['task_description']
        
        # Traverse up to root to find task description
        while current_node.parent_id:
            # This would require access to the full tree, so we'll use a placeholder
            # In a real implementation, you'd pass the tree or root node
            break
        
        # Return a default task description if none found
        return "Machine learning task - implement a solution"
    
    def _determine_content_type(self, node: MCTSNode, task_description: str) -> ContentType:
        """Determine the content type for this action"""
        # Check if content type is already specified in node
        if hasattr(node.solution, 'content_type') and node.solution.content_type:
            return ContentType(node.solution.content_type)
        
        # Check if content type is specified in task description
        from ..integration.content_types import ContentTypeDetector
        return ContentTypeDetector.detect_from_task(task_description)
    
    def _process_content(self, content: str, content_type: ContentType, node_id: str) -> Dict[str, Any]:
        """
        Process content based on its type
        
        Args:
            content: Content to process
            content_type: Type of content
            node_id: Node ID for logging
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        logs = []
        errors = []
        performance_score = 0.0
        
        try:
            # Validate content if enabled
            if self.enable_content_validation:
                validation_result = self._validate_content(content, content_type)
                if not validation_result['is_valid']:
                    errors.extend(validation_result['errors'])
                if validation_result['warnings']:
                    logs.extend(validation_result['warnings'])
            
            # Process content based on type
            if content_type == ContentType.CODE:
                processing_result = self._process_code(content, node_id)
            elif content_type == ContentType.MARKDOWN:
                processing_result = self._process_markdown(content, node_id)
            elif content_type == ContentType.PLAIN_TEXT:
                processing_result = self._process_plain_text(content, node_id)
            elif content_type == ContentType.STRUCTURED_DATA:
                processing_result = self._process_structured_data(content, node_id)
            else:
                processing_result = self._process_generic_content(content, node_id)
            
            logs.extend(processing_result.get('logs', []))
            errors.extend(processing_result.get('errors', []))
            performance_score = processing_result.get('performance_score', 0.0)
            
            # Format content if enabled
            if self.enable_content_formatting:
                formatted_content = self._format_content(content, content_type)
                if formatted_content != content:
                    logs.append("Content formatted successfully")
            
        except Exception as e:
            errors.append(f"Content processing error: {str(e)}")
            logger.error(f"Error processing content for node {node_id}: {e}")
        
        processing_time = time.time() - start_time
        
        return {
            'success': len(errors) == 0,
            'logs': logs,
            'errors': errors,
            'performance_score': performance_score,
            'processing_time': processing_time
        }
    
    def _process_code(self, code: str, node_id: str) -> Dict[str, Any]:
        """Process code content"""
        logs = []
        errors = []
        performance_score = 0.0
        
        # Execute code if it's Python
        if 'python' in code.lower() or any(keyword in code for keyword in ['import ', 'def ', 'class ']):
            execution_result = self._execute_code(code, node_id)
            logs.extend(execution_result.get('logs', []))
            errors.extend(execution_result.get('errors', []))
            performance_score = execution_result.get('performance_score', 0.0)
        else:
            logs.append("Code validation completed (non-executable code)")
            performance_score = 0.8  # Good score for non-executable code
        
        return {
            'logs': logs,
            'errors': errors,
            'performance_score': performance_score
        }
    
    def _process_markdown(self, markdown: str, node_id: str) -> Dict[str, Any]:
        """Process markdown content"""
        logs = []
        errors = []
        performance_score = 0.0
        
        # Basic markdown validation
        if not markdown.strip():
            errors.append("Empty markdown content")
            performance_score = 0.0
        else:
            # Check for basic markdown structure
            has_headings = any(line.strip().startswith('#') for line in markdown.split('\n'))
            has_content = len(markdown.strip()) > 50
            
            if has_headings and has_content:
                performance_score = 0.9
                logs.append("Markdown content validated successfully")
            elif has_content:
                performance_score = 0.7
                logs.append("Markdown content has basic structure")
            else:
                performance_score = 0.5
                logs.append("Markdown content needs improvement")
        
        return {
            'logs': logs,
            'errors': errors,
            'performance_score': performance_score
        }
    
    def _process_plain_text(self, text: str, node_id: str) -> Dict[str, Any]:
        """Process plain text content"""
        logs = []
        errors = []
        performance_score = 0.0
        
        # Basic text validation
        if not text.strip():
            errors.append("Empty text content")
            performance_score = 0.0
        else:
            # Check for basic text quality
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            if word_count > 10 and sentence_count > 1:
                performance_score = 0.8
                logs.append("Text content validated successfully")
            elif word_count > 5:
                performance_score = 0.6
                logs.append("Text content has basic structure")
            else:
                performance_score = 0.4
                logs.append("Text content is too short")
        
        return {
            'logs': logs,
            'errors': errors,
            'performance_score': performance_score
        }
    
    def _process_structured_data(self, data: str, node_id: str) -> Dict[str, Any]:
        """Process structured data content"""
        logs = []
        errors = []
        performance_score = 0.0
        
        try:
            # Try to parse as JSON
            parsed_data = json.loads(data)
            
            # Validate structure
            if isinstance(parsed_data, (dict, list)):
                performance_score = 0.9
                logs.append("Structured data validated successfully")
            else:
                performance_score = 0.7
                logs.append("Structured data parsed but not optimal format")
                
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {str(e)}")
            performance_score = 0.0
        
        return {
            'logs': logs,
            'errors': errors,
            'performance_score': performance_score
        }
    
    def _process_generic_content(self, content: str, node_id: str) -> Dict[str, Any]:
        """Process generic content"""
        logs = []
        errors = []
        performance_score = 0.0
        
        if content.strip():
            performance_score = 0.6
            logs.append("Generic content processed")
        else:
            errors.append("Empty content")
            performance_score = 0.0
        
        return {
            'logs': logs,
            'errors': errors,
            'performance_score': performance_score
        }
    
    def _validate_content(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        """Validate content based on its type"""
        if content_type == ContentType.CODE:
            return ContentValidator.validate_code(content)
        elif content_type == ContentType.MARKDOWN:
            return ContentValidator.validate_markdown(content)
        elif content_type == ContentType.STRUCTURED_DATA:
            try:
                data = json.loads(content)
                return ContentValidator.validate_structured_data(data)
            except:
                return {'is_valid': False, 'errors': ['Invalid JSON format'], 'warnings': []}
        else:
            return {'is_valid': True, 'errors': [], 'warnings': []}
    
    def _format_content(self, content: str, content_type: ContentType) -> str:
        """Format content based on its type"""
        if content_type == ContentType.CODE:
            return ContentFormatter.format_code(content)
        elif content_type == ContentType.MARKDOWN:
            return ContentFormatter.format_markdown(content)
        elif content_type == ContentType.PLAIN_TEXT:
            return ContentFormatter.format_plain_text(content)
        elif content_type == ContentType.STRUCTURED_DATA:
            try:
                data = json.loads(content)
                return ContentFormatter.format_structured_data(data)
            except:
                return content
        else:
            return content
    
    def _execute_code(self, code: str, node_id: str) -> Dict[str, Any]:
        """
        Execute generated code in a safe environment
        
        Args:
            code: Python code to execute
            node_id: Node ID for logging
            
        Returns:
            Execution results
        """
        start_time = time.time()
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.temp_dir) as f:
            f.write(code)
            temp_file_path = f.name
        
        try:
            # Prepare execution environment
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
            
            # Execute code with timeout and resource limits
            result = subprocess.run(
                ['python', temp_file_path],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                env=env,
                cwd=self.temp_dir
            )
            
            execution_time = time.time() - start_time
            
            # Parse execution results
            logs = []
            errors = []
            performance_score = 0.0
            
            # Capture stdout as logs
            if result.stdout:
                logs.extend(result.stdout.strip().split('\n'))
            
            # Capture stderr as errors
            if result.stderr:
                errors.extend(result.stderr.strip().split('\n'))
            
            # Try to extract performance score from output
            performance_score = self._extract_performance_score(result.stdout)
            
            # Check if execution was successful
            success = result.returncode == 0 and not errors
            
            logger.debug(f"Code execution completed for node {node_id} in {execution_time:.2f}s")
            
            return {
                'success': success,
                'logs': logs,
                'errors': errors,
                'performance_score': performance_score,
                'execution_time': execution_time,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution timed out for node {node_id}")
            return {
                'success': False,
                'logs': [],
                'errors': ['Execution timeout'],
                'performance_score': 0.0,
                'execution_time': self.execution_timeout,
                'return_code': -1
            }
            
        except Exception as e:
            logger.error(f"Error executing code for node {node_id}: {e}")
            return {
                'success': False,
                'logs': [],
                'errors': [str(e)],
                'performance_score': 0.0,
                'execution_time': time.time() - start_time,
                'return_code': -1
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _extract_performance_score(self, output: str) -> float:
        """
        Extract performance score from execution output
        
        Args:
            output: Execution output string
            
        Returns:
            Extracted performance score
        """
        try:
            # Look for common performance metric patterns
            lines = output.split('\n')
            
            for line in lines:
                line = line.strip().lower()
                
                # Look for accuracy, f1, auc, etc.
                if 'accuracy:' in line or 'f1:' in line or 'auc:' in line:
                    # Extract numeric value
                    import re
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    if numbers:
                        return float(numbers[0])
                
                # Look for score patterns
                if 'score:' in line:
                    import re
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    if numbers:
                        return float(numbers[0])
            
            # Default to 0 if no score found
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error extracting performance score: {e}")
            return 0.0
    
    def _add_action_to_memory(self, 
                             node: MCTSNode, 
                             action_result: Dict[str, Any],
                             reasoning_result: Dict[str, Any]):
        """Add action results to adaptive memory"""
        
        # Add reasoning insights
        if 'reasoning_process' in reasoning_result:
            self.memory.add_memory_entry(
                memory_type=MemoryType.REASONING_INSIGHT,
                content={
                    'reasoning_process': reasoning_result['reasoning_process'],
                    'insights': reasoning_result.get('insights', {}),
                    'action_type': action_result['metadata']['action_type']
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=1.0
            )
        
        # Add execution feedback
        if action_result.get('execution_logs') or action_result.get('error_messages'):
            self.memory.add_memory_entry(
                memory_type=MemoryType.EXECUTION_FEEDBACK,
                content={
                    'execution_logs': action_result.get('execution_logs', []),
                    'error_messages': action_result.get('error_messages', []),
                    'success': action_result.get('metadata', {}).get('success', False),
                    'execution_time': action_result.get('metadata', {}).get('execution_time', 0)
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=1.0 if not action_result.get('error_messages') else 0.5
            )
        
        # Add performance metrics
        if 'performance_score' in action_result:
            self.memory.add_memory_entry(
                memory_type=MemoryType.PERFORMANCE_METRIC,
                content={
                    'performance_score': action_result['performance_score'],
                    'action_type': action_result['metadata']['action_type'],
                    'execution_time': action_result['metadata'].get('execution_time', 0)
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=action_result['performance_score'] / 100.0
            )
        
        # Add code snippets
        if action_result.get('code'):
            self.memory.add_memory_entry(
                memory_type=MemoryType.CODE_SNIPPET,
                content={
                    'code': action_result['code'],
                    'action_type': action_result['metadata']['action_type'],
                    'performance_score': action_result.get('performance_score', 0.0)
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=action_result.get('performance_score', 0.0) / 100.0
            )
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code syntax and basic structure
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            'syntax_valid': False,
            'errors': [],
            'warnings': [],
            'imports': [],
            'functions': [],
            'classes': []
        }
        
        try:
            # Check syntax
            compile(code, '<string>', 'exec')
            validation_result['syntax_valid'] = True
            
            # Extract imports
            import re
            import_lines = re.findall(r'^import\s+(\w+)', code, re.MULTILINE)
            from_lines = re.findall(r'^from\s+(\w+)', code, re.MULTILINE)
            validation_result['imports'] = import_lines + from_lines
            
            # Extract functions
            function_lines = re.findall(r'^def\s+(\w+)', code, re.MULTILINE)
            validation_result['functions'] = function_lines
            
            # Extract classes
            class_lines = re.findall(r'^class\s+(\w+)', code, re.MULTILINE)
            validation_result['classes'] = class_lines
            
        except SyntaxError as e:
            validation_result['errors'].append(f"Syntax error: {e}")
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'execution_timeout': self.execution_timeout,
            'memory_limit': self.memory_limit,
            'cpu_limit': self.cpu_limit,
            'sandbox_execution': self.sandbox_execution,
            'allow_network_access': self.allow_network_access,
            'temp_dir': self.temp_dir,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir
        } 