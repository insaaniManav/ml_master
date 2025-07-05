"""
ML-Master Main Integration

This module provides the main ML-Master class that integrates all
components into a unified AI-for-AI framework.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from ..memory import AdaptiveMemory
from ..exploration import TreeSearch, MCTSNode, ActionType
from ..reasoning import SteerableReasoner
from ..exploration.action_executor import ActionExecutor
from .content_types import ContentType, ContentOutput, ContentFormatter, ContentTypeDetector


class MLMaster:
    """
    ML-Master: AI-for-AI Framework
    
    Main integration class that combines adaptive memory, balanced
    multi-trajectory exploration, and steerable reasoning into a
    unified AI4AI framework supporting multiple content types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML-Master framework
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing ML-Master components...")
        
        # Memory system
        self.memory = AdaptiveMemory(config.get('memory', {}))
        
        # Reasoning system
        self.reasoner = SteerableReasoner(config.get('reasoning', {}), self.memory)
        
        # Action executor
        self.action_executor = ActionExecutor(
            config.get('environment', {}),
            self.reasoner,
            self.memory
        )
        
        # Exploration system
        self.tree_search = TreeSearch(
            config.get('exploration', {}),
            self.memory,
            self._action_executor_wrapper,
            self._reward_calculator
        )
        
        # Framework state
        self.is_initialized = False
        self.current_task = None
        self.start_time = None
        
        logger.info("ML-Master initialization completed")
    
    def solve_task(self, 
                   task_description: str,
                   content_type: Optional[ContentType] = None,
                   max_time: Optional[float] = None,
                   max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Solve a task using the ML-Master framework
        
        Args:
            task_description: Description of the task to solve
            content_type: Type of content to generate (auto-detected if None)
            max_time: Maximum time to spend on the task (in seconds)
            max_iterations: Maximum number of MCTS iterations
            
        Returns:
            Task solution and results
        """
        logger.info(f"Starting task: {task_description[:100]}...")
        
        # Detect content type if not specified
        if content_type is None:
            content_type = ContentTypeDetector.detect_from_task(task_description)
            logger.info(f"Auto-detected content type: {content_type.value}")
        
        # Initialize task
        self.current_task = task_description
        self.start_time = time.time()
        
        # Override max iterations if specified
        if max_iterations:
            self.tree_search.max_iterations = max_iterations
        
        # Initialize search tree with content type
        root_id = self.tree_search.initialize_search(task_description)
        
        # Run exploration and reasoning
        search_results = self.tree_search.run_search(max_time=max_time)
        
        # Compile final results
        final_results = self._compile_results(search_results, content_type)
        
        logger.info(f"Task completed in {time.time() - self.start_time:.2f}s")
        return final_results
    
    def generate_content(self,
                        task_description: str,
                        content_type: Optional[ContentType] = None,
                        format_output: bool = True) -> ContentOutput:
        """
        Generate content for a specific task
        
        Args:
            task_description: Description of the task
            content_type: Type of content to generate
            format_output: Whether to format the output
            
        Returns:
            Generated content output
        """
        # Detect content type if not specified
        if content_type is None:
            content_type = ContentTypeDetector.detect_from_task(task_description)
        
        # Use a simplified approach for content generation
        reasoning_result = self.reasoner.reason(
            task_description=task_description,
            action_type="draft",
            content_type=content_type
        )
        
        content = reasoning_result.get('content', '')
        
        # Format content if requested
        if format_output:
            content = self._format_content_output(content, content_type)
        
        return ContentOutput(
            content_type=content_type,
            content=content,
            metadata={
                'task_description': task_description,
                'reasoning_process': reasoning_result.get('reasoning_process', ''),
                'insights': reasoning_result.get('insights', {}),
                'generation_time': time.time() - self.start_time if self.start_time else 0
            }
        )
    
    def _format_content_output(self, content: str, content_type: ContentType) -> str:
        """Format content output based on type"""
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
    
    def _action_executor_wrapper(self, 
                                node: MCTSNode,
                                action_type: ActionType,
                                contextual_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper for action executor to integrate with tree search
        
        Args:
            node: MCTS node
            action_type: Action type to execute
            contextual_memory: Contextual memory
            
        Returns:
            Action execution results
        """
        return self.action_executor.execute_action(node, action_type, contextual_memory)
    
    def _reward_calculator(self, 
                          node: MCTSNode,
                          improvement_threshold: float,
                          best_score_so_far: float) -> float:
        """
        Calculate reward for a node based on the paper's reward function
        
        Args:
            node: MCTS node
            improvement_threshold: Threshold for improvement
            best_score_so_far: Best score observed so far
            
        Returns:
            Calculated reward
        """
        return node.calculate_reward(improvement_threshold, best_score_so_far)
    
    def _compile_results(self, search_results: Dict[str, Any], content_type: ContentType) -> Dict[str, Any]:
        """
        Compile final results from search
        
        Args:
            search_results: Results from tree search
            content_type: Type of content generated
            
        Returns:
            Compiled final results
        """
        best_solution = search_results.get('best_solution', {})
        
        # Get memory statistics
        memory_stats = self.memory.get_memory_stats()
        
        # Get execution statistics
        execution_stats = self.action_executor.get_execution_stats()
        
        # Compile comprehensive results
        final_results = {
            'task': self.current_task,
            'content_type': content_type.value,
            'solution': {
                'content': best_solution.get('content', ''),
                'content_type': content_type.value,
                'performance_score': best_solution.get('performance_score', 0.0),
                'node_id': best_solution.get('node_id', ''),
                'path_to_root': best_solution.get('path_to_root', [])
            },
            'performance': {
                'best_score': search_results.get('performance_metrics', {}).get('best_score', 0.0),
                'iterations_completed': search_results.get('performance_metrics', {}).get('iterations_completed', 0),
                'total_time': search_results.get('performance_metrics', {}).get('total_time', 0),
                'iterations_per_second': search_results.get('search_stats', {}).get('iterations_per_second', 0)
            },
            'exploration': {
                'total_nodes': search_results.get('tree_stats', {}).get('total_nodes', 0),
                'terminal_nodes': search_results.get('tree_stats', {}).get('terminal_nodes', 0),
                'expanded_nodes': search_results.get('tree_stats', {}).get('expanded_nodes', 0),
                'average_depth': search_results.get('tree_stats', {}).get('average_depth', 0),
                'tree_height': search_results.get('tree_stats', {}).get('tree_height', 0)
            },
            'memory': memory_stats,
            'execution': execution_stats,
            'framework_info': {
                'version': self.config.get('framework', {}).get('version', '1.0.0'),
                'model_used': self.config.get('reasoning', {}).get('model_name', 'o3'),
                'parallel_workers': self.config.get('exploration', {}).get('num_workers', 3),
                'content_type': content_type.value
            }
        }
        
        return final_results
    
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific node"""
        return self.tree_search.get_node_info(node_id)
    
    def get_memory_insights(self, 
                           node_id: str,
                           memory_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory insights for a specific node
        
        Args:
            node_id: Node ID
            memory_type: Optional memory type filter
            
        Returns:
            Memory insights
        """
        # Get contextual memory for the node
        node = self.tree_search.nodes.get(node_id)
        if not node:
            return {}
        
        contextual_memory = self.memory.get_contextual_memory(
            current_node_id=node_id,
            parent_node_id=node.parent_id,
            sibling_node_ids=node.sibling_ids
        )
        
        if memory_type:
            return {memory_type: contextual_memory.get(memory_type, [])}
        
        return contextual_memory
    
    def save_state(self, filepath: str):
        """Save current framework state"""
        state = {
            'config': self.config,
            'current_task': self.current_task,
            'start_time': self.start_time,
            'memory': None,  # Will be saved separately
            'tree_search': None  # Will be saved separately
        }
        
        # Save main state
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save memory
        memory_filepath = filepath.replace('.json', '_memory.json')
        self.memory.save_memory(memory_filepath)
        
        # Save tree search
        tree_filepath = filepath.replace('.json', '_tree.json')
        self.tree_search.save_tree(tree_filepath)
        
        logger.info(f"Saved ML-Master state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load framework state from file"""
        # Load main state
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state['config']
        self.current_task = state['current_task']
        self.start_time = state['start_time']
        
        # Load memory
        memory_filepath = filepath.replace('.json', '_memory.json')
        if os.path.exists(memory_filepath):
            self.memory.load_memory(memory_filepath)
        
        # Load tree search
        tree_filepath = filepath.replace('.json', '_tree.json')
        if os.path.exists(tree_filepath):
            self.tree_search.load_tree(tree_filepath)
        
        logger.info(f"Loaded ML-Master state from {filepath}")
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics"""
        return {
            'framework': {
                'version': self.config.get('framework', {}).get('version', '1.0.0'),
                'is_initialized': self.is_initialized,
                'current_task': self.current_task is not None
            },
            'memory': self.memory.get_memory_stats(),
            'exploration': {
                'total_nodes': len(self.tree_search.nodes),
                'best_score': self.tree_search.best_score_so_far,
                'iterations_completed': self.tree_search.iteration_count
            },
            'execution': self.action_executor.get_execution_stats(),
            'reasoning': {
                'model_name': self.config.get('reasoning', {}).get('model_name', 'o3'),
                'context_compression': self.reasoner.context_compression,
                'sibling_inclusion': self.reasoner.sibling_node_inclusion
            }
        }
    
    def reset(self):
        """Reset the framework to initial state"""
        logger.info("Resetting ML-Master framework")
        
        # Clear memory
        self.memory.clear_memory()
        
        # Reset tree search
        self.tree_search.nodes.clear()
        self.tree_search.root_node_id = None
        self.tree_search.best_score_so_far = -float('inf')
        self.tree_search.best_node_id = None
        self.tree_search.iteration_count = 0
        
        # Reset state
        self.current_task = None
        self.start_time = None
        
        logger.info("ML-Master framework reset completed")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up ML-Master resources")
        
        # Clean up tree search
        self.tree_search.cleanup()
        
        # Clean up reasoner
        self.reasoner.cleanup()
        
        # Clean up action executor
        # (No specific cleanup needed for action executor)
        
        logger.info("ML-Master cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup() 