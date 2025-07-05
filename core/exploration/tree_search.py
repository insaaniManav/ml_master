"""
Tree Search Implementation

This module implements the MCTS-inspired tree search algorithm for
balanced multi-trajectory exploration as described in the ML-Master paper.
"""

import time
import math
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from .mcts_node import MCTSNode, ActionType, NodeState
from ..memory import AdaptiveMemory, MemoryType


class TreeSearch:
    """
    MCTS-Inspired Tree Search for Balanced Multi-Trajectory Exploration
    
    Implements the tree-guided exploration component of ML-Master using
    Monte Carlo Tree Search principles with parallel execution capabilities.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 memory: AdaptiveMemory,
                 action_executor: Callable,
                 reward_calculator: Callable):
        """
        Initialize the tree search system
        
        Args:
            config: Configuration dictionary
            memory: Adaptive memory instance
            action_executor: Function to execute actions (Draft, Debug, Improve)
            reward_calculator: Function to calculate rewards
        """
        self.config = config
        self.memory = memory
        self.action_executor = action_executor
        self.reward_calculator = reward_calculator
        
        # MCTS parameters
        self.uct_constant = config.get('uct_constant', 1.414)
        self.max_iterations = config.get('max_iterations', 1000)
        self.max_depth = config.get('max_depth', 50)
        
        # Parallel execution settings
        self.num_workers = config.get('num_workers', 3)
        self.max_parallel_branches = config.get('max_parallel_branches', 5)
        
        # Stopping conditions
        self.improvement_threshold = config.get('improvement_threshold', 0.001)
        self.max_failed_improvements = config.get('max_failed_improvements', 3)
        self.max_debug_depth = config.get('max_debug_depth', 20)
        self.max_debug_attempts = config.get('max_debug_attempts', 3)
        
        # Tree state
        self.nodes: Dict[str, MCTSNode] = {}
        self.root_node_id: Optional[str] = None
        self.best_score_so_far = -float('inf')
        self.best_node_id: Optional[str] = None
        
        # Exploration tracking
        self.iteration_count = 0
        self.start_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        logger.info(f"Initialized TreeSearch with {self.num_workers} workers")
    
    def initialize_search(self, task_description: str) -> str:
        """
        Initialize the search tree with a root node
        
        Args:
            task_description: Description of the ML task to solve
            
        Returns:
            Root node ID
        """
        # Create root node
        root_id = f"root_{uuid.uuid4().hex[:8]}"
        root_node = MCTSNode(
            node_id=root_id,
            parent_id=None,
            action_type=None,
            depth=0
        )
        
        # Store task description in root node
        root_node.solution.metadata['task_description'] = task_description
        
        # Add to tree
        self.nodes[root_id] = root_node
        self.root_node_id = root_id
        
        logger.info(f"Initialized search tree with root node {root_id}")
        return root_id
    
    def run_search(self, max_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the MCTS search algorithm
        
        Args:
            max_time: Maximum time to run search (in seconds)
            
        Returns:
            Search results and statistics
        """
        if not self.root_node_id:
            raise ValueError("Search tree not initialized. Call initialize_search() first.")
        
        logger.info(f"Starting MCTS search with max_iterations={self.max_iterations}")
        
        # Run MCTS iterations
        for iteration in range(self.max_iterations):
            # Check time limit
            if max_time and (time.time() - self.start_time) > max_time:
                logger.info(f"Time limit reached after {iteration} iterations")
                break
            
            # Single MCTS iteration
            self._mcts_iteration()
            self.iteration_count += 1
            
            # Log progress
            if iteration % 100 == 0:
                self._log_progress(iteration)
        
        # Get best solution
        best_solution = self._get_best_solution()
        
        # Compile results
        results = {
            'best_solution': best_solution,
            'search_stats': self._get_search_stats(),
            'tree_stats': self._get_tree_stats(),
            'performance_metrics': {
                'best_score': self.best_score_so_far,
                'iterations_completed': self.iteration_count,
                'total_time': time.time() - self.start_time
            }
        }
        
        logger.info(f"Search completed. Best score: {self.best_score_so_far}")
        return results
    
    def _mcts_iteration(self):
        """Execute a single MCTS iteration"""
        # Selection: traverse from root to leaf
        leaf_node = self._selection(self.root_node_id)
        
        # Expansion: expand leaf node if possible
        if not leaf_node.is_terminal and leaf_node.depth < self.max_depth:
            self._expansion(leaf_node)
        
        # Simulation: execute action and get reward
        reward = self._simulation(leaf_node)
        
        # Backpropagation: propagate reward up the tree
        self._backpropagation(leaf_node.node_id, reward)
    
    def _selection(self, node_id: str) -> MCTSNode:
        """
        Selection phase: traverse from root to leaf using UCT
        
        Args:
            node_id: Starting node ID
            
        Returns:
            Selected leaf node
        """
        current_node = self.nodes[node_id]
        
        # If node is not fully expanded or is terminal, return it
        if not current_node.is_expanded or current_node.is_terminal:
            return current_node
        
        # Find child with highest UCT value
        best_child_id = None
        best_uct_value = -float('inf')
        
        for child_id in current_node.children:
            child_node = self.nodes[child_id]
            uct_value = child_node.metrics.get_uct_value(
                current_node.metrics.visit_count,
                self.uct_constant
            )
            
            if uct_value > best_uct_value:
                best_uct_value = uct_value
                best_child_id = child_id
        
        if best_child_id:
            return self._selection(best_child_id)
        else:
            return current_node
    
    def _expansion(self, node: MCTSNode):
        """
        Expansion phase: create new child nodes
        
        Args:
            node: Node to expand
        """
        # Determine next action
        next_action = node.get_next_action()
        if not next_action:
            node.mark_terminal()
            return
        
        # Create child node
        child_id = f"{node.node_id}_{next_action.value}_{uuid.uuid4().hex[:8]}"
        child_node = MCTSNode(
            node_id=child_id,
            parent_id=node.node_id,
            action_type=next_action,
            depth=node.depth + 1
        )
        
        # Add to tree
        self.nodes[child_id] = child_node
        node.add_child(child_id)
        
        # Update node state
        node.update_state(NodeState.EXPANDING)
        node.mark_expanded()
        
        # Add sibling relationships
        for sibling_id in node.children:
            if sibling_id != child_id:
                child_node.add_sibling(sibling_id)
                self.nodes[sibling_id].add_sibling(child_id)
        
        logger.debug(f"Expanded node {node.node_id} with child {child_id} (action: {next_action.value})")
    
    def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: execute action and calculate reward
        
        Args:
            node: Node to simulate
            
        Returns:
            Calculated reward
        """
        # Get contextual memory for reasoning
        contextual_memory = self.memory.get_contextual_memory(
            current_node_id=node.node_id,
            parent_node_id=node.parent_id,
            sibling_node_ids=node.sibling_ids
        )
        
        # Determine action type for this node
        if node.action_type is None:
            # For root node or nodes without explicit action type, determine based on state
            action_type = node.get_next_action()
            if action_type is None:
                # Default to draft if no action can be determined
                action_type = ActionType.DRAFT
        else:
            action_type = node.action_type
        
        # Execute action
        try:
            action_result = self.action_executor(
                node=node,
                action_type=action_type,
                contextual_memory=contextual_memory
            )
            
            # Update node with results
            node.update_solution(
                code=action_result.get('code', ''),
                performance_score=action_result.get('performance_score', 0.0),
                execution_logs=action_result.get('execution_logs', []),
                error_messages=action_result.get('error_messages', []),
                metadata=action_result.get('metadata', {})
            )
            
            # Calculate reward
            reward = self.reward_calculator(
                node=node,
                improvement_threshold=self.improvement_threshold,
                best_score_so_far=self.best_score_so_far
            )
            
            # Update best score if improved
            if node.solution.performance_score > self.best_score_so_far:
                self.best_score_so_far = node.solution.performance_score
                self.best_node_id = node.node_id
                logger.info(f"New best score: {self.best_score_so_far} (node: {node.node_id})")
            
            # Add to memory
            self._add_to_memory(node, action_result, reward)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error in simulation for node {node.node_id}: {e}")
            return -1.0
    
    def _backpropagation(self, node_id: str, reward: float):
        """
        Backpropagation phase: propagate reward up the tree
        
        Args:
            node_id: Node ID to start backpropagation from
            reward: Reward to propagate
        """
        current_id = node_id
        
        while current_id:
            node = self.nodes[current_id]
            node.metrics.update_reward(reward)
            current_id = node.parent_id
    
    def _add_to_memory(self, node: MCTSNode, action_result: Dict[str, Any], reward: float):
        """Add exploration results to adaptive memory"""
        # Get action type safely
        action_type_value = node.action_type.value if node.action_type else "unknown"
        
        # Add reasoning insights
        if 'reasoning_process' in action_result:
            self.memory.add_memory_entry(
                memory_type=MemoryType.REASONING_INSIGHT,
                content={
                    'reasoning_process': action_result['reasoning_process'],
                    'action_type': action_type_value,
                    'reward': reward
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=reward + 1.0  # Normalize to positive
            )
        
        # Add execution feedback
        if 'execution_logs' in action_result or 'error_messages' in action_result:
            self.memory.add_memory_entry(
                memory_type=MemoryType.EXECUTION_FEEDBACK,
                content={
                    'execution_logs': action_result.get('execution_logs', []),
                    'error_messages': action_result.get('error_messages', []),
                    'action_type': action_type_value
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
                    'action_type': action_type_value,
                    'improvement': action_result['performance_score'] - self.best_score_so_far
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=action_result['performance_score'] / 100.0  # Normalize
            )
        
        # Add code snippets
        if 'code' in action_result and action_result['code']:
            self.memory.add_memory_entry(
                memory_type=MemoryType.CODE_SNIPPET,
                content={
                    'code': action_result['code'],
                    'action_type': action_type_value,
                    'performance_score': action_result.get('performance_score', 0.0)
                },
                node_id=node.node_id,
                parent_node_id=node.parent_id,
                sibling_node_ids=node.sibling_ids,
                relevance_score=action_result.get('performance_score', 0.0) / 100.0
            )
    
    def _get_best_solution(self) -> Dict[str, Any]:
        """Get the best solution found during search"""
        if not self.best_node_id:
            return {}
        
        best_node = self.nodes[self.best_node_id]
        return {
            'node_id': best_node.node_id,
            'code': best_node.solution.code,
            'performance_score': best_node.solution.performance_score,
            'execution_logs': best_node.solution.execution_logs,
            'error_messages': best_node.solution.error_messages,
            'metadata': best_node.solution.metadata,
            'path_to_root': self._get_path_to_root(best_node.node_id)
        }
    
    def _get_path_to_root(self, node_id: str) -> List[str]:
        """Get the path from a node to the root"""
        path = []
        current_id = node_id
        
        while current_id:
            path.append(current_id)
            current_id = self.nodes[current_id].parent_id
        
        return list(reversed(path))
    
    def _log_progress(self, iteration: int):
        """Log search progress"""
        elapsed_time = time.time() - self.start_time
        logger.info(f"Iteration {iteration}/{self.max_iterations} "
                   f"(Time: {elapsed_time:.1f}s, Best: {self.best_score_so_far:.4f})")
    
    def _get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'iterations_completed': self.iteration_count,
            'total_time': time.time() - self.start_time,
            'iterations_per_second': self.iteration_count / (time.time() - self.start_time),
            'best_score': self.best_score_so_far,
            'best_node_id': self.best_node_id
        }
    
    def _get_tree_stats(self) -> Dict[str, Any]:
        """Get tree statistics"""
        total_nodes = len(self.nodes)
        terminal_nodes = sum(1 for node in self.nodes.values() if node.is_terminal)
        expanded_nodes = sum(1 for node in self.nodes.values() if node.is_expanded)
        
        # Calculate average depth
        depths = [node.depth for node in self.nodes.values()]
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        # Calculate branching factor
        children_counts = [len(node.children) for node in self.nodes.values()]
        avg_branching = sum(children_counts) / len(children_counts) if children_counts else 0
        
        return {
            'total_nodes': total_nodes,
            'terminal_nodes': terminal_nodes,
            'expanded_nodes': expanded_nodes,
            'average_depth': avg_depth,
            'average_branching_factor': avg_branching,
            'tree_height': max(depths) if depths else 0
        }
    
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific node"""
        if node_id not in self.nodes:
            return {}
        
        node = self.nodes[node_id]
        return node.get_node_info()
    
    def save_tree(self, filepath: str):
        """Save the search tree to file"""
        import json
        
        tree_data = {
            'config': self.config,
            'root_node_id': self.root_node_id,
            'best_score_so_far': self.best_score_so_far,
            'best_node_id': self.best_node_id,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(tree_data, f, indent=2)
        
        logger.info(f"Saved search tree to {filepath}")
    
    def load_tree(self, filepath: str):
        """Load the search tree from file"""
        import json
        
        with open(filepath, 'r') as f:
            tree_data = json.load(f)
        
        self.config = tree_data['config']
        self.root_node_id = tree_data['root_node_id']
        self.best_score_so_far = tree_data['best_score_so_far']
        self.best_node_id = tree_data['best_node_id']
        
        self.nodes = {}
        for node_id, node_data in tree_data['nodes'].items():
            self.nodes[node_id] = MCTSNode.from_dict(node_data)
        
        logger.info(f"Loaded search tree from {filepath}")
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("TreeSearch cleanup completed") 