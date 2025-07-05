"""
MCTS Node Implementation

This module implements the MCTS node structure used in the balanced
multi-trajectory exploration system.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class NodeState(Enum):
    """Possible states of an MCTS node"""
    DRAFT = "draft"
    DEBUG = "debug"
    IMPROVE = "improve"
    TERMINAL = "terminal"
    EXPANDING = "expanding"


class ActionType(Enum):
    """Types of actions that can be performed on nodes"""
    DRAFT = "draft"
    DEBUG = "debug"
    IMPROVE = "improve"


@dataclass
class NodeMetrics:
    """Metrics associated with a node"""
    visit_count: int = 0
    total_reward: float = 0.0
    best_reward: float = -float('inf')
    average_reward: float = 0.0
    last_improvement: float = 0.0
    failed_improvements: int = 0
    debug_attempts: int = 0
    
    def update_reward(self, reward: float):
        """Update node metrics with new reward"""
        self.visit_count += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.visit_count
        self.best_reward = max(self.best_reward, reward)
        
        if reward > 0:
            self.last_improvement = time.time()
        else:
            self.failed_improvements += 1
    
    def get_uct_value(self, parent_visits: int, uct_constant: float) -> float:
        """Calculate UCT value for this node"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.average_reward
        exploration = uct_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration


@dataclass
class NodeSolution:
    """Solution associated with a node"""
    code: str = ""
    performance_score: float = 0.0
    execution_logs: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if the solution is valid"""
        return len(self.error_messages) == 0 and len(self.code.strip()) > 0
    
    def has_errors(self) -> bool:
        """Check if the solution has errors"""
        return len(self.error_messages) > 0


class MCTSNode:
    """
    MCTS Node for Balanced Multi-Trajectory Exploration
    
    Represents a node in the Monte Carlo Tree Search tree used for
    exploring different solution trajectories in parallel.
    """
    
    def __init__(self, 
                 node_id: str,
                 parent_id: Optional[str] = None,
                 action_type: Optional[ActionType] = None,
                 depth: int = 0):
        """
        Initialize an MCTS node
        
        Args:
            node_id: Unique identifier for the node
            parent_id: ID of the parent node
            action_type: Type of action that created this node
            depth: Depth of the node in the tree
        """
        self.node_id = node_id
        self.parent_id = parent_id
        self.action_type = action_type
        self.depth = depth
        
        # Node state and metrics
        self.state = NodeState.DRAFT
        self.metrics = NodeMetrics()
        
        # Solution data
        self.solution = NodeSolution()
        
        # Tree structure
        self.children: List[str] = []  # List of child node IDs
        self.sibling_ids: List[str] = []  # List of sibling node IDs
        
        # Exploration tracking
        self.creation_time = time.time()
        self.last_visited = time.time()
        self.is_expanded = False
        self.is_terminal = False
        
        # Stopping conditions tracking
        self.consecutive_debug_attempts = 0
        self.consecutive_failed_improvements = 0
        
        logger.debug(f"Created MCTS node {node_id} at depth {depth}")
    
    def add_child(self, child_id: str):
        """Add a child node"""
        if child_id not in self.children:
            self.children.append(child_id)
            logger.debug(f"Added child {child_id} to node {self.node_id}")
    
    def add_sibling(self, sibling_id: str):
        """Add a sibling node"""
        if sibling_id not in self.sibling_ids:
            self.sibling_ids.append(sibling_id)
            logger.debug(f"Added sibling {sibling_id} to node {self.node_id}")
    
    def update_state(self, new_state: NodeState):
        """Update the node state"""
        self.state = new_state
        logger.debug(f"Updated node {self.node_id} state to {new_state.value}")
    
    def update_solution(self, 
                       code: str = None,
                       performance_score: float = None,
                       execution_logs: List[str] = None,
                       error_messages: List[str] = None,
                       metadata: Dict[str, Any] = None):
        """Update the node solution"""
        if code is not None:
            self.solution.code = code
        if performance_score is not None:
            self.solution.performance_score = performance_score
        if execution_logs is not None:
            self.solution.execution_logs = execution_logs
        if error_messages is not None:
            self.solution.error_messages = error_messages
        if metadata is not None:
            self.solution.metadata.update(metadata)
        
        logger.debug(f"Updated solution for node {self.node_id}")
    
    def calculate_reward(self, 
                        improvement_threshold: float,
                        best_score_so_far: float) -> float:
        """
        Calculate reward for this node based on the paper's reward function
        
        Args:
            improvement_threshold: Threshold for considering improvement significant
            best_score_so_far: Best performance score observed so far
            
        Returns:
            Calculated reward value
        """
        # Check if node has defects (errors)
        if self.solution.has_errors():
            return -1.0
        
        reward = 0.0
        
        # Quality reward: improvement over best score
        if self.solution.performance_score > best_score_so_far:
            reward += 1.0
        
        # Debugging reward: successful error resolution
        if (self.action_type == ActionType.DEBUG and 
            self.parent_id and 
            not self.solution.has_errors()):
            reward += 1.0
        
        # Structural improvement reward: successful improvement completion
        if (self.action_type == ActionType.IMPROVE and
            self.solution.performance_score > best_score_so_far + improvement_threshold):
            reward += 1.0
        
        return reward
    
    def should_terminate(self, 
                        max_failed_improvements: int,
                        max_debug_depth: int) -> bool:
        """
        Check if this node should be terminated based on stopping conditions
        
        Args:
            max_failed_improvements: Maximum allowed failed improvements
            max_debug_depth: Maximum allowed debug depth
            
        Returns:
            True if node should be terminated
        """
        # Improvement-based termination
        if self.consecutive_failed_improvements >= max_failed_improvements:
            logger.debug(f"Node {self.node_id} terminated due to max failed improvements")
            return True
        
        # Debug depth constraint
        if self.consecutive_debug_attempts >= max_debug_depth:
            logger.debug(f"Node {self.node_id} terminated due to max debug depth")
            return True
        
        return False
    
    def get_next_action(self) -> Optional[ActionType]:
        """
        Determine the next action to take based on current state
        
        Returns:
            Next action type or None if no action should be taken
        """
        # If no solution exists, draft one
        if not self.solution.code.strip():
            return ActionType.DRAFT
        
        # If solution has errors, debug it
        if self.solution.has_errors():
            return ActionType.DEBUG
        
        # If solution is valid but can be improved, improve it
        if self.solution.is_valid():
            return ActionType.IMPROVE
        
        return None
    
    def mark_visited(self):
        """Mark the node as visited"""
        self.last_visited = time.time()
        self.metrics.visit_count += 1
    
    def mark_expanded(self):
        """Mark the node as expanded"""
        self.is_expanded = True
    
    def mark_terminal(self):
        """Mark the node as terminal"""
        self.is_terminal = True
        self.state = NodeState.TERMINAL
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the node"""
        return {
            'node_id': self.node_id,
            'parent_id': self.parent_id,
            'action_type': self.action_type.value if self.action_type else None,
            'depth': self.depth,
            'state': self.state.value,
            'metrics': {
                'visit_count': self.metrics.visit_count,
                'total_reward': self.metrics.total_reward,
                'average_reward': self.metrics.average_reward,
                'best_reward': self.metrics.best_reward
            },
            'solution': {
                'has_code': bool(self.solution.code.strip()),
                'is_valid': self.solution.is_valid(),
                'has_errors': self.solution.has_errors(),
                'performance_score': self.solution.performance_score,
                'error_count': len(self.solution.error_messages)
            },
            'tree_info': {
                'children_count': len(self.children),
                'siblings_count': len(self.sibling_ids),
                'is_expanded': self.is_expanded,
                'is_terminal': self.is_terminal
            },
            'timing': {
                'creation_time': self.creation_time,
                'last_visited': self.last_visited,
                'age': time.time() - self.creation_time
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'parent_id': self.parent_id,
            'action_type': self.action_type.value if self.action_type else None,
            'depth': self.depth,
            'state': self.state.value,
            'metrics': {
                'visit_count': self.metrics.visit_count,
                'total_reward': self.metrics.total_reward,
                'best_reward': self.metrics.best_reward,
                'average_reward': self.metrics.average_reward,
                'last_improvement': self.metrics.last_improvement,
                'failed_improvements': self.metrics.failed_improvements,
                'debug_attempts': self.metrics.debug_attempts
            },
            'solution': {
                'code': self.solution.code,
                'performance_score': self.solution.performance_score,
                'execution_logs': self.solution.execution_logs,
                'error_messages': self.solution.error_messages,
                'metadata': self.solution.metadata
            },
            'tree_structure': {
                'children': self.children,
                'sibling_ids': self.sibling_ids
            },
            'exploration_info': {
                'creation_time': self.creation_time,
                'last_visited': self.last_visited,
                'is_expanded': self.is_expanded,
                'is_terminal': self.is_terminal,
                'consecutive_debug_attempts': self.consecutive_debug_attempts,
                'consecutive_failed_improvements': self.consecutive_failed_improvements
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCTSNode':
        """Create node from dictionary"""
        node = cls(
            node_id=data['node_id'],
            parent_id=data['parent_id'],
            action_type=ActionType(data['action_type']) if data['action_type'] else None,
            depth=data['depth']
        )
        
        # Restore metrics
        metrics_data = data['metrics']
        node.metrics.visit_count = metrics_data['visit_count']
        node.metrics.total_reward = metrics_data['total_reward']
        node.metrics.best_reward = metrics_data['best_reward']
        node.metrics.average_reward = metrics_data['average_reward']
        node.metrics.last_improvement = metrics_data['last_improvement']
        node.metrics.failed_improvements = metrics_data['failed_improvements']
        node.metrics.debug_attempts = metrics_data['debug_attempts']
        
        # Restore solution
        solution_data = data['solution']
        node.solution.code = solution_data['code']
        node.solution.performance_score = solution_data['performance_score']
        node.solution.execution_logs = solution_data['execution_logs']
        node.solution.error_messages = solution_data['error_messages']
        node.solution.metadata = solution_data['metadata']
        
        # Restore tree structure
        tree_data = data['tree_structure']
        node.children = tree_data['children']
        node.sibling_ids = tree_data['sibling_ids']
        
        # Restore exploration info
        exploration_data = data['exploration_info']
        node.creation_time = exploration_data['creation_time']
        node.last_visited = exploration_data['last_visited']
        node.is_expanded = exploration_data['is_expanded']
        node.is_terminal = exploration_data['is_terminal']
        node.consecutive_debug_attempts = exploration_data['consecutive_debug_attempts']
        node.consecutive_failed_improvements = exploration_data['consecutive_failed_improvements']
        
        # Restore state
        node.state = NodeState(data['state'])
        
        return node 