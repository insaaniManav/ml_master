"""
Adaptive Memory Implementation

This module implements the adaptive memory mechanism that selectively captures
insights from exploration history and integrates them into reasoning processes.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from loguru import logger


class MemoryType(Enum):
    """Types of memory entries"""
    REASONING_INSIGHT = "reasoning_insight"
    EXECUTION_FEEDBACK = "execution_feedback"
    PERFORMANCE_METRIC = "performance_metric"
    CODE_SNIPPET = "code_snippet"


@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    type: MemoryType
    content: Dict[str, Any]
    timestamp: float
    node_id: str
    parent_node_id: Optional[str]
    sibling_node_ids: List[str]
    relevance_score: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data['type'] = MemoryType(data['type'])
        return cls(**data)


class AdaptiveMemory:
    """
    Adaptive Memory Mechanism
    
    Implements the selective memory capture and utilization system described
    in the ML-Master paper. This class manages the adaptive memory that
    combines insights from exploration trajectories with reasoning processes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adaptive memory system
        
        Args:
            config: Configuration dictionary containing memory settings
        """
        self.config = config
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.max_memory_size = config.get('max_memory_size', 1000)
        self.memory_decay_factor = config.get('memory_decay_factor', 0.95)
        self.insight_threshold = config.get('insight_extraction_threshold', 0.7)
        self.context_window_size = config.get('context_window_size', 10)
        
        # Memory type filters
        self.enabled_memory_types = {
            MemoryType.REASONING_INSIGHT: config.get('reasoning_insights', True),
            MemoryType.EXECUTION_FEEDBACK: config.get('execution_feedback', True),
            MemoryType.PERFORMANCE_METRIC: config.get('performance_metrics', True),
            MemoryType.CODE_SNIPPET: config.get('code_snippets', True)
        }
        
        logger.info(f"Initialized AdaptiveMemory with max_size={self.max_memory_size}")
    
    def add_memory_entry(self, 
                        memory_type: MemoryType,
                        content: Dict[str, Any],
                        node_id: str,
                        parent_node_id: Optional[str] = None,
                        sibling_node_ids: List[str] = None,
                        relevance_score: float = 1.0) -> str:
        """
        Add a new memory entry
        
        Args:
            memory_type: Type of memory entry
            content: Content of the memory entry
            node_id: ID of the exploration node
            parent_node_id: ID of the parent node
            sibling_node_ids: List of sibling node IDs
            relevance_score: Relevance score for this entry
            
        Returns:
            Memory entry ID
        """
        if not self.enabled_memory_types.get(memory_type, False):
            logger.debug(f"Memory type {memory_type} is disabled, skipping")
            return None
            
        if relevance_score < self.insight_threshold:
            logger.debug(f"Relevance score {relevance_score} below threshold, skipping")
            return None
        
        # Generate unique ID
        entry_id = f"{memory_type.value}_{node_id}_{int(time.time() * 1000)}"
        
        # Create memory entry
        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            timestamp=time.time(),
            node_id=node_id,
            parent_node_id=parent_node_id,
            sibling_node_ids=sibling_node_ids or [],
            relevance_score=relevance_score
        )
        
        # Add to memory
        self.memory_entries[entry_id] = entry
        
        # Apply memory management if needed
        self._manage_memory_size()
        
        logger.debug(f"Added memory entry {entry_id} of type {memory_type.value}")
        return entry_id
    
    def get_contextual_memory(self, 
                             current_node_id: str,
                             parent_node_id: Optional[str] = None,
                             sibling_node_ids: List[str] = None) -> Dict[str, Any]:
        """
        Get contextual memory for reasoning
        
        This implements the adaptive memory mechanism described in the paper:
        - Immediate parent node insights
        - Parallel sibling node insights at same exploration depth
        - Selective inclusion based on relevance and recency
        
        Args:
            current_node_id: Current exploration node ID
            parent_node_id: Parent node ID for continuity
            sibling_node_ids: Sibling node IDs for diversity
            
        Returns:
            Contextual memory dictionary
        """
        contextual_memory = {
            'parent_insights': [],
            'sibling_insights': [],
            'execution_feedback': [],
            'performance_metrics': []
        }
        
        # Get parent node insights for continuity
        if parent_node_id:
            parent_entries = self._get_entries_by_node(parent_node_id)
            contextual_memory['parent_insights'] = self._extract_insights(parent_entries)
        
        # Get sibling node insights for diversity
        if sibling_node_ids:
            sibling_entries = []
            for sibling_id in sibling_node_ids:
                sibling_entries.extend(self._get_entries_by_node(sibling_id))
            contextual_memory['sibling_insights'] = self._extract_insights(sibling_entries)
        
        # Get recent execution feedback and performance metrics
        recent_entries = self._get_recent_entries(self.context_window_size)
        contextual_memory['execution_feedback'] = self._extract_execution_feedback(recent_entries)
        contextual_memory['performance_metrics'] = self._extract_performance_metrics(recent_entries)
        
        # Apply relevance filtering and decay
        contextual_memory = self._apply_memory_decay(contextual_memory)
        
        logger.debug(f"Retrieved contextual memory for node {current_node_id}")
        return contextual_memory
    
    def _get_entries_by_node(self, node_id: str) -> List[MemoryEntry]:
        """Get all memory entries for a specific node"""
        return [entry for entry in self.memory_entries.values() 
                if entry.node_id == node_id]
    
    def _get_recent_entries(self, limit: int) -> List[MemoryEntry]:
        """Get most recent memory entries"""
        sorted_entries = sorted(
            self.memory_entries.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_entries[:limit]
    
    def _extract_insights(self, entries: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Extract reasoning insights from memory entries"""
        insights = []
        for entry in entries:
            if entry.type == MemoryType.REASONING_INSIGHT:
                insights.append({
                    'content': entry.content,
                    'relevance_score': entry.relevance_score,
                    'timestamp': entry.timestamp
                })
        return insights
    
    def _extract_execution_feedback(self, entries: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Extract execution feedback from memory entries"""
        feedback = []
        for entry in entries:
            if entry.type == MemoryType.EXECUTION_FEEDBACK:
                feedback.append({
                    'content': entry.content,
                    'relevance_score': entry.relevance_score,
                    'timestamp': entry.timestamp
                })
        return feedback
    
    def _extract_performance_metrics(self, entries: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Extract performance metrics from memory entries"""
        metrics = []
        for entry in entries:
            if entry.type == MemoryType.PERFORMANCE_METRIC:
                metrics.append({
                    'content': entry.content,
                    'relevance_score': entry.relevance_score,
                    'timestamp': entry.timestamp
                })
        return metrics
    
    def _apply_memory_decay(self, contextual_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory decay to reduce relevance of older entries"""
        current_time = time.time()
        
        for category in contextual_memory:
            if isinstance(contextual_memory[category], list):
                for item in contextual_memory[category]:
                    if 'timestamp' in item:
                        time_diff = current_time - item['timestamp']
                        decay_factor = self.memory_decay_factor ** (time_diff / 3600)  # Hourly decay
                        item['relevance_score'] *= decay_factor
        
        return contextual_memory
    
    def _manage_memory_size(self):
        """Manage memory size by removing least relevant entries"""
        if len(self.memory_entries) <= self.max_memory_size:
            return
        
        # Calculate composite score (relevance * recency * access_count)
        current_time = time.time()
        entry_scores = []
        
        for entry_id, entry in self.memory_entries.items():
            time_factor = 1.0 / (1.0 + (current_time - entry.timestamp) / 3600)
            access_factor = 1.0 + entry.access_count * 0.1
            composite_score = entry.relevance_score * time_factor * access_factor
            
            entry_scores.append((entry_id, composite_score))
        
        # Sort by composite score and remove lowest scoring entries
        entry_scores.sort(key=lambda x: x[1])
        entries_to_remove = len(self.memory_entries) - self.max_memory_size
        
        for entry_id, _ in entry_scores[:entries_to_remove]:
            del self.memory_entries[entry_id]
        
        logger.debug(f"Removed {entries_to_remove} memory entries to maintain size limit")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'total_entries': len(self.memory_entries),
            'max_size': self.max_memory_size,
            'memory_usage_percent': len(self.memory_entries) / self.max_memory_size * 100,
            'entries_by_type': {}
        }
        
        for memory_type in MemoryType:
            count = sum(1 for entry in self.memory_entries.values() 
                       if entry.type == memory_type)
            stats['entries_by_type'][memory_type.value] = count
        
        return stats
    
    def clear_memory(self):
        """Clear all memory entries"""
        self.memory_entries.clear()
        logger.info("Cleared all memory entries")
    
    def save_memory(self, filepath: str):
        """Save memory to file"""
        data = {
            'config': self.config,
            'entries': [entry.to_dict() for entry in self.memory_entries.values()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved memory to {filepath}")
    
    def load_memory(self, filepath: str):
        """Load memory from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = data['config']
        self.memory_entries = {}
        
        for entry_data in data['entries']:
            entry = MemoryEntry.from_dict(entry_data)
            self.memory_entries[entry.id] = entry
        
        logger.info(f"Loaded memory from {filepath}") 