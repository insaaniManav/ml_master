"""
Context Manager Implementation

This module provides context management utilities for adaptive memory and
reasoning integration in ML-Master. It helps curate, summarize, and select
relevant context for steerable reasoning and exploration.
"""

from typing import Dict, List, Any
from loguru import logger

class ContextManager:
    """
    Context Manager for ML-Master
    
    Handles the selection, summarization, and curation of contextual memory
    for use in reasoning and exploration modules.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_context_entries = self.config.get('max_context_entries', 5)
        logger.info("Initialized ContextManager")

    def select_context(self, memory_entries: List[Dict[str, Any]],
                      node_depth: int = 0,
                      sibling_entries: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Select and summarize relevant context for reasoning.
        Args:
            memory_entries: List of memory entries (dicts)
            node_depth: Current node depth in the search tree
            sibling_entries: List of sibling node memory entries
        Returns:
            Curated context dictionary
        """
        context = {}
        # Select most recent or most relevant entries
        context['recent_insights'] = [e.get('insight') for e in memory_entries[-self.max_context_entries:]]
        if sibling_entries:
            context['sibling_insights'] = [e.get('insight') for e in sibling_entries[-self.max_context_entries:]]
        return context

    def summarize_context(self, context: Dict[str, Any]) -> str:
        """
        Summarize context for prompt inclusion.
        Args:
            context: Context dictionary
        Returns:
            String summary
        """
        summary = []
        if 'recent_insights' in context:
            summary.append("Recent Insights:")
            for insight in context['recent_insights']:
                summary.append(f"- {insight}")
        if 'sibling_insights' in context:
            summary.append("Sibling Insights:")
            for insight in context['sibling_insights']:
                summary.append(f"- {insight}")
        return "\n".join(summary) 