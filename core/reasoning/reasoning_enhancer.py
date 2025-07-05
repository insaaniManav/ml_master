"""
Reasoning Enhancer Implementation

This module implements advanced reasoning enhancement techniques for LLMs,
including chain-of-thought prompting, self-verification, and context-aware
reasoning strategies.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from .llm_agent import LLMAgent


class ReasoningEnhancer:
    """
    Reasoning Enhancer for ML-Master
    
    Enhances LLM reasoning capabilities through advanced prompting techniques,
    self-verification, and context-aware reasoning strategies.
    """
    
    def __init__(self, config: Dict[str, Any], llm_agent: LLMAgent):
        """
        Initialize the reasoning enhancer
        
        Args:
            config: Configuration dictionary
            llm_agent: LLM agent instance
        """
        self.config = config
        self.llm_agent = llm_agent
        
        # Enhancement settings
        self.enable_chain_of_thought = config.get('enable_chain_of_thought', True)
        self.enable_self_verification = config.get('enable_self_verification', True)
        self.enable_context_awareness = config.get('enable_context_awareness', True)
        self.max_reasoning_steps = config.get('max_reasoning_steps', 10)
        self.verification_threshold = config.get('verification_threshold', 0.8)
        
        # Prompt templates
        self.reasoning_templates = self._load_reasoning_templates()
        
        logger.info("Initialized ReasoningEnhancer")
    
    def enhance_reasoning(self,
                         task_description: str,
                         action_type: str,
                         current_code: str = "",
                         error_messages: List[str] = None,
                         contextual_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance reasoning process with advanced techniques
        
        Args:
            task_description: Description of the ML task
            action_type: Type of action (draft, debug, improve)
            current_code: Current code if any
            error_messages: List of error messages if any
            contextual_memory: Contextual memory from exploration
            
        Returns:
            Enhanced reasoning results
        """
        logger.debug(f"Enhancing reasoning for {action_type} action")
        
        # Step 1: Initial reasoning with chain-of-thought
        initial_reasoning = self._perform_chain_of_thought(
            task_description, action_type, current_code, error_messages, contextual_memory
        )
        
        # Step 2: Self-verification if enabled
        verification_result = None
        if self.enable_self_verification:
            verification_result = self._perform_self_verification(
                initial_reasoning, task_description, action_type
            )
        
        # Step 3: Context-aware refinement
        refined_reasoning = None
        if self.enable_context_awareness and contextual_memory:
            refined_reasoning = self._perform_context_aware_refinement(
                initial_reasoning, contextual_memory, action_type
            )
        
        # Step 4: Generate final code
        final_code = self._generate_code_from_reasoning(
            refined_reasoning or initial_reasoning, action_type
        )
        
        # Compile results
        results = {
            'reasoning_process': initial_reasoning.get('reasoning_process', ''),
            'verification_result': verification_result,
            'refined_reasoning': refined_reasoning,
            'code': final_code,
            'insights': self._extract_insights(initial_reasoning),
            'confidence_score': self._calculate_confidence(verification_result),
            'context_used': self._summarize_context_usage(contextual_memory)
        }
        
        logger.debug(f"Reasoning enhancement completed for {action_type}")
        return results
    
    def _perform_chain_of_thought(self,
                                 task_description: str,
                                 action_type: str,
                                 current_code: str,
                                 error_messages: List[str],
                                 contextual_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform chain-of-thought reasoning
        
        Args:
            task_description: Task description
            action_type: Action type
            current_code: Current code
            error_messages: Error messages
            contextual_memory: Contextual memory
            
        Returns:
            Chain-of-thought reasoning results
        """
        # Build prompt based on action type
        if action_type == "draft":
            prompt = self._build_draft_prompt(task_description, contextual_memory)
        elif action_type == "debug":
            prompt = self._build_debug_prompt(task_description, current_code, error_messages, contextual_memory)
        elif action_type == "improve":
            prompt = self._build_improve_prompt(task_description, current_code, contextual_memory)
        else:
            prompt = self._build_generic_prompt(task_description, action_type, contextual_memory)
        
        # Perform reasoning with step-by-step thinking
        reasoning_result = self.llm_agent.generate_with_thinking(
            prompt=prompt,
            max_steps=self.max_reasoning_steps,
            temperature=0.1  # Low temperature for consistent reasoning
        )
        
        return {
            'reasoning_process': reasoning_result.get('thinking_process', ''),
            'final_answer': reasoning_result.get('answer', ''),
            'steps_taken': reasoning_result.get('steps', []),
            'confidence': reasoning_result.get('confidence', 0.0)
        }
    
    def _perform_self_verification(self,
                                  reasoning_result: Dict[str, Any],
                                  task_description: str,
                                  action_type: str) -> Dict[str, Any]:
        """
        Perform self-verification of reasoning
        
        Args:
            reasoning_result: Results from initial reasoning
            task_description: Task description
            action_type: Action type
            
        Returns:
            Verification results
        """
        verification_prompt = self._build_verification_prompt(
            reasoning_result, task_description, action_type
        )
        
        verification_response = self.llm_agent.generate(
            prompt=verification_prompt,
            temperature=0.0  # Deterministic verification
        )
        
        # Parse verification response
        verification_score = self._parse_verification_score(verification_response)
        verification_feedback = self._parse_verification_feedback(verification_response)
        
        return {
            'verification_score': verification_score,
            'verification_feedback': verification_feedback,
            'is_verified': verification_score >= self.verification_threshold,
            'verification_response': verification_response
        }
    
    def _perform_context_aware_refinement(self,
                                         initial_reasoning: Dict[str, Any],
                                         contextual_memory: Dict[str, Any],
                                         action_type: str) -> Dict[str, Any]:
        """
        Perform context-aware refinement of reasoning
        
        Args:
            initial_reasoning: Initial reasoning results
            contextual_memory: Contextual memory
            action_type: Action type
            
        Returns:
            Refined reasoning results
        """
        # Extract relevant context
        relevant_context = self._extract_relevant_context(
            contextual_memory, action_type
        )
        
        if not relevant_context:
            return initial_reasoning
        
        # Build refinement prompt
        refinement_prompt = self._build_refinement_prompt(
            initial_reasoning, relevant_context, action_type
        )
        
        # Perform refinement
        refinement_response = self.llm_agent.generate(
            prompt=refinement_prompt,
            temperature=0.2
        )
        
        return {
            'original_reasoning': initial_reasoning,
            'refined_reasoning': refinement_response,
            'context_used': relevant_context,
            'refinement_changes': self._identify_refinement_changes(
                initial_reasoning, refinement_response
            )
        }
    
    def _generate_code_from_reasoning(self,
                                     reasoning_result: Dict[str, Any],
                                     action_type: str) -> str:
        """
        Generate code from reasoning results
        
        Args:
            reasoning_result: Reasoning results
            action_type: Action type
            
        Returns:
            Generated code
        """
        # Extract reasoning content
        reasoning_content = reasoning_result.get('reasoning_process', '')
        if not reasoning_content:
            reasoning_content = reasoning_result.get('final_answer', '')
        
        # Build code generation prompt
        code_prompt = self._build_code_generation_prompt(
            reasoning_content, action_type
        )
        
        # Generate code
        code_response = self.llm_agent.generate(
            prompt=code_prompt,
            temperature=0.1
        )
        
        # Extract code from response
        code = self._extract_code_from_response(code_response)
        
        return code
    
    def _build_draft_prompt(self, task_description: str, contextual_memory: Dict[str, Any]) -> str:
        """Build prompt for draft action"""
        template = self.reasoning_templates.get('draft', '')
        
        # Add context if available
        context_str = ""
        if contextual_memory:
            context_str = self._format_context_for_prompt(contextual_memory)
        
        return template.format(
            task_description=task_description,
            context=context_str
        )
    
    def _build_debug_prompt(self, task_description: str, current_code: str, 
                           error_messages: List[str], contextual_memory: Dict[str, Any]) -> str:
        """Build prompt for debug action"""
        template = self.reasoning_templates.get('debug', '')
        
        error_str = "\n".join(error_messages) if error_messages else "No specific errors"
        context_str = self._format_context_for_prompt(contextual_memory) if contextual_memory else ""
        
        return template.format(
            task_description=task_description,
            current_code=current_code,
            errors=error_str,
            context=context_str
        )
    
    def _build_improve_prompt(self, task_description: str, current_code: str, 
                             contextual_memory: Dict[str, Any]) -> str:
        """Build prompt for improve action"""
        template = self.reasoning_templates.get('improve', '')
        
        context_str = self._format_context_for_prompt(contextual_memory) if contextual_memory else ""
        
        return template.format(
            task_description=task_description,
            current_code=current_code,
            context=context_str
        )
    
    def _build_verification_prompt(self, reasoning_result: Dict[str, Any], 
                                  task_description: str, action_type: str) -> str:
        """Build prompt for self-verification"""
        template = self.reasoning_templates.get('verification', '')
        
        return template.format(
            task_description=task_description,
            action_type=action_type,
            reasoning_process=reasoning_result.get('reasoning_process', ''),
            final_answer=reasoning_result.get('final_answer', '')
        )
    
    def _build_refinement_prompt(self, initial_reasoning: Dict[str, Any], 
                                relevant_context: Dict[str, Any], action_type: str) -> str:
        """Build prompt for context-aware refinement"""
        template = self.reasoning_templates.get('refinement', '')
        
        return template.format(
            action_type=action_type,
            initial_reasoning=initial_reasoning.get('reasoning_process', ''),
            context=json.dumps(relevant_context, indent=2)
        )
    
    def _build_code_generation_prompt(self, reasoning_content: str, action_type: str) -> str:
        """Build prompt for code generation"""
        template = self.reasoning_templates.get('code_generation', '')
        
        return template.format(
            action_type=action_type,
            reasoning_content=reasoning_content
        )
    
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load reasoning prompt templates"""
        return {
            'draft': """You are an expert machine learning engineer. Your task is to create an initial solution for the following machine learning problem:

Task: {task_description}

{context}

Please think through this step by step:
1. What type of machine learning problem is this?
2. What are the key requirements and constraints?
3. What approach would be most suitable?
4. What libraries and techniques should I use?
5. How should I structure the solution?

Provide your reasoning and then generate a complete, runnable Python solution.""",
            
            'debug': """You are debugging a machine learning solution. Here's the current situation:

Task: {task_description}

Current Code:
{current_code}

Errors:
{errors}

{context}

Please analyze the errors and think through the debugging process:
1. What are the root causes of these errors?
2. What specific fixes are needed?
3. How can I prevent similar errors in the future?
4. What improvements can I make to the code structure?

Provide your debugging reasoning and then generate the corrected code.""",
            
            'improve': """You are improving an existing machine learning solution. Here's the current state:

Task: {task_description}

Current Code:
{current_code}

{context}

Please analyze the current solution and think through improvements:
1. What are the current limitations of this solution?
2. What performance bottlenecks exist?
3. What advanced techniques could be applied?
4. How can I make the solution more robust and efficient?
5. What additional features or optimizations would be beneficial?

Provide your improvement reasoning and then generate the enhanced code.""",
            
            'verification': """Please verify the following reasoning for a {action_type} action:

Task: {task_description}

Reasoning Process:
{reasoning_process}

Final Answer:
{final_answer}

Rate the quality of this reasoning on a scale of 0-1, where:
- 0.0-0.3: Poor reasoning with significant flaws
- 0.4-0.6: Adequate reasoning with some issues
- 0.7-0.8: Good reasoning with minor issues
- 0.9-1.0: Excellent reasoning

Provide your score and brief feedback explaining your rating.""",
            
            'refinement': """Refine the following reasoning for a {action_type} action using the provided context:

Original Reasoning:
{initial_reasoning}

Context:
{context}

Please refine the reasoning by incorporating relevant insights from the context while maintaining the core logic. Focus on:
1. Incorporating successful patterns from similar tasks
2. Avoiding approaches that failed in the past
3. Leveraging specific techniques that worked well
4. Addressing any gaps in the original reasoning

Provide the refined reasoning.""",
            
            'code_generation': """Based on the following reasoning for a {action_type} action, generate complete, runnable Python code:

Reasoning:
{reasoning_content}

Generate clean, well-documented Python code that implements the solution described in the reasoning. Include all necessary imports and ensure the code is ready to run."""
        }
    
    def _format_context_for_prompt(self, contextual_memory: Dict[str, Any]) -> str:
        """Format contextual memory for prompt inclusion"""
        if not contextual_memory:
            return ""
        
        context_parts = []
        
        # Add recent insights
        if 'recent_insights' in contextual_memory:
            insights = contextual_memory['recent_insights']
            if insights:
                context_parts.append("Recent Insights:")
                for insight in insights[:3]:  # Limit to 3 most recent
                    context_parts.append(f"- {insight}")
        
        # Add successful patterns
        if 'successful_patterns' in contextual_memory:
            patterns = contextual_memory['successful_patterns']
            if patterns:
                context_parts.append("Successful Patterns:")
                for pattern in patterns[:2]:  # Limit to 2 patterns
                    context_parts.append(f"- {pattern}")
        
        # Add error patterns to avoid
        if 'error_patterns' in contextual_memory:
            errors = contextual_memory['error_patterns']
            if errors:
                context_parts.append("Error Patterns to Avoid:")
                for error in errors[:2]:  # Limit to 2 errors
                    context_parts.append(f"- {error}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _extract_relevant_context(self, contextual_memory: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        """Extract context relevant to the current action type"""
        if not contextual_memory:
            return {}
        
        relevant_context = {}
        
        # Filter context based on action type
        if action_type == "draft":
            relevant_context.update({
                'successful_patterns': contextual_memory.get('successful_patterns', []),
                'task_similarities': contextual_memory.get('task_similarities', [])
            })
        elif action_type == "debug":
            relevant_context.update({
                'error_patterns': contextual_memory.get('error_patterns', []),
                'debugging_strategies': contextual_memory.get('debugging_strategies', [])
            })
        elif action_type == "improve":
            relevant_context.update({
                'improvement_strategies': contextual_memory.get('improvement_strategies', []),
                'performance_insights': contextual_memory.get('performance_insights', [])
            })
        
        return relevant_context
    
    def _parse_verification_score(self, verification_response: str) -> float:
        """Parse verification score from response"""
        try:
            # Look for score in the response
            score_match = re.search(r'(\d+\.?\d*)', verification_response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
        except (ValueError, AttributeError):
            pass
        
        # Default score if parsing fails
        return 0.5
    
    def _parse_verification_feedback(self, verification_response: str) -> str:
        """Parse verification feedback from response"""
        # Extract feedback after the score
        lines = verification_response.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'\d+\.?\d*', line):
                # Return everything after the score line
                return '\n'.join(lines[i+1:]).strip()
        
        return verification_response.strip()
    
    def _extract_insights(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from reasoning results"""
        reasoning_process = reasoning_result.get('reasoning_process', '')
        
        insights = {
            'key_decisions': [],
            'assumptions_made': [],
            'alternative_approaches': [],
            'potential_risks': []
        }
        
        # Simple extraction based on keywords
        lines = reasoning_process.split('\n')
        for line in lines:
            line = line.strip().lower()
            if 'decision' in line or 'choose' in line:
                insights['key_decisions'].append(line)
            elif 'assume' in line or 'assumption' in line:
                insights['assumptions_made'].append(line)
            elif 'alternative' in line or 'could also' in line:
                insights['alternative_approaches'].append(line)
            elif 'risk' in line or 'potential issue' in line:
                insights['potential_risks'].append(line)
        
        return insights
    
    def _calculate_confidence(self, verification_result: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score based on verification"""
        if not verification_result:
            return 0.5
        
        verification_score = verification_result.get('verification_score', 0.5)
        is_verified = verification_result.get('is_verified', False)
        
        # Boost confidence if verified
        if is_verified:
            return min(verification_score * 1.2, 1.0)
        else:
            return verification_score * 0.8
    
    def _summarize_context_usage(self, contextual_memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize how context was used"""
        if not contextual_memory:
            return {'context_used': False, 'context_types': []}
        
        context_types = []
        if 'recent_insights' in contextual_memory:
            context_types.append('recent_insights')
        if 'successful_patterns' in contextual_memory:
            context_types.append('successful_patterns')
        if 'error_patterns' in contextual_memory:
            context_types.append('error_patterns')
        
        return {
            'context_used': True,
            'context_types': context_types,
            'context_size': len(contextual_memory)
        }
    
    def _identify_refinement_changes(self, original: Dict[str, Any], refined: str) -> List[str]:
        """Identify changes made during refinement"""
        changes = []
        
        # Simple change detection
        original_text = original.get('reasoning_process', '')
        if original_text != refined:
            changes.append("Reasoning process was refined based on context")
        
        return changes
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response"""
        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for code without markdown
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code = True
            elif in_code and line.strip() and not line.startswith('#'):
                code_lines.append(line)
            elif in_code and not line.strip():
                continue
            else:
                in_code = False
        
        return '\n'.join(code_lines) if code_lines else response.strip() 