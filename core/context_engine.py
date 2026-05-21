"""
Context Optimization Engine (COE) — Enterprise LLM Context Window Optimizer
Reduces token costs and attention dilution by intelligently projecting and compressing state.
"""

import ast
import yaml
import os
from typing import Dict, Any, List


class ContextOptimizationEngine:
    """
    Middleware for optimizing state dictionaries before LLM ingestion.
    Uses dot-notation key projection and YAML serialization for token efficiency.
    """

    @staticmethod
    def project_state(state: dict, required_keys: List[str]) -> dict:
        """
        Extract a projection of state containing only specified keys (dot-notation).
        
        Args:
            state: The full state dictionary (potentially deeply nested)
            required_keys: List of dot-notation paths (e.g., ["brd.title", "adr.decisions"])
        
        Returns:
            New dictionary containing only the requested keys and their values
        
        Example:
            state = {"brd": {"title": "Project X", "objectives": [...]}, "prd": {...}}
            result = project_state(state, ["brd.title"])
            # Returns: {"brd": {"title": "Project X"}}
        """
        result = {}

        for key_path in required_keys:
            parts = key_path.split(".")
            current = state

            # Navigate through nested structure
            try:
                for part in parts[:-1]:
                    current = current[part]

                # Get the final value
                final_value = current[parts[-1]]

                # Build nested result structure
                if len(parts) == 1:
                    # Top-level key
                    result[parts[0]] = final_value
                else:
                    # Nested key — rebuild hierarchy
                    nested = result
                    for part in parts[:-1]:
                        if part not in nested:
                            nested[part] = {}
                        nested = nested[part]
                    nested[parts[-1]] = final_value
            except (KeyError, TypeError):
                # Key path not found in state — skip gracefully
                continue

        return result

    @staticmethod
    def to_llm_yaml(data: dict) -> str:
        """
        Convert a dictionary to YAML string optimized for LLM consumption.
        YAML is significantly more token-efficient than JSON for LLM prompts.
        
        Args:
            data: Dictionary to convert
        
        Returns:
            YAML string with:
            - default_flow_style=False (preserves readability)
            - sort_keys=False (maintains insertion order)
        
        Example:
            data = {"title": "Project X", "items": [1, 2, 3]}
            yaml_str = to_llm_yaml(data)
            # Returns YAML representation
        """
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )

    @staticmethod
    def optimize_for_prompt(state: dict, required_keys: List[str]) -> str:
        """
        End-to-end context optimization: project state and serialize to YAML.
        
        This is the primary integration point — use this in prompts instead of
        json.dumps(state) or manual dictionary slicing.
        
        Args:
            state: Full state dictionary
            required_keys: Dot-notation keys to include
        
        Returns:
            Optimized YAML string ready for prompt injection
        
        Example:
            optimized = optimize_for_prompt(state, ["brd.title", "brd.kpis"])
            prompt = f"Context:\n{optimized}\n\nYour task..."
        """
        projected = ContextOptimizationEngine.project_state(state, required_keys)
        return ContextOptimizationEngine.to_llm_yaml(projected)

    @staticmethod
    def prune_python_code_ast(source_code: str, target_node_name: str) -> str:
        """
        Extract source code for a specific function or class definition using AST.
        
        Parses Python source code and returns ONLY the code segment for the
        named ClassDef or FunctionDef. Useful for token-efficient code context.
        
        Args:
            source_code: Full Python source code string
            target_node_name: Name of class or function to extract (e.g., "MyClass", "helper_func")
        
        Returns:
            Source code of the target node, or original source_code if not found
        
        Example:
            code = '''
            def helper():
                return 1
            
            def main():
                return helper() + 1
            '''
            result = prune_python_code_ast(code, "main")
            # Returns the source code of main() function only
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If parsing fails, return original source
            return source_code

        # Search for matching ClassDef or FunctionDef
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == target_node_name:
                    # Use ast.get_source_segment to extract source
                    try:
                        segment = ast.get_source_segment(source_code, node)
                        if segment:
                            return segment
                    except (AttributeError, TypeError):
                        # Fallback if get_source_segment unavailable
                        pass

        # Node not found — return original source
        return source_code

    @staticmethod
    def semantic_search_limited(query: str, top_k: int = 3) -> list:
        """
        Find most relevant files and symbols for the change request.
        Limited to top 3 results (reduced from 10) for Context Precision.
        
        Args:
            query: Search query string
            top_k: Maximum results to return (default: 3 for context efficiency)
        
        Returns:
            List of top matches with repo, file path, symbol info, and relevance score
        """
        try:
            from core.db_clients import qdrant_client as qdrant
        except ImportError:
            return []

        try:
            from core.embeddings import get_embedder
            embedder = get_embedder()
            query_vector = embedder.encode(query).tolist()

            results = qdrant.query_points(
                collection_name="code_embeddings",
                query=query_vector,
                limit=top_k,
                with_payload=True
            )

            hits = []
            for r in results.points:
                hits.append({
                    "repo_name": r.payload.get("repo_name"),
                    "file_path": r.payload.get("file_path"),
                    "symbol_name": r.payload.get("symbol_name"),
                    "symbol_type": r.payload.get("symbol_type"),
                    "score": round(r.score, 4)
                })

            return hits
        except Exception:
            return []


# Global singleton instance for use throughout the application
coe = ContextOptimizationEngine()
