"""
Cost tracking utilities for monitoring API costs and token usage.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class TokenUsage:
    """Token usage information for a single API call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class APICall:
    """Information about a single API call."""
    timestamp: str
    model: str
    tokens: TokenUsage
    cost_usd: float
    operation_type: str  # e.g., "question_generation", "answer_generation", "simulation", etc.
    section_id: Optional[str] = None
    chapter_id: Optional[str] = None


@dataclass
class CostSummary:
    """Summary of all costs and token usage."""
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    call_count: int
    calls_by_operation: Dict[str, int]
    calls_by_model: Dict[str, int]


class CostTracker:
    """Tracks API costs and token usage across the application."""
    
    # OpenAI pricing as of 2024 (per 1M tokens)
    PRICING = {
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4": {"input": 2.50, "output": 10.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize cost tracker.
        
        Args:
            log_file: Optional path to log file. If None, uses default naming.
        """
        self.calls: List[APICall] = []
        self.log_file = log_file or f"cost_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation_type: str,
        section_id: Optional[str] = None,
        chapter_id: Optional[str] = None
    ) -> APICall:
        """Track a single API call.
        
        Args:
            model: The model used for the call
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_type: Type of operation (e.g., "question_generation")
            section_id: Optional section identifier
            chapter_id: Optional chapter identifier
            
        Returns:
            APICall object with cost information
        """
        # Calculate cost
        cost_per_1m = self.PRICING.get(model, {"input": 0.15, "output": 0.60})
        input_cost = (input_tokens / 1_000_000) * cost_per_1m["input"]
        output_cost = (output_tokens / 1_000_000) * cost_per_1m["output"]
        total_cost = input_cost + output_cost
        
        # Create call record
        call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            tokens=TokenUsage(input_tokens, output_tokens, input_tokens + output_tokens),
            cost_usd=total_cost,
            operation_type=operation_type,
            section_id=section_id,
            chapter_id=chapter_id
        )
        
        self.calls.append(call)
        
        # Log the call
        logger.info(f"API Call - Model: {model}, Tokens: {input_tokens + output_tokens} "
                   f"(in: {input_tokens}, out: {output_tokens}), Cost: ${total_cost:.6f}, "
                   f"Operation: {operation_type}")
        
        return call
    
    def get_summary(self) -> CostSummary:
        """Get summary of all tracked costs and usage."""
        if not self.calls:
            return CostSummary(0, 0, 0, 0.0, 0, {}, {})
        
        total_input = sum(call.tokens.input_tokens for call in self.calls)
        total_output = sum(call.tokens.output_tokens for call in self.calls)
        total_tokens = total_input + total_output
        total_cost = sum(call.cost_usd for call in self.calls)
        
        # Count by operation type
        calls_by_operation = {}
        for call in self.calls:
            calls_by_operation[call.operation_type] = calls_by_operation.get(call.operation_type, 0) + 1
        
        # Count by model
        calls_by_model = {}
        for call in self.calls:
            calls_by_model[call.model] = calls_by_model.get(call.model, 0) + 1
        
        return CostSummary(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            call_count=len(self.calls),
            calls_by_operation=calls_by_operation,
            calls_by_model=calls_by_model
        )
    
    def save_log(self, filepath: Optional[str] = None) -> str:
        """Save detailed call log to JSONL file.
        
        Args:
            filepath: Optional custom filepath. If None, uses the default log file.
            
        Returns:
            Path to the saved log file.
        """
        log_path = filepath or self.log_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        
        with open(log_path, 'w') as f:
            for call in self.calls:
                f.write(json.dumps(asdict(call)) + '\n')
        
        logger.info(f"Cost tracking log saved to: {log_path}")
        return log_path
    
    def print_summary(self):
        """Print a formatted summary of costs and usage."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("COST TRACKING SUMMARY")
        print("="*60)
        print(f"Total API Calls: {summary.call_count}")
        print(f"Total Input Tokens: {summary.total_input_tokens:,}")
        print(f"Total Output Tokens: {summary.total_output_tokens:,}")
        print(f"Total Tokens: {summary.total_tokens:,}")
        print(f"Total Cost: ${summary.total_cost_usd:.6f}")
        print()
        
        if summary.calls_by_operation:
            print("Calls by Operation Type:")
            for operation, count in summary.calls_by_operation.items():
                print(f"  {operation}: {count}")
            print()
        
        if summary.calls_by_model:
            print("Calls by Model:")
            for model, count in summary.calls_by_model.items():
                print(f"  {model}: {count}")
            print()
        
        print("="*60)


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_tracker
    _global_tracker = None
