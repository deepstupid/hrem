"""Scientific timing manager that extends the existing timing utilities."""

from typing import Dict, Any, Optional
from demo_timing_utils import EnhancedTimingManager

class ScientificTimingManager(EnhancedTimingManager):
    """Extended timing manager for scientific discovery metrics."""
    
    def __init__(self):
        super().__init__()
        self.discovery_metrics: Dict[str, list] = {}
        
    def record_discovery_timing(self, activity: str, elapsed_time: float, insights_generated: int):
        """
        Record timing with scientific discovery metrics.
        
        Args:
            activity: Name of the activity
            elapsed_time: Time elapsed for the activity
            insights_generated: Number of insights generated during the activity
        """
        # Record standard timing
        self.record_timing(activity, elapsed_time)
        
        # Record discovery-specific metrics
        self.record_metric(f"discovery_insights_{activity}", insights_generated)
        self.record_metric(f"discovery_efficiency_{activity}", 
                          insights_generated / elapsed_time if elapsed_time > 0 else 0)
        
        # Store in discovery metrics
        if f"discovery_{activity}" not in self.discovery_metrics:
            self.discovery_metrics[f"discovery_{activity}"] = []
        self.discovery_metrics[f"discovery_{activity}"].append({
            'elapsed_time': elapsed_time,
            'insights_generated': insights_generated,
            'timestamp': self.current_iteration
        })
    
    def predict_discovery_value(self, time_investment: float) -> Dict[str, Any]:
        """
        Predict scientific value of time investment.
        
        Args:
            time_investment: Amount of time to invest
            
        Returns:
            Dictionary with expected insights and value metrics
        """
        # This is a simplified prediction model
        # In practice, this would be based on historical data and machine learning
        
        # Calculate average insights per second from historical data
        avg_insights_per_second = 0.1  # Placeholder value
        
        # Get recent discovery efficiency
        recent_efficiency = self.get_metric_stats("discovery_efficiency")
        if recent_efficiency and recent_efficiency.get('avg'):
            avg_insights_per_second = recent_efficiency['avg']
        
        expected_insights = time_investment * avg_insights_per_second
        discovery_value = min(1.0, expected_insights / 5.0)  # Normalize to 0-1 scale
        
        return {
            'expected_insights': expected_insights,
            'discovery_value': discovery_value,
            'confidence': 0.7  # Simplified confidence measure
        }
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics on discovery metrics."""
        stats = {}
        for metric_name in self.discovery_metrics.keys():
            if self.discovery_metrics[metric_name]:
                values = self.discovery_metrics[metric_name]
                insights = [v['insights_generated'] for v in values]
                times = [v['elapsed_time'] for v in values]
                
                stats[metric_name] = {
                    'total_insights': sum(insights),
                    'avg_insights_per_run': sum(insights) / len(insights) if insights else 0,
                    'total_time': sum(times),
                    'avg_time_per_run': sum(times) / len(times) if times else 0
                }
        
        return stats