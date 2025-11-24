# visualization.py
"""
Fairness Visualization Module
Provides comprehensive visualization tools for fairness analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings

# Set up plotting style
plt.style.use('default')
try:
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Using matplotlib defaults.")


class FairnessVisualizer:
    """Comprehensive visualization tools for fairness analysis."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_fairness_accuracy_tradeoff(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        title: str = "Fairness vs Accuracy Trade-off"
    ) -> plt.Figure:
        """
        Plot fairness vs accuracy trade-off across different models/strategies.
        
        Args:
            results: List of evaluation results from FairnessOptimizer
            save_path: Optional path to save the plot
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data for plotting
        models = [r.get("model_name", "Unknown") for r in results]
        strategies = [r.get("strategy_name", "Unknown") for r in results]
        accuracies = [r.get("overall_accuracy", 0) for r in results]
        
        # Try different keys for disparity
        disparities = []
        for r in results:
            if "accuracy_disparity" in r:
                disparities.append(r["accuracy_disparity"])
            elif "fairness" in r and "disparities" in r["fairness"]:
                disparities.append(r["fairness"]["disparities"].get("accuracy_disparity", 0))
            else:
                disparities.append(0)
        
        # Create color map for models
        unique_models = list(set(models))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_models)))
        color_map = dict(zip(unique_models, colors))
        
        # Plot points
        plotted_models = set()
        for model, strategy, acc, disp in zip(models, strategies, accuracies, disparities):
            label = f"{model}" if model not in plotted_models else ""
            ax.scatter(disp, acc, 
                      c=[color_map[model]], 
                      s=100, 
                      alpha=0.7,
                      label=label)
            
            if model not in plotted_models:
                plotted_models.add(model)
            
            # Annotate with strategy name
            ax.annotate(strategy, (disp, acc), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Accuracy Disparity (lower is more fair)', fontsize=12)
        ax.set_ylabel('Overall Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add ideal region annotation
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
        ax.text(0.02, 0.85, 'Ideal Region\n(High Acc, Low Disparity)', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_group_performance_comparison(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "Performance Across Groups"
    ) -> plt.Figure:
        """
        Plot performance metrics comparison across demographic groups.
        
        Args:
            results: Evaluation results from FairnessOptimizer
            save_path: Optional path to save the plot
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Extract group-wise metrics
        if "fairness" in results and "by_group" in results["fairness"]:
            group_metrics = results["fairness"]["by_group"]
        else:
            # Create dummy data if no group metrics available
            group_metrics = {
                'accuracy': {'group_0': 0.8, 'group_1': 0.75},
                'precision': {'group_0': 0.78, 'group_1': 0.73},
                'recall': {'group_0': 0.82, 'group_1': 0.77},
                'selection_rate': {'group_0': 0.6, 'group_1': 0.5}
            }
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'selection_rate']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in group_metrics:
                groups = list(group_metrics[metric].keys())
                values = list(group_metrics[metric].values())
                
                bars = axes[i].bar(groups, values, alpha=0.7, 
                                  color=plt.cm.Set2(np.arange(len(groups))))
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                
                axes[i].set_title(f'{metric.title()}', fontweight='bold')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1.1)
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'No data for {metric}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{metric.title()}', fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_bias_detection_summary(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "Bias Detection Summary"
    ) -> plt.Figure:
        """
        Create a comprehensive bias detection summary visualization.
        
        Args:
            results: Evaluation results from FairnessOptimizer
            save_path: Optional path to save the plot
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Disparity metrics heatmap
        if "fairness" in results and "disparities" in results["fairness"]:
            disparities = results["fairness"]["disparities"]
            
            metrics = list(disparities.keys())
            values = [disparities[m] for m in metrics]
            
            # Create color mapping for severity
            colors = []
            for val in values:
                if val < 0.05:
                    colors.append('green')
                elif val < 0.1:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            bars1 = ax1.barh(metrics, values, color=colors, alpha=0.7)
            ax1.set_xlabel('Disparity Value')
            ax1.set_title('Fairness Disparities')
            ax1.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
            ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='High Threshold')
            ax1.legend()
        
        # 2. Overall vs Group Performance
        if "overall" in results:
            overall_acc = results["overall"].get("accuracy", 0)
            
            if "fairness" in results and "by_group" in results["fairness"]:
                group_accs = results["fairness"]["by_group"].get("accuracy", {})
                groups = list(group_accs.keys())
                group_values = list(group_accs.values())
                
                x_pos = np.arange(len(groups) + 1)
                all_values = [overall_acc] + group_values
                all_labels = ['Overall'] + groups
                
                bars2 = ax2.bar(x_pos, all_values, alpha=0.7,
                              color=['blue'] + ['lightblue'] * len(groups))
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(all_labels, rotation=45)
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Overall vs Group Accuracy')
                ax2.grid(True, alpha=0.3)
        
        # 3. Statistical Tests (if available)
        if "statistical_tests" in results:
            stat_tests = results["statistical_tests"]
            if "selection_rate_test" in stat_tests:
                test_result = stat_tests["selection_rate_test"]
                p_val = test_result.get("p_value", 1.0)
                
                # Significance indicator
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                color = "red" if p_val < 0.05 else "green"
                
                ax3.bar(['P-value'], [p_val], color=color, alpha=0.7)
                ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
                ax3.set_ylabel('P-value')
                ax3.set_title(f'Statistical Significance\n({significance})')
                ax3.legend()
        
        # 4. Model Performance Summary
        if "overall" in results:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [results["overall"].get(m, 0) for m in metrics]
            
            bars4 = ax4.bar(metrics, values, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(metrics))))
            ax4.set_ylabel('Score')
            ax4.set_title('Model Performance Metrics')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars4, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_confidence_intervals(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "Model Performance with Confidence Intervals"
    ) -> plt.Figure:
        """
        Plot model performance metrics with confidence intervals.
        
        Args:
            results: Evaluation results with confidence intervals
            save_path: Optional path to save the plot
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if "confidence_intervals" in results:
            ci_data = results["confidence_intervals"]
            
            if "accuracy" in ci_data:
                acc_ci = ci_data["accuracy"]
                mean_acc = acc_ci.get("mean", 0)
                ci_lower = acc_ci.get("ci_lower", mean_acc)
                ci_upper = acc_ci.get("ci_upper", mean_acc)
                
                # Plot confidence interval
                ax.errorbar(['Accuracy'], [mean_acc], 
                           yerr=[[mean_acc - ci_lower], [ci_upper - mean_acc]],
                           fmt='o', capsize=10, capthick=2, markersize=10)
                
                ax.set_ylabel('Accuracy')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Add text with CI values
                ax.text(0, mean_acc + 0.1, 
                       f'Mean: {mean_acc:.3f}\nCI: [{ci_lower:.3f}, {ci_upper:.3f}]',
                       ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_fairness_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "Comprehensive Fairness Analysis Dashboard"
    ) -> plt.Figure:
        """
        Create a comprehensive fairness analysis dashboard.
        
        Args:
            results: Evaluation results from FairnessOptimizer
            save_path: Optional path to save the dashboard
            title: Dashboard title
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main performance metrics
        ax1 = fig.add_subplot(gs[0, 0])
        if "overall" in results:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [results["overall"].get(m, 0) for m in metrics]
            bars = ax1.bar(metrics, values, color=plt.cm.Set3(np.linspace(0, 1, len(metrics))))
            ax1.set_title('Overall Performance', fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
        
        # Group comparison
        ax2 = fig.add_subplot(gs[0, 1])
        if "fairness" in results and "by_group" in results["fairness"]:
            group_metrics = results["fairness"]["by_group"]
            if "accuracy" in group_metrics:
                groups = list(group_metrics["accuracy"].keys())
                values = list(group_metrics["accuracy"].values())
                ax2.bar(groups, values, alpha=0.7)
                ax2.set_title('Accuracy by Group', fontweight='bold')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
        
        # Disparity visualization
        ax3 = fig.add_subplot(gs[0, 2])
        if "fairness" in results and "disparities" in results["fairness"]:
            disparities = results["fairness"]["disparities"]
            metrics = list(disparities.keys())
            values = list(disparities.values())
            
            colors = ['red' if v > 0.1 else 'orange' if v > 0.05 else 'green' for v in values]
            ax3.barh(metrics, values, color=colors, alpha=0.7)
            ax3.set_title('Fairness Disparities', fontweight='bold')
            ax3.axvline(x=0.05, color='orange', linestyle='--', alpha=0.7)
            ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.7)
        
        # Statistical tests
        ax4 = fig.add_subplot(gs[1, :2])
        if "statistical_tests" in results and "selection_rate_test" in results["statistical_tests"]:
            test_data = results["statistical_tests"]["selection_rate_test"]
            if "group_rates" in test_data:
                groups = list(test_data["group_rates"].keys())
                rates = list(test_data["group_rates"].values())
                
                ax4.bar(groups, rates, alpha=0.7, color=['lightblue', 'lightcoral'])
                ax4.set_title('Selection Rates by Group', fontweight='bold')
                ax4.set_ylabel('Selection Rate')
                ax4.grid(True, alpha=0.3)
                
                # Add p-value annotation
                p_val = test_data.get("p_value", 1.0)
                significance = "Significant" if p_val < 0.05 else "Not Significant"
                ax4.text(0.5, max(rates) * 0.9, f'P-value: {p_val:.4f}\n({significance})',
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # Confidence intervals
        ax5 = fig.add_subplot(gs[1, 2])
        if "confidence_intervals" in results and "accuracy" in results["confidence_intervals"]:
            ci_data = results["confidence_intervals"]["accuracy"]
            mean_val = ci_data.get("mean", 0)
            ci_lower = ci_data.get("ci_lower", mean_val)
            ci_upper = ci_data.get("ci_upper", mean_val)
            
            ax5.errorbar(['Accuracy'], [mean_val], 
                        yerr=[[mean_val - ci_lower], [ci_upper - mean_val]],
                        fmt='o', capsize=10, capthick=2, markersize=10, color='blue')
            ax5.set_title('95% Confidence Interval', fontweight='bold')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
        
        # Summary text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create summary text
        summary_parts = []
        
        if "overall" in results:
            acc = results["overall"].get("accuracy", 0)
            summary_parts.append(f"Overall Accuracy: {acc:.3f}")
        
        if "fairness" in results and "disparities" in results["fairness"]:
            acc_disp = results["fairness"]["disparities"].get("accuracy_disparity", 0)
            summary_parts.append(f"Accuracy Disparity: {acc_disp:.3f}")
            
            if acc_disp < 0.05:
                fairness_level = "GOOD"
            elif acc_disp < 0.1:
                fairness_level = "MODERATE"
            else:
                fairness_level = "POOR"
            summary_parts.append(f"Fairness Level: {fairness_level}")
        
        if "statistical_tests" in results:
            if "selection_rate_test" in results["statistical_tests"]:
                p_val = results["statistical_tests"]["selection_rate_test"].get("p_value", 1.0)
                sig_status = "SIGNIFICANT BIAS DETECTED" if p_val < 0.05 else "NO SIGNIFICANT BIAS"
                summary_parts.append(f"Statistical Test: {sig_status}")
        
        model_name = results.get("model_name", "Unknown")
        strategy = results.get("strategy_name", "Unknown")
        summary_parts.extend([f"Model: {model_name}", f"Strategy: {strategy}"])
        
        summary_text = " | ".join(summary_parts)
        
        ax6.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                transform=ax6.transAxes, wrap=True)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig