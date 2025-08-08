"""
Training Metrics Visualization for J&J Contact Lens Defect Detection
Comprehensive visualization suite for training analysis and model comparison
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import argparse
import warnings
import yaml
import re

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class TrainingMetricsVisualizer:
    """Comprehensive training metrics visualization for J&J defect detection"""

    def __init__(self, logs_dir: Path = Path("logs"), output_dir: Path = Path("visualizations")):
        """
        Initialize the visualizer

        Args:
            logs_dir: Directory containing training logs and metrics
            output_dir: Directory to save visualization outputs
        """
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load all available metrics
        self.metrics_data = self._load_all_metrics()
        self.training_data = self._load_training_logs()
        self.eval_summaries, self.pass_fail_csvs, self.defect_instances_csvs = self._load_evaluation_artifacts()

        # J&J specific defect classes (update based on your actual classes)
        self.jj_defect_classes = [
            'background', 'edge_tear', 'scratch', 'surface_blob', 'bubble',
            'ring_defect', 'macro_defect', 'edge_not_closed', 'debris',
            'folded', 'inverted', 'missing', 'multiple'
        ]

        # Critical defects that always cause failure
        self.critical_defects = ['folded', 'inverted', 'missing', 'multiple', 'edge_not_closed']

        print(f"Loaded metrics from {len(self.metrics_data)} training sessions")
        print(f"Found {len(self.training_data)} training log files")
        print(f"Found {len(self.eval_summaries)} evaluation summaries")

    def _load_all_metrics(self) -> List[Dict]:
        """Load all metrics JSON files from logs directory"""
        metrics_files = list(self.logs_dir.glob("metrics_*.json"))
        all_metrics = []

        for file_path in metrics_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    data['file_path'] = str(file_path)
                    data['timestamp'] = file_path.stem.split('_', 1)[1] if '_' in file_path.stem else 'unknown'
                    all_metrics.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        return sorted(all_metrics, key=lambda x: x.get('session_start', ''))

    def _load_training_logs(self) -> List[Dict]:
        """Load training log files for detailed analysis"""
        log_files = list(self.logs_dir.glob("armor_*.log"))
        training_data = []

        for log_file in log_files:
            try:
                # Parse log file for training metrics
                log_data = self._parse_training_log(log_file)
                if log_data:
                    training_data.append(log_data)
            except Exception as e:
                print(f"Warning: Could not parse {log_file}: {e}")

        return training_data

    def _load_evaluation_artifacts(self) -> Tuple[List[Dict[str, Any]], List[Path], List[Path]]:
        """Load evaluator outputs: evaluation summaries and relevant CSVs from logs_dir."""
        eval_summaries = []
        for yml in sorted(self.logs_dir.glob("evaluation_summary_*.yaml")):
            try:
                with open(yml, 'r') as f:
                    data = yaml.safe_load(f) or {}
                data['file_path'] = str(yml)
                # Extract timestamp from filename
                m = re.search(r"evaluation_summary_(\d{8}_\d{6})", yml.stem)
                data['timestamp'] = m.group(1) if m else 'unknown'
                eval_summaries.append(data)
            except Exception as e:
                print(f"Warning: Could not load {yml}: {e}")

        pass_fail_csvs = sorted(self.logs_dir.glob("pass_fail_*.csv"))
        defect_instances_csvs = sorted(self.logs_dir.glob("defect_instances_*.csv"))
        return eval_summaries, pass_fail_csvs, defect_instances_csvs

    def _parse_training_log(self, log_file: Path) -> Optional[Dict]:
        """Parse training log file to extract loss curves and training progress.
        Recognizes lines like:
        - "Epoch X ... Train Loss: 0.1234 ... Val mAP: 0.8567"
        - "Batch i/N ... Loss: 0.2345"
        Returns dict with 'epochs' and optional 'batches'.
        """
        training_epochs: List[Dict[str, Any]] = []
        batch_losses: List[float] = []
        epoch_re = re.compile(r"Train Loss:\s*([0-9]*\.?[0-9]+).*Val mAP:\s*([0-9]*\.?[0-9]+)")
        batch_re = re.compile(r"Batch\s*\d+\s*/\s*\d+.*Loss:\s*([0-9]*\.?[0-9]+)")

        with open(log_file, 'r') as f:
            for line in f:
                m = epoch_re.search(line)
                if m:
                    try:
                        train_loss = float(m.group(1))
                        val_map = float(m.group(2))
                    except ValueError:
                        continue
                    training_epochs.append({
                        'train_loss': train_loss,
                        'val_map': val_map,
                        'epoch': len(training_epochs) + 1
                    })
                    continue
                b = batch_re.search(line)
                if b:
                    try:
                        batch_losses.append(float(b.group(1)))
                    except ValueError:
                        pass

        if training_epochs or batch_losses:
            return {
                'log_file': str(log_file),
                'timestamp': log_file.stem.split('_', 1)[1] if '_' in log_file.stem else 'unknown',
                'epochs': training_epochs,
                'batches': batch_losses
            }

        return None

    def create_model_comparison_dashboard(self) -> Path:
        """Create comprehensive model comparison dashboard"""
        if not self.metrics_data:
            print("No metrics data available for comparison")
            return None

        # Set up the dashboard layout
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Model Performance Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_performance_comparison(ax1)

        # 2. Defect Size Distribution (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_defect_size_distribution(ax2)

        # 3. Batch-level Trend Analysis (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_batch_trend_analysis(ax3)

        # 4. Per-Class Performance Heatmap (Second Row, Full Width)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_per_class_performance_heatmap(ax4)

        # 5. Training Loss Curves (Third Row Left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_training_loss_curves(ax5)

        # 6. Validation mAP Progression (Third Row Center)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_validation_map_progression(ax6)

        # 7. J&J Pass/Fail Analysis (Third Row Right)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_jj_pass_fail_analysis(ax7)

        # 8. Critical Defect Detection Performance (Bottom Row)
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_critical_defect_performance(ax8)

        # Add main title
        fig.suptitle(
            'J&J Contact Lens Defect Detection - Training Dashboard\n'
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=16, fontweight='bold'
        )

        # Save dashboard
        output_path = self.output_dir / f"training_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Training dashboard saved to: {output_path}")
        return output_path

    def _plot_model_performance_comparison(self, ax):
        """Plot performance comparison using latest evaluation summaries (mAP over sessions)."""
        if not self.eval_summaries:
            ax.text(0.5, 0.5, 'No evaluation summaries', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Evaluation Performance Comparison')
            return

        labels = [e.get('timestamp', str(i)) for i, e in enumerate(self.eval_summaries, 1)]
        maps = [float(e.get('mAP', 0.0)) for e in self.eval_summaries]

        bars = ax.bar(labels, maps, color=sns.color_palette("viridis", len(labels)))
        ax.set_ylabel('mAP')
        ax.set_title('Evaluation Performance Comparison (by session)')
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, maps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target mAP ≥ 0.90')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_inference_time_analysis(self, ax):
        """Plot inference time analysis"""
        inference_times = []
        models = []

        for session in self.metrics_data:
            for time_entry in session.get('metrics', {}).get('inference_times', []):
                inference_times.append(time_entry['time_ms'])
                models.append(time_entry['model'])

        if inference_times:
            # Create box plot of inference times by model
            df = pd.DataFrame({'Model': models, 'Inference_Time_ms': inference_times})

            if len(df['Model'].unique()) > 1:
                sns.boxplot(data=df, x='Model', y='Inference_Time_ms', ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.hist(inference_times, bins=20, alpha=0.7)
                ax.set_xlabel('Inference Time (ms)')
                ax.set_ylabel('Frequency')

            # Add 200ms target line for Edge Algorithm
            ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Edge Algorithm Limit (200ms)')
            ax.legend()
            ax.set_title('Inference Time Analysis')
        else:
            ax.text(0.5, 0.5, 'No inference time data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Inference Time Analysis')

    def _plot_memory_usage_analysis(self, ax):
        """Plot memory usage analysis"""
        memory_data = []
        models = []

        for session in self.metrics_data:
            for mem_entry in session.get('metrics', {}).get('memory_usage', []):
                memory_data.append(mem_entry['allocated_mb'])
                models.append(mem_entry['model'])

        if memory_data:
            df = pd.DataFrame({'Model': models, 'Memory_MB': memory_data})

            if len(df['Model'].unique()) > 1:
                sns.boxplot(data=df, x='Model', y='Memory_MB', ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.hist(memory_data, bins=20, alpha=0.7)
                ax.set_xlabel('Memory Usage (MB)')
                ax.set_ylabel('Frequency')

            ax.set_title('Memory Usage Analysis')
        else:
            ax.text(0.5, 0.5, 'No memory usage data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Memory Usage Analysis')

    def _plot_per_class_performance_heatmap(self, ax):
        """Plot per-class AP heatmap using evaluator's summary if available."""
        # Use the latest evaluation summary
        if not self.eval_summaries:
            ax.text(0.5, 0.5, 'No evaluation summaries found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Class AP Heatmap')
            return
        latest = self.eval_summaries[-1]
        ap_per_class = latest.get('AP_per_class', {}) or {}
        if not ap_per_class:
            ax.text(0.5, 0.5, 'No per-class AP data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Class AP Heatmap')
            return
        # Build a DataFrame with a single AP column
        classes = list(ap_per_class.keys())
        values = [float(ap_per_class[c]) for c in classes]
        df = pd.DataFrame({'AP': values}, index=classes)
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': 'AP'})
        ax.set_title('Per-Class AP Heatmap (latest evaluation)')
        ax.set_ylabel('Defect Classes')

    def _plot_training_loss_curves(self, ax):
        """Plot training loss curves"""
        if not self.training_data:
            ax.text(0.5, 0.5, 'No training loss data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Loss Curves')
            return

        for i, training_session in enumerate(self.training_data[:5]):  # Show up to 5 recent sessions
            epochs = training_session['epochs']
            if epochs:
                epoch_nums = [e['epoch'] for e in epochs]
                losses = [e['train_loss'] for e in epochs]

                ax.plot(epoch_nums, losses, label=f"Session {training_session['timestamp'][:8]}",
                        alpha=0.8, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_validation_map_progression(self, ax):
        """Plot validation mAP progression"""
        if not self.training_data:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation mAP Progression')
            return

        for i, training_session in enumerate(self.training_data[:5]):
            epochs = training_session['epochs']
            if epochs:
                epoch_nums = [e['epoch'] for e in epochs]
                val_maps = [e['val_map'] for e in epochs]

                ax.plot(epoch_nums, val_maps, label=f"Session {training_session['timestamp'][:8]}",
                        alpha=0.8, linewidth=2, marker='o', markersize=3)

        # Add target line
        ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target mAP ≥ 0.90')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation mAP')
        ax.set_title('Validation mAP Progression')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_defect_size_distribution(self, ax):
        """Plot distribution of defect sizes from exported evaluator CSV."""
        if not self.defect_instances_csvs:
            ax.text(0.5, 0.5, 'No defect instances CSV found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Defect Size Distribution')
            return
        latest_csv = self.defect_instances_csvs[-1]
        try:
            df = pd.read_csv(latest_csv)
        except Exception as e:
            ax.text(0.5, 0.5, f'Failed to read {latest_csv.name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Defect Size Distribution')
            return
        # Compute area if missing
        if 'area' not in df.columns and {'x1','y1','x2','y2'}.issubset(df.columns):
            df['width'] = (df['x2'] - df['x1']).clip(lower=0)
            df['height'] = (df['y2'] - df['y1']).clip(lower=0)
            df['area'] = df['width'] * df['height']
        if 'area' not in df.columns:
            ax.text(0.5, 0.5, 'Area not available in CSV', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Defect Size Distribution')
            return
        df = df.copy()
        df['log_area'] = np.log10(df['area'].clip(lower=1e-6))
        # Pick top classes by count to keep plot readable
        if 'class_name' in df.columns:
            top_classes = df['class_name'].value_counts().head(6).index
            sub = df[df['class_name'].isin(top_classes)]
            sns.boxplot(data=sub, x='class_name', y='log_area', ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_xlabel('Class (Top 6 by frequency)')
        else:
            ax.hist(df['log_area'], bins=30, alpha=0.7)
            ax.set_xlabel('log10(area)')
        ax.set_ylabel('log10(area)')
        ax.set_title('Defect Size Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_batch_trend_analysis(self, ax):
        """Plot batch-level loss trend if available in logs."""
        # Find latest session with batch data
        session = None
        for s in reversed(self.training_data):
            if s.get('batches'):
                session = s
                break
        if not session:
            ax.text(0.5, 0.5, 'No batch-level loss data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Batch-level Loss Trend')
            return
        losses = session['batches']
        x = np.arange(1, len(losses) + 1)
        ax.plot(x, losses, color='grey', alpha=0.3, label='Batch Loss')
        # Smooth
        n = len(losses)
        if n >= 5:
            w = max(5, min(101, (n // 20) * 2 + 1))  # odd window
            kernel = np.ones(w, dtype=float) / w
            smooth = np.convolve(np.array(losses, dtype=float), kernel, mode='same')
            ax.plot(x, smooth, color='blue', linewidth=2, label=f'Smoothed (w={w})')
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Loss')
        ax.set_title('Batch-level Training Loss Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_jj_pass_fail_analysis(self, ax):
        """Plot J&J pass/fail analysis using evaluator outputs."""
        if not self.eval_summaries:
            ax.text(0.5, 0.5, 'No evaluation summaries found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('J&J Quality Metrics')
            return
        latest = self.eval_summaries[-1]
        pass_rate = float(latest.get('pass_rate', np.nan))
        critical_recall = float(latest.get('jj_metrics', {}).get('critical_recall', np.nan))
        critical_miss_rate = 1.0 - critical_recall if not np.isnan(critical_recall) else np.nan

        categories = ['Overall Pass Rate', 'Critical Defect Miss Rate']
        values = [pass_rate, critical_miss_rate]
        colors = ['green', 'red']

        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Rate')
        ax.set_title('J&J Quality Metrics (latest evaluation)')
        ax.set_ylim(0, 1)

        for bar, value in zip(bars, values):
            height = bar.get_height() if not np.isnan(value) else 0
            label = f'{value:.2%}' if not np.isnan(value) else 'N/A'
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, label, ha='center', va='bottom')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_critical_defect_performance(self, ax):
        """Plot critical defect performance using per-class AP for critical classes."""
        if not self.eval_summaries:
            ax.text(0.5, 0.5, 'No evaluation summaries', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Critical Defect Performance')
            return
        latest = self.eval_summaries[-1]
        ap_per_class = latest.get('AP_per_class', {}) or {}
        # Filter to critical defects present in AP dict
        metrics = {c: ap_per_class[c] for c in self.critical_defects if c in ap_per_class}
        if not metrics:
            ax.text(0.5, 0.5, 'No critical class AP available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Critical Defect Performance')
            return
        classes = list(metrics.keys())
        aps = [float(metrics[c]) for c in classes]
        x = np.arange(len(classes))
        bars = ax.bar(x, aps, color=sns.color_palette("Set2", len(classes)))
        for bar, val in zip(bars, aps):
            ax.text(bar.get_x() + bar.get_width() / 2., val + 0.01, f"{val:.2f}", ha='center', va='bottom')
        ax.set_xlabel('Critical Defect Types')
        ax.set_ylabel('AP')
        ax.set_title('Critical Defect AP (latest evaluation)')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Minimum Acceptable (95%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def generate_training_summary_report(self) -> Path:
        """Generate comprehensive training summary report"""
        report_data = {
            'generation_time': datetime.now().isoformat(),
            'total_training_sessions': len(self.metrics_data),
            'models_evaluated': [],
            'best_performing_model': None,
            'training_efficiency': {},
            'recommendations': []
        }

        # Analyze model performance
        best_map = 0
        for session in self.metrics_data:
            for model_name, accuracy_data in session.get('metrics', {}).get('model_accuracy', {}).items():
                if model_name not in report_data['models_evaluated']:
                    report_data['models_evaluated'].append(model_name)

                if accuracy_data:
                    latest_metrics = accuracy_data[-1]['metrics']
                    if 'mAP' in latest_metrics and latest_metrics['mAP'] > best_map:
                        best_map = latest_metrics['mAP']
                        report_data['best_performing_model'] = {
                            'name': model_name,
                            'mAP': latest_metrics['mAP']
                        }

        # Add recommendations based on analysis
        if best_map < 0.90:
            report_data['recommendations'].append(
                f"Current best mAP ({best_map:.3f}) is below target (0.90). "
                "Consider: longer training, data augmentation, or ensemble methods."
            )

        # Save report
        output_path = self.output_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"Training summary report saved to: {output_path}")
        return output_path

    def create_model_specific_analysis(self, model_name: str) -> Optional[Path]:
        """Create detailed analysis for a specific model"""
        model_data = []

        for session in self.metrics_data:
            if model_name in session.get('metrics', {}).get('model_accuracy', {}):
                model_data.append(session)

        if not model_data:
            print(f"No data found for model: {model_name}")
            return None

        # Create detailed plots for this model
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detailed Analysis: {model_name}', fontsize=16, fontweight='bold')

        # Model-specific visualizations would go here
        # This is a framework - implement based on your specific needs

        output_path = self.output_dir / f"{model_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Visualize J&J Contact Lens Training Metrics")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"),
                        help="Directory containing training logs")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"),
                        help="Output directory for visualizations")
    parser.add_argument("--model", type=str, help="Generate detailed analysis for specific model")
    parser.add_argument("--dashboard", action="store_true",
                        help="Generate comprehensive training dashboard")
    parser.add_argument("--summary", action="store_true",
                        help="Generate training summary report")

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = TrainingMetricsVisualizer(args.logs_dir, args.output_dir)

    # Generate requested outputs
    if args.dashboard:
        dashboard_path = visualizer.create_model_comparison_dashboard()
        if dashboard_path:
            print(f"Dashboard created: {dashboard_path}")

    if args.summary:
        summary_path = visualizer.generate_training_summary_report()
        print(f"Summary report created: {summary_path}")

    if args.model:
        analysis_path = visualizer.create_model_specific_analysis(args.model)
        if analysis_path:
            print(f"Model analysis created: {analysis_path}")

    # If no specific options, create everything
    if not any([args.dashboard, args.summary, args.model]):
        print("Creating comprehensive visualization suite...")
        visualizer.create_model_comparison_dashboard()
        visualizer.generate_training_summary_report()
        print("All visualizations generated!")


if __name__ == "__main__":
    main()