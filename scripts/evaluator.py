"""
evaluator.py
============
Model evaluation module for Credit Score Classification project.
Provides confusion matrix, per-class metrics, overall metrics,
ROC curves, classification reports, and benchmark comparison tools.
"""

import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Consistent style
sbn.set_theme(style="whitegrid")


class EvalClassifier:
    """Comprehensive model evaluation class."""

    def __init__(self, model):
        self.model = model

    # =========================================================================
    # 1. CONFUSION MATRIX
    # =========================================================================

    def get_confusion_matrix(self, y_true, y_pred, classes):
        """Build confusion matrix manually."""
        class_idx = {label: index for index, label in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in class_idx and pred_label in class_idx:
                cm[class_idx[true_label]][class_idx[pred_label]] += 1
        return cm

    def plot_confusion_matrix(self, cm, classes, title="Confusion Matrix", normalize=False):
        """Plot confusion matrix as a heatmap."""
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            annot = cm_norm
            fmt = '.2f'
            title = title + " (Normalized)"
        else:
            annot = cm
            fmt = 'd'

        plt.figure(figsize=(9, 7))
        sbn.heatmap(data=annot, annot=True, fmt=fmt, cmap="Blues",
                    cbar=True, linewidths=0.5, linecolor='white',
                    xticklabels=classes, yticklabels=classes)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.xlabel("Predicted Labels", fontweight='bold')
        plt.ylabel("True Labels", fontweight='bold')
        plt.title(title, fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 2. METRICS BY CLASS
    # =========================================================================

    def calculate_metrics_by_class(self, confusion_matrix, classes):
        """Calculate accuracy, precision, recall, specificity, F1 for each class."""
        metrics_table = []

        for i in range(len(classes)):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            tn = np.sum(confusion_matrix) - (tp + fp + fn)

            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_table.append([accuracy, precision, recall, specificity, f1_score])

        metrics_df = pd.DataFrame(
            metrics_table, index=classes,
            columns=["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
        )
        return metrics_df

    def plot_metrics_by_class(self, metrics, title="Classification Metrics by Class"):
        """Plot metrics by class as a heatmap."""
        plt.figure(figsize=(10, max(5, len(metrics) * 0.8)))
        sbn.heatmap(metrics, annot=True, fmt=".4f", cmap="YlGnBu",
                    cbar=True, linewidths=0.5, linecolor='white')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.xlabel("Metrics", fontweight='bold')
        plt.ylabel("Classes", fontweight='bold')
        plt.title(title, fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_metrics_barplot(self, metrics, title="Metrics Comparison by Class"):
        """Plot grouped bar chart of metrics by class."""
        metrics_melted = metrics.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        metrics_melted.rename(columns={'index': 'Class'}, inplace=True)

        plt.figure(figsize=(12, 6))
        sbn.barplot(data=metrics_melted, x='Class', y='Score', hue='Metric', palette='Set2')
        plt.title(title, fontweight='bold', fontsize=13)
        plt.xlabel("Class", fontweight='bold')
        plt.ylabel("Score", fontweight='bold')
        plt.ylim(0, 1.1)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 3. OVERALL METRICS (ALL CLASSES)
    # =========================================================================

    def calculate_metrics_xall(self, confusion_matrix, metrics):
        """Calculate mean metrics across all classes."""
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = metrics["Precision"].mean()
        recall = metrics["Recall"].mean()
        specificity = metrics["Specificity"].mean()
        f1_score = metrics["F1-Score"].mean()

        metrics_table = [[accuracy, precision, recall, specificity, f1_score]]
        metrics_df = pd.DataFrame(
            metrics_table, index=["Overall (Macro Avg)"],
            columns=["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
        )
        return metrics_df

    # =========================================================================
    # 4. SKLEARN CLASSIFICATION REPORT
    # =========================================================================

    def sklearn_classification_report(self, y_true, y_pred, classes, title="Classification Report"):
        """Generate and print sklearn's classification report."""
        report = classification_report(y_true, y_pred, target_names=classes)
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(report)
        return report

    # =========================================================================
    # 5. ROC CURVES (for multi-class)
    # =========================================================================

    def plot_roc_curves(self, y_true, y_prob, classes, title="ROC Curves (One-vs-Rest)"):
        """
        Plot ROC curves for multi-class classification using One-vs-Rest strategy.
        y_prob: array of shape (n_samples, n_classes) with predicted probabilities.
        """
        # Binarize true labels
        y_true_bin = label_binarize(y_true, classes=classes)
        n_classes = len(classes)

        plt.figure(figsize=(10, 8))
        colors = sbn.color_palette("Set1", n_colors=n_classes)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{classes[i]} (AUC = {roc_auc:.3f})')

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title(title, fontweight='bold', fontsize=13)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


class BenchmarkEvaluator:
    """Compare multiple models side by side."""

    def __init__(self):
        self.results = {}
        self.allowed_models = ['KNN', 'Random Forest', 'XGBoost', 'LightGBM']

    def add_model(self, model_name, confusion_matrix, metrics_df, overall_metrics, training_time=None):
        """Add evaluation results for a model."""
        self.results[model_name] = {
            'confusion_matrix': confusion_matrix,
            'metrics_by_class': metrics_df,
            'overall_metrics': overall_metrics,
            'training_time': training_time
        }

    def get_benchmark_table(self):
        """Generate a consolidated benchmark table."""
        rows = []
        for name, res in self.results.items():
            if name in self.allowed_models:
                overall = res['overall_metrics'].iloc[0]
                row = {
                    'Model': name,
                    'Accuracy': overall['Accuracy'],
                    'Precision': overall['Precision'],
                    'Recall': overall['Recall'],
                    'Specificity': overall['Specificity'],
                    'F1-Score': overall['F1-Score']
                }
                if res['training_time'] is not None:
                    row['Training Time (s)'] = res['training_time']
                rows.append(row)

        df = pd.DataFrame(rows)
        return df.set_index('Model')

    def plot_benchmark(self, title="Model Benchmark Comparison"):
        """Plot benchmark comparison as a grouped bar chart."""
        df = self.get_benchmark_table()
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']

        plt.figure(figsize=(14, 7))
        df[metric_cols].plot(kind='bar', ax=plt.gca(), colormap='Set2', edgecolor='white', width=0.8)
        plt.title(title, fontweight='bold', fontsize=14)
        plt.xlabel("Model", fontweight='bold')
        plt.ylabel("Score", fontweight='bold')
        plt.ylim(0, 1.15)
        plt.xticks(rotation=30, ha='right')
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)

        # Add value annotations on bars
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='%.3f', fontsize=7, padding=2)

        plt.tight_layout()
        plt.show()

    def plot_benchmark_heatmap(self, title="Model Benchmark Heatmap"):
        """Plot benchmark as heatmap for easy visual comparison."""
        df = self.get_benchmark_table()
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']

        plt.figure(figsize=(10, max(4, len(df) * 0.8)))
        sbn.heatmap(df[metric_cols], annot=True, fmt='.4f', cmap='YlGnBu',
                    linewidths=0.5, linecolor='white', cbar_kws={'label': 'Score'})
        plt.title(title, fontweight='bold', fontsize=13)
        plt.xlabel("Metric", fontweight='bold')
        plt.ylabel("Model", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices_grid(self, classes, title="Confusion Matrices Comparison"):
        """Plot all confusion matrices in a grid layout."""
        filtered_results = {name: res for name, res in self.results.items() if name in self.allowed_models}
        n_models = len(filtered_results)
        ncol = min(2, n_models)
        nrow = (n_models + ncol - 1) // ncol

        fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))
        if n_models == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        for i, (name, res) in enumerate(filtered_results.items()):
            cm = res['confusion_matrix']
            sbn.heatmap(data=cm, annot=True, fmt='d', cmap='Blues',
                        cbar=True, linewidths=0.5,
                        xticklabels=classes, yticklabels=classes,
                        ax=axs[i])
            axs[i].set_title(name, fontweight='bold')
            axs[i].set_xlabel("Predicted")
            axs[i].set_ylabel("True")
            axs[i].tick_params(axis='x', rotation=45)
            axs[i].tick_params(axis='y', rotation=0)

        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_training_time(self, title="Model Training Time Comparison"):
        """Plot bar chart of training times."""
        times = {}
        for name, res in self.results.items():
            if name in self.allowed_models and res['training_time'] is not None:
                times[name] = res['training_time']

        if not times:
            print("No training time data available.")
            return

        plt.figure(figsize=(10, 6))
        names = list(times.keys())
        values = list(times.values())
        colors = sbn.color_palette("Set2", n_colors=len(names))
        sbn.barplot(x=names, y=values, palette=colors)
        plt.title(title, fontweight='bold', fontsize=13)
        plt.xlabel("Model", fontweight='bold')
        plt.ylabel("Training Time (seconds)", fontweight='bold')
        plt.xticks(rotation=30, ha='right')

        for i, v in enumerate(values):
            plt.text(i, v + max(values) * 0.01, f"{v:.2f}s", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.show()
