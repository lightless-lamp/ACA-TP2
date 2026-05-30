"""
Classification Metrics for Baseline CNN Evaluation
=====================================================
Implements evaluation metrics for the butterfly image classifier:
  - Accuracy (overall + per-class)
  - Precision, Recall, F1-Score (per-class, macro, weighted)
  - Confusion Matrix
  - Learning curves (loss / accuracy over epochs)

Usage
-----
    from classification_metrics import ClassificationEvaluator

    evaluator = ClassificationEvaluator(class_names=class_names)
    evaluator.update(y_true_batch, y_pred_batch)   # call after each batch/epoch
    report = evaluator.compute()
    evaluator.print_report(report)
    evaluator.plot_confusion_matrix(report)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict


# ---------------------------------------------------------------------------
# Core metric functions (no sklearn required)
# ---------------------------------------------------------------------------

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall accuracy = correct predictions / total predictions.

    Parameters
    ----------
    y_true : array-like, shape (N,)   — ground-truth class indices
    y_pred : array-like, shape (N,)   — predicted class indices

    Returns
    -------
    accuracy : float  in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same length"
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true,
    y_pred,
    n_classes = None,
):
    """
    Compute the confusion matrix C where C[i, j] is the number of samples
    with true label i predicted as label j.

    Parameters
    ----------
    y_true    : array-like, shape (N,)
    y_pred    : array-like, shape (N,)
    n_classes : int or None — inferred from data if not provided

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1_per_class(
    cm,
    eps = 1e-9,
):
    """
    Per-class Precision, Recall, and F1-Score derived from a confusion matrix.

    Precision_i = TP_i / (TP_i + FP_i)
    Recall_i    = TP_i / (TP_i + FN_i)
    F1_i        = 2 * P_i * R_i / (P_i + R_i)

    Parameters
    ----------
    cm  : np.ndarray, shape (C, C) — confusion matrix (rows=true, cols=pred)
    eps : float — numerical stability constant

    Returns
    -------
    precision : np.ndarray, shape (C,)
    recall    : np.ndarray, shape (C,)
    f1        : np.ndarray, shape (C,)
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp          # column sum − TP
    fn = cm.sum(axis=1) - tp          # row sum    − TP

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2.0 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def macro_average(
    precision,
    recall,
    f1
):
    """Unweighted mean across all classes."""
    return {
        "precision_macro": float(precision.mean()),
        "recall_macro":    float(recall.mean()),
        "f1_macro":        float(f1.mean()),
    }


def weighted_average(
    precision,
    recall,
    f1,
    support
):
    """Mean weighted by the number of true samples per class (support)."""
    total = support.sum()
    w = support / (total + 1e-9)
    return {
        "precision_weighted": float((precision * w).sum()),
        "recall_weighted":    float((recall    * w).sum()),
        "f1_weighted":        float((f1        * w).sum()),
    }


# ---------------------------------------------------------------------------
# High-level evaluator class
# ---------------------------------------------------------------------------

class ClassificationEvaluator:
    """
    Stateful evaluator: accumulates predictions across batches, then computes
    all classification metrics in one call.

    Example
    -------
    >>> evaluator = ClassificationEvaluator(class_names=class_names)
    >>> for images, labels in val_loader:
    ...     outputs = model(images)
    ...     preds   = outputs.argmax(dim=1)
    ...     evaluator.update(labels.numpy(), preds.numpy())
    >>> report = evaluator.compute()
    >>> evaluator.print_report(report)
    """

    def __init__(self, class_names = None, n_classes= None):
        """
        Parameters
        ----------
        class_names : list of str, optional
            Human-readable class labels (length must equal n_classes).
        n_classes   : int, optional
            Number of classes. Inferred from data if not provided.
        """
        self.class_names = class_names
        self.n_classes   = n_classes if n_classes is not None else (
            len(class_names) if class_names is not None else None
        )
        self._y_true: list[int] = []
        self._y_pred: list[int] = []

    # ------------------------------------------------------------------
    def update(self, y_true, y_pred):
        """Accumulate a batch of ground-truth and predicted labels."""
        self._y_true.extend(np.asarray(y_true, dtype=int).tolist())
        self._y_pred.extend(np.asarray(y_pred, dtype=int).tolist())

    def reset(self):
        """Clear accumulated predictions (call between epochs/models)."""
        self._y_true.clear()
        self._y_pred.clear()

    # ------------------------------------------------------------------
    def compute(self):
        """
        Compute all metrics from accumulated predictions.

        Returns
        -------
        report : dict with keys:
            accuracy        — float
            cm              — np.ndarray (C, C)
            precision       — np.ndarray (C,)  per-class
            recall          — np.ndarray (C,)  per-class
            f1              — np.ndarray (C,)  per-class
            support         — np.ndarray (C,)  samples per class
            macro           — dict  (precision/recall/f1 macro)
            weighted        — dict  (precision/recall/f1 weighted)
            class_names     — list[str] or None
        """
        y_true = np.array(self._y_true, dtype=int)
        y_pred = np.array(self._y_pred, dtype=int)

        n_classes = self.n_classes or int(max(y_true.max(), y_pred.max())) + 1

        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred, n_classes=n_classes)
        precision, recall, f1 = precision_recall_f1_per_class(cm)
        support = cm.sum(axis=1).astype(float)   # true samples per class

        return {
            "accuracy":    acc,
            "cm":          cm,
            "precision":   precision,
            "recall":      recall,
            "f1":          f1,
            "support":     support,
            "macro":       macro_average(precision, recall, f1),
            "weighted":    weighted_average(precision, recall, f1, support),
            "class_names": self.class_names,
        }

    # ------------------------------------------------------------------
    def print_report(self, report = None):
        """
        Print a classification report similar to sklearn's, but self-contained.
        """
        if report is None:
            report = self.compute()

        names   = report["class_names"] or [str(i) for i in range(len(report["precision"]))]
        col_w   = max(len(n) for n in names) + 2

        header = f"{'Class':<{col_w}}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>10}"
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))

        for i, name in enumerate(names):
            print(
                f"{name:<{col_w}}  "
                f"{report['precision'][i]:>10.4f}  "
                f"{report['recall'][i]:>10.4f}  "
                f"{report['f1'][i]:>10.4f}  "
                f"{int(report['support'][i]):>10d}"
            )

        print("-" * len(header))
        m = report["macro"]
        w = report["weighted"]
        total = int(report["support"].sum())
        print(
            f"{'Macro avg':<{col_w}}  "
            f"{m['precision_macro']:>10.4f}  "
            f"{m['recall_macro']:>10.4f}  "
            f"{m['f1_macro']:>10.4f}  "
            f"{total:>10d}"
        )
        print(
            f"{'Weighted avg':<{col_w}}  "
            f"{w['precision_weighted']:>10.4f}  "
            f"{w['recall_weighted']:>10.4f}  "
            f"{w['f1_weighted']:>10.4f}  "
            f"{total:>10d}"
        )
        print("=" * len(header))
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}  ({report['accuracy']*100:.2f}%)\n")

    # ------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        report= None,
        normalise= True,
        figsize = (14, 12),
        cmap = "Blues",
        save_path = None,
    ):
        """
        Plot the confusion matrix as a heatmap.

        Parameters
        ----------
        report    : pre-computed report dict (calls compute() if None)
        normalise : if True, show row-normalised values (recall per class)
        figsize   : matplotlib figure size
        cmap      : colormap name
        save_path : file path to save the figure (e.g. "cm.png")
        """
        if report is None:
            report = self.compute()

        cm     = report["cm"].astype(float)
        names  = report["class_names"] or [str(i) for i in range(cm.shape[0])]
        n      = cm.shape[0]

        if normalise:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot  = cm / (row_sums + 1e-9)
            title    = "Confusion Matrix (row-normalised)"
            fmt      = ".2f"
            vmax     = 1.0
        else:
            cm_plot  = cm
            title    = "Confusion Matrix (counts)"
            fmt      = "d"
            vmax     = None

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.03)

        ax.set(
            xticks=np.arange(n),
            yticks=np.arange(n),
            xticklabels=names,
            yticklabels=names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)

        # Annotate cells only when n is manageable
        if n <= 20:
            thresh = cm_plot.max() / 2.0
            for i in range(n):
                for j in range(n):
                    val = f"{cm_plot[i, j]:{fmt}}"
                    ax.text(
                        j, i, val,
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_plot[i, j] > thresh else "black",
                    )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Learning curve tracker
# ---------------------------------------------------------------------------

class LearningCurveTracker:
    """
    Records training and validation loss/accuracy per epoch and plots them.

    Example
    -------
    >>> tracker = LearningCurveTracker()
    >>> for epoch in range(n_epochs):
    ...     train_loss, train_acc = run_epoch(train_loader)
    ...     val_loss,   val_acc   = run_epoch(val_loader, eval=True)
    ...     tracker.update(train_loss, train_acc, val_loss, val_acc)
    >>> tracker.plot()
    """

    def __init__(self):
        self.history: dict[str, list[float]] = defaultdict(list)

    def update(
        self,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    ):
        self.history["train_loss"].append(float(train_loss))
        self.history["train_acc"].append(float(train_acc))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_acc"].append(float(val_acc))

    def plot(
        self,
        title= "Learning Curves",
        save_path = None,
    ):
        """Plot loss and accuracy curves side by side."""
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # --- Loss ---
        ax1.plot(epochs, self.history["train_loss"], label="Train loss",      marker="o", ms=4)
        ax1.plot(epochs, self.history["val_loss"],   label="Validation loss", marker="s", ms=4)
        ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        ax1.legend()
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax1.grid(alpha=0.3)

        # --- Accuracy ---
        ax2.plot(epochs, self.history["train_acc"], label="Train acc",      marker="o", ms=4)
        ax2.plot(epochs, self.history["val_acc"],   label="Validation acc", marker="s", ms=4)
        ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax2.legend()
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.grid(alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def best_epoch(self, monitor = "val_acc"):
        """Return epoch index and value of the best monitored metric."""
        values = self.history[monitor]
        if not values:
            return {}
        best_fn = np.argmax if "acc" in monitor else np.argmin
        idx = int(best_fn(values))
        return {
            "epoch":      idx + 1,
            monitor:      values[idx],
            "train_loss": self.history["train_loss"][idx],
            "val_loss":   self.history["val_loss"][idx],
            "train_acc":  self.history["train_acc"][idx],
            "val_acc":    self.history["val_acc"][idx],
        }


# ---------------------------------------------------------------------------
# Convenience: compare multiple models side by side
# ---------------------------------------------------------------------------

def compare_models(
    reports,
    metric="f1_macro",
    save_path= None,
):
    """
    Bar chart comparing a chosen metric across multiple models/experiments.

    Parameters
    ----------
    reports  : { model_name: report_dict }  — one report per model
    metric   : key inside report['macro'] or 'accuracy'
    save_path: optional file path to save figure
    """
    names  = list(reports.keys())
    values = []
    for name, rep in reports.items():
        if metric == "accuracy":
            values.append(rep["accuracy"])
        elif metric in rep.get("macro", {}):
            values.append(rep["macro"][metric])
        elif metric in rep.get("weighted", {}):
            values.append(rep["weighted"][metric])
        else:
            raise KeyError(f"Metric '{metric}' not found in report.")

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    bars = ax.bar(names, values, color="steelblue", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set(
        title=f"Model Comparison — {metric}",
        ylabel=metric,
        ylim=(max(0, min(values) - 0.05), min(1.0, max(values) + 0.1)),
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    C   = 75                              # 75 butterfly classes
    N   = 1000
    class_names = [f"class_{i:02d}" for i in range(C)]

    y_true = rng.integers(0, C, size=N)
    # Simulate a decent classifier: 70% correct
    y_pred = np.where(rng.random(N) < 0.70, y_true, rng.integers(0, C, N))

    # --- Evaluator ---
    ev = ClassificationEvaluator(class_names=class_names, n_classes=C)
    ev.update(y_true, y_pred)
    report = ev.compute()
    ev.print_report(report)
    # ev.plot_confusion_matrix(report)   # uncomment to show plot

    # --- Learning curves ---
    tracker = LearningCurveTracker()
    for epoch in range(20):
        tracker.update(
            train_loss=2.0 * np.exp(-epoch * 0.15) + rng.normal(0, 0.02),
            train_acc =0.3 + 0.6 * (1 - np.exp(-epoch * 0.2)) + rng.normal(0, 0.01),
            val_loss  =2.2 * np.exp(-epoch * 0.13) + rng.normal(0, 0.03),
            val_acc   =0.25 + 0.55 * (1 - np.exp(-epoch * 0.18)) + rng.normal(0, 0.015),
        )
    print("Best epoch:", tracker.best_epoch("val_acc"))
    # tracker.plot()   # uncomment to show plot

    # --- Model comparison ---
    ev2 = ClassificationEvaluator(class_names=class_names, n_classes=C)
    y_pred2 = np.where(rng.random(N) < 0.80, y_true, rng.integers(0, C, N))
    ev2.update(y_true, y_pred2)
    report2 = ev2.compute()

    compare_models(
        {"Baseline": report, "Baseline + GAN aug": report2},
        metric="f1_macro",
    )