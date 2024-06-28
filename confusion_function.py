import numpy as np
import matplotlib.pyplot as plt
from calculate_metrics import CalculateMetrics

class ConfusionFunction:
    """
    Class for calculating confusion matrix metrics and generating error, ROC, and AUC curves.
    """

    def __init__(self, true_labels, predicted_scores, thresholds=None):
        """
        Initialize the class with true labels, predicted scores, and thresholds.

        Parameters:
        true_labels (list or np.array): True labels.
        predicted_scores (list or np.array): Predicted scores.
        thresholds (list): Thresholds for calculating metrics. If None, use [0.5].
        """
        self.true_labels = np.array(true_labels)
        self.predicted_scores = np.array(predicted_scores)
        self.thresholds = thresholds if thresholds is not None and len(thresholds) > 0 else [0.5]
        self.metrics = self._calculate_metrics()

    def _calculate_confusion_matrix(self, threshold):
        """
        Calculate the confusion matrix for a given threshold.

        Parameters:
        threshold (float): Threshold for classification.

        Returns:
        tuple: Values of TP, FP, TN, FN.
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        for true, pred in zip(self.true_labels, self.predicted_scores):
            if pred >= threshold:
                if true == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if true == 0:
                    tn += 1
                else:
                    fn += 1
        return tp, fp, tn, fn

    def _calculate_metrics(self):
        """
        Calculate the metrics for all thresholds.

        Returns:
        list: List of dictionaries with metrics for each threshold.
        """
        metrics = []
        for threshold in self.thresholds:
            tp, fp, tn, fn = self._calculate_confusion_matrix(threshold)
            accuracy = CalculateMetrics._calculate_accuracy(tp, fp, tn, fn)
            precision = CalculateMetrics._calculate_precision(tp, fp)
            recall = CalculateMetrics._calculate_recall(tp, fn)
            f1_score = CalculateMetrics._calculate_f1_score(precision, recall)
            specificity = CalculateMetrics._calculate_specificity(tn, fp)
            npv = CalculateMetrics._calculate_npv(tn, fn)
            fpr = CalculateMetrics._calculate_fpr(fp, tn)
            fdr = CalculateMetrics._calculate_fdr(fp, tp)
            fnr = CalculateMetrics._calculate_fnr(fn, tp)
            cohens_kappa = CalculateMetrics._calculate_cohens_kappa(tp, fp, tn, fn)
            jaccard_index = CalculateMetrics._calculate_jaccard_index(tp, fp, fn)
            metrics.append({
                'threshold': threshold,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'specificity': specificity,
                'npv': npv,
                'fpr': fpr,
                'fdr': fdr,
                'fnr': fnr,
                'cohens_kappa': cohens_kappa,
                'jaccard_index': jaccard_index
            })
        return metrics

    def plot_error_curve(self, title='Error Curve', xlabel='Threshold', ylabel='Error Rate'):
        """
        Plot the error curve as a function of the threshold.

        Parameters:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        errors = [(1 - m['accuracy']) for m in self.metrics]
        thresholds = [m['threshold'] for m in self.metrics]
        plt.plot(thresholds, errors, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_roc_curve(self, title='ROC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate'):
        """
        Plot the ROC curve.

        Parameters:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        tprs = [m['recall'] for m in self.metrics]
        fprs = [m['fpr'] for m in self.metrics]
        plt.plot(fprs, tprs, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_auc_curve(self, title='AUC Curve', xlabel='False Positive Rate', ylabel='True Positive Rate'):
        """
        Plot the AUC curve.

        Parameters:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        tprs = [m['recall'] for m in self.metrics]
        fprs = [m['fpr'] for m in self.metrics]
        auc = np.trapz(tprs, fprs)
        plt.plot(fprs, tprs, marker='o')
        plt.fill_between(fprs, tprs, alpha=0.3)
        plt.title(f'{title} (AUC = {auc:.2f})')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def print_confusion_matrix(self):
        """
        Print the confusion matrix and metrics for each threshold.
        """
        for metric in self.metrics:
            print(f"Threshold: {metric['threshold']}")
            print(f"TP: {metric['tp']}, FP: {metric['fp']}, TN: {metric['tn']}, FN: {metric['fn']}")
            print(f"Accuracy: {metric['accuracy']:.2f}, Precision: {metric['precision']:.2f}, Recall: {metric['recall']:.2f}, F1 Score: {metric['f1_score']:.2f}")
            print(f"Specificity: {metric['specificity']:.2f}, NPV: {metric['npv']:.2f}, FPR: {metric['fpr']:.2f}, FDR: {metric['fdr']:.2f}, FNR: {metric['fnr']:.2f}")
            print(f"Cohen's Kappa: {metric['cohens_kappa']:.2f}, Jaccard Index: {metric['jaccard_index']:.2f}")
            print()
