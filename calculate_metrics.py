class CalculateMetrics:

    """
    Calcula métricas de avaliação de classificadores.
    
    """

    def __init__(self) -> None:
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None

    def _calculate_accuracy(tp, fp, tn, fn):
        """
        Calculate accuracy.

        Parameters:
        tp (int): True Positives.
        fp (int): False Positives.
        tn (int): True Negatives.
        fn (int): False Negatives.

        Returns:
        float: Accuracy.
        """
        return (tp + tn) / (tp + fp + tn + fn)

    def _calculate_precision(tp, fp):
        """
        Calculate precision.

        Parameters:
        tp (int): True Positives.
        fp (int): False Positives.

        Returns:
        float: Precision.
        """
        return tp / (tp + fp) if tp + fp > 0 else 0

    def _calculate_recall(tp, fn):
        """
        Calculate recall.

        Parameters:
        tp (int): True Positives.
        fn (int): False Negatives.

        Returns:
        float: Recall.
        """
        return tp / (tp + fn) if tp + fn > 0 else 0

    def _calculate_f1_score(precision, recall):
        """
        Calculate F1 Score.

        Parameters:
        precision (float): Precision.
        recall (float): Recall.

        Returns:
        float: F1 Score.
        """
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    def _calculate_specificity(tn, fp):
        """
        Calculate specificity (True Negative Rate).

        Parameters:
        tn (int): True Negatives.
        fp (int): False Positives.

        Returns:
        float: Specificity.
        """
        return tn / (tn + fp) if tn + fp > 0 else 0

    def _calculate_npv(tn, fn):
        """
        Calculate Negative Predictive Value.

        Parameters:
        tn (int): True Negatives.
        fn (int): False Negatives.

        Returns:
        float: Negative Predictive Value.
        """
        return tn / (tn + fn) if tn + fn > 0 else 0

    def _calculate_fpr(fp, tn):
        """
        Calculate False Positive Rate.

        Parameters:
        fp (int): False Positives.
        tn (int): True Negatives.

        Returns:
        float: False Positive Rate.
        """
        return fp / (fp + tn) if fp + tn > 0 else 0

    def _calculate_fdr(fp, tp):
        """
        Calculate False Discovery Rate.

        Parameters:
        fp (int): False Positives.
        tp (int): True Positives.

        Returns:
        float: False Discovery Rate.
        """
        return fp / (fp + tp) if fp + tp > 0 else 0

    def _calculate_fnr(fn, tp):
        """
        Calculate False Negative Rate.

        Parameters:
        fn (int): False Negatives.
        tp (int): True Positives.

        Returns:
        float: False Negative Rate.
        """
        return fn / (fn + tp) if fn + tp > 0 else 0

    def _calculate_cohens_kappa(tp, fp, tn, fn):
        """
        Calculate Cohen's Kappa.

        Parameters:
        tp (int): True Positives.
        fp (int): False Positives.
        tn (int): True Negatives.
        fn (int): False Negatives.

        Returns:
        float: Cohen's Kappa.
        """
        total = tp + fp + tn + fn
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
        return (po - pe) / (1 - pe) if 1 - pe > 0 else 0

    def _calculate_jaccard_index(tp, fp, fn):
        """
        Calculate Jaccard Index.

        Parameters:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.

        Returns:
        float: Jaccard Index.
        """
        return tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

