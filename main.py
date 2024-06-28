import numpy as np
from confusion_function import ConfusionFunction

# Example data
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.85, 0.3, 0.7])
thresholds = np.linspace(0, 1, 11)

# Create the class instance and calculate metrics
metrics = ConfusionFunction(y_true, y_pred, thresholds)

# Print confusion matrix and metrics
metrics.print_confusion_matrix()

# Generate plots
metrics.plot_error_curve()
metrics.plot_roc_curve()
metrics.plot_auc_curve()
