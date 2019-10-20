from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.val = None
        self.avg = 0.
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0.

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1. - self.momentum)
        self.val = val


def plot_aucs(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(12, 6), dpi=100)

    ax = axs[0]
    ax.plot([0, 1], [0, 1], linestyle='--', rasterized=True)
    # plot the roc curve for the model
    ax.plot(fpr, tpr, marker='.')
    ax.set_title('ROC curve')

    ax = axs[1]
    ax.plot([0, 1], [0.5, 0.5], linestyle='--')
    ax.plot(recall, precision, marker='.', rasterized=True)
    ax.set_title('Precision-Recall Curve')

    fig.suptitle('Test set metrics')
    return fig
