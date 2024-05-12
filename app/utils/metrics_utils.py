from sklearn.metrics import classification_report


def compute_metrics(y_true, y_pred, target_names):
    """
    Compute the accuracy of the model
    :param y_true: true labels
    :param y_pred: predicted labels
    """
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    print(report)
