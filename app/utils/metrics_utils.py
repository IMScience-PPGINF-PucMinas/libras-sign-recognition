from sklearn.metrics import classification_report


def compute_metrics(y_true, y_pred):
    """
    Compute the accuracy of the model
    :param y_true: true labels
    :param y_pred: predicted labels
    """

    report = classification_report(y_true, y_pred, zero_division=0, labels=list(set(y_true)))

    print(report)
