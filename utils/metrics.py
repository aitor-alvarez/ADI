from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def multi_auc_score(y, y_predicted, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y)
    y_true = lb.transform(y)
    y_pred = lb.transform(y_predicted)

    return roc_auc_score(y_true, y_pred)