# -*- coding: utf-8 -*-

from sklearn import metrics

def metrics_assign_total(x, y, y_pred):
    m_su = metrics_supervised(y, y_pred)
    m_unsu = metrics_unsupervised(x, y_pred)
    return dict(m_su, **m_unsu)

def metrics_supervised(y, y_pred):
    return {
        'Homo': metrics.homogeneity_score(y, y_pred),
        'Compl': metrics.completeness_score(y, y_pred),
        'V-meas': metrics.v_measure_score(y, y_pred),
        'ARI': metrics.adjusted_rand_score(y, y_pred),
        'AMI': metrics.adjusted_mutual_info_score(y, y_pred),
    }

def metrics_unsupervised(x, y):
    return {'Sil': metrics.silhouette_score(x, y, metric='euclidean', sample_size=4000)}


def predict_accuracy(y, y_pred, need_acc=False):
    if need_acc:
        return {'Accuracy': metrics.accuracy_score(y, y_pred)}
    else:
        return {'Precision': metrics.precision_score(y, y_pred),
                'Recall': metrics.recall_score(y, y_pred),
                'F1': metrics.f1_score(y, y_pred)
                }


def predict_report(y, y_pred):
    return {'Matrix': metrics.confusion_matrix(y, y_pred), 'Report': metrics.classification_report(y, y_pred)}