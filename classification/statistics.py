import numpy as np

# reference: https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
class Confution_Matrix:

    def __init__(self):
        self.confusion_matrix = None
        # fraction of samples correctly classified in the dataset
        self.accuracy = None
        # fraction of samples correctly classified in the positive class among the ones classified in the positive class
        self.precision = None
        # fraction of samples correctly classified in the positive class among the ones belong to the positive class
        self.recall = None
        # harmonic mean of the precision and recall
        self.F1 = None

    def compute(self, y_pred, y_real):
        '''
        compute the confusion matrix
        '''
        # initialization
        n = y_pred.size
        C = int(y_real.max())
        self.confusion_matrix = np.zeros((C, C))

        # compute confusion matrix
        for i in range(n):
            self.confusion_matrix[
                y_pred[i].astype(int) - 1, y_real[i].astype(int) - 1
            ] += 1

        # accuracy
        self.accuracy = np.sum(np.diag(self.confusion_matrix)) / n
        # precision
        self.precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        # recall
        self.recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        # F1
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)


# TODO: silhouette index
