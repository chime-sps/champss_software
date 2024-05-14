import pickle
import numpy.lib.recfunctions as rfn
import numpy as np
import scipy
import sklearn
import sys
from sklearn import neural_network, svm
from sklearn.model_selection import cross_val_score


class Trainer:
    """
    Base class for classifier training

    Parameters
    ----------
    kwargs
        Arguments that are possible for the classification algorithm to use.
    """

    def __init__(self, **kwargs):
        self.trainer = self.initialise_trainer(**kwargs)

    def initialise_trainer(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self, candidate_list, compute_metrics=True, save_model=True, filename=""):
        """
        Train the classifier using a list of MultiPointingCandidates along with their labels.

        Parameters
        ----------
        candidate_list: list(MultiPointingCandidate)
            A list of MultiPointingCandidates that forms the training dataset.

        labels: np.ndarray
            A 1-D numpy array with the labels of each MultiPointingCandidate.

        save_model: bool
            Whether to save the trained classifier model.

        filename: str
            Path to where to save the trained classifier model and a features.txt file showing the features used.
        """
        data, labels, features = self.get_features(candidate_list)
        self.trainer.fit(data, labels)
        if compute_metrics:
            metrics = self.compute_performance_metrics(data, labels)
        if save_model:
            if not filename.endswith(".pickle"):
                filename = filename + ".pickle"
            with open(filename, "wb") as outfile:
                outfile.write(pickle.dumps(self.trainer))
            with open(filename.rstrip(".pickle") + "_metadata.txt", "w") as outfile:
                outfile.write("features : ")
                for f in features:
                    outfile.write(f + ", ")
                outfile.write("\n")
                if compute_metrics:
                    for m in metrics:
                        outfile.write("{} : {}\n".format(m, metrics[m]))
                outfile.write(
                    "python version : {}.{}.{}\n".format(
                        sys.version_info[0], sys.version_info[1], sys.version_info[2]
                    )
                )
                outfile.write("numpy version : {}\n".format(np.__version__))
                outfile.write("scipy version : {}\n".format(scipy.__version__))
                outfile.write("scikit-learn version : {}\n".format(sklearn.__version__))

    def compute_performance_metrics(self, data, labels):
        """
        Compute metrics on the viability of the classifier. Currently using several cross validation scores
        """
        scores = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metrics = {}
        for s in scores:
            metrics[s] = cross_val_score(self.trainer, data, labels, cv=5, scoring=s)
        return metrics

    def get_features(self, candidate_list):
        """
        Obtain the features and labels of the list of MultiPointingCandidate

        Parameters
        ----------
        candidate_list: list(MultiPointingCandidate)
            A list of MultiPointingCandidates that forms the training dataset.

        Returns
        -------
        data: np.ndarray
            2-D numpy array of the features from all candidates with shape (n_candidates, n_features)

        labels: np.ndarray
            1-D numpy array of the labels from all candidates with size n_candidates

        features: list
            A list of features used by the candidates with size n_features
        """
        data = []
        labels = []
        features = []
        for i, cand in enumerate(candidate_list):
            cand_array = rfn.merge_arrays(
                [cand.features, cand.position_features], flatten=True
            )
            labels.append(np.min([cand.classification.label.value, 1]))
            data.append(rfn.structured_to_unstructured(cand_array)[0])
            if i == 0:
                features = cand_array.dtype.names
            else:
                if cand_array.dtype.names != features:
                    raise ValueError("The features of the candidates are not the same")
        return np.asarray(data), np.asarray(labels), list(features)


class SvmTrainer(Trainer):
    """
    Class to implement support vector machine classifier.

    See https://scikit-learn.org/stable/modules/svm.html for a list of properties that can be called.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialise_trainer(self, **kwargs):
        return svm.SVC(**kwargs)


class MlpTrainer(Trainer):
    """
    Class to implement multilayer perceptron classifier

    See https://scikit-learn.org/stable/modules/neural_networks_supervised.html for a list of properties that can be
    called.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialise_trainer(self, **kwargs):
        return neural_network.MLPClassifier(**kwargs)
