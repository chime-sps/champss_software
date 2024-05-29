import importlib
import logging
import pickle
import sys
from typing import Dict, List, Type, Union

import numpy as np
import numpy.lib.recfunctions as rfn
from attr import Factory, attrib, attrs
from attr.validators import instance_of
from sps_common.interfaces import (
    CandidateClassification,
    CandidateClassificationLabel,
    MultiPointingCandidate,
)
from sps_multi_pointing import classifier

log = logging.getLogger(__name__)


class Classifier:
    """Base class for implementing classifiers."""

    def __init__(self, classifier_file):
        self.classifier = self.load_classifier(classifier_file)

    def load_classifier(self, classifier_file):
        with open(classifier_file, "rb") as infile:
            return pickle.loads(infile.read())

    def classify(self, candidate: MultiPointingCandidate) -> CandidateClassification:
        """Classifies a multi-pointing `candidate` as astro-physical or RFI."""
        raise NotImplementedError("Subclasses must implement this method")

    def create_candidate_array(self, candidate: MultiPointingCandidate) -> np.ndarray:
        cand_array = rfn.merge_arrays(
            [candidate.features, candidate.position_features], flatten=True
        )
        return rfn.structured_to_unstructured(cand_array)


class DummyClassifier(Classifier):
    """Placeholder classifier that just grades everything as an A-OK Astro candidate."""

    def __init__(self):
        pass

    def classify(self, candidate):
        return CandidateClassification(
            label=CandidateClassificationLabel.Astro, grade=1
        )


class SvmClassifier(Classifier):
    """
    Support Vector Machine classifier which is a purely binary classifier. It does not
    come with probability of a instance belonging to a particular class. Here we assume
    SVM only gives 0 or 1 (RFI or Astro) to a class instance.

    Parameters
    ----------
    classifier_file: str
        Path to the classifier file.
    """

    def __init__(self, classifier_file):
        super().__init__(classifier_file)

    def classify(self, candidate):
        """
        Classifies a multi-pointing `candidate` as astro-physical or RFI.

        Parameters
        ----------
        candidate: MultiPointingCandidate
            A MultiPointingCandidate for classification

        Returns
        -------
        candidate_classification: CandidateClassification
            A CandidateClassification class with label and grade for the classification
        """
        classification = self.classifier.predict(
            super().create_candidate_array(candidate)
        )[0]
        if classification == 1:
            return CandidateClassification(
                label=CandidateClassificationLabel(classification + 1).name,
                grade=classification,
            )
        else:
            return CandidateClassification(
                label=CandidateClassificationLabel(classification).name,
                grade=classification,
            )


class MlpClassifier(Classifier):
    """
    Multilayer Perceptron classifier that returns a probability for classification. By
    default each classification by a MLP comes with a probability that an instance
    belongs to a particular class. Here we expect the classifier to be a binary
    classifier. We can set the threshold for classifying as RFI, Ambiguous or Astro.

    Parameters
    ----------
    classifier_file: str
        Path to the classifier file.

    rfi_threshold: float
        The classification likelihood threshold to determine if a candidate is an RFI. Must be between 0 and 1. Default = 0.4

    ambiguous_threshold: float
        The classification likelihood threshold to determine if a candidate is an RFI. Must be between 0 and 1 and larger
        than rfi_threshold. Default = 0.6
    """

    def __init__(self, classifier_file, rfi_threshold=0.4, ambiguous_threshold=0.6):
        super().__init__(classifier_file)
        if rfi_threshold > 1.0:
            rfi_threshold = 1.0
        elif rfi_threshold < 0.0:
            rfi_threshold = 0.0
        if ambiguous_threshold < rfi_threshold:
            ambiguous_threshold = rfi_threshold
        elif ambiguous_threshold > 1.0:
            ambiguous_threshold = 1.0
        self.rfi = rfi_threshold
        self.ambiguous = ambiguous_threshold

    def classify(self, candidate):
        """
        Classifies a multi-pointing `candidate` as astro-physical, ambiguous or RFI.

        Parameters
        ----------
        candidate: MultiPointingCandidate
            A MultiPointingCandidate for classification

        Returns
        -------
        candidate_classification: CandidateClassification
            A CandidateClassification class with label and grade for the classification
        """
        classification = self.classifier.predict_proba(
            super().create_candidate_array(candidate)
        )[0, 1]
        if classification < self.rfi:
            return CandidateClassification(
                label=CandidateClassificationLabel.RFI, grade=1 - classification
            )
        elif classification < self.ambiguous:
            return CandidateClassification(
                label=CandidateClassificationLabel.Ambiguous, grade=classification
            )
        else:
            return CandidateClassification(
                label=CandidateClassificationLabel.Astro, grade=classification
            )


def load_active_classifiers(
    classifiers_config: List[Union[Classifier, str, Dict]]
) -> List[Classifier]:
    """
    Return instantiated classifiers given in the `classifiers_config`.

    The argument is a list of classifier instances or names (such as section
    in the config file). If the latter, the element is interpreted as follows:

    - class name: instantiated by calling the class constructor with no arguments
    - dictionary with a single key: the key is treated as the class name, and its
      value is another dictionary of argument names and values for the class constructor

    The class name can be unqualified (e.g., "SomeClassifier"), in which case it
    is looked up in this module (i.e., `sps_multi_pointing.classifier.classifier`) or
    fully-qualified (e.g., "another.module.SomeOtherClassifier"), in which case the
    module name ("another.module") is resolved using `importlib`, and then the class
    ("SomeOtherClassifier") looked up within it.

    If a classifier name cannot be resolved, it is ignored.

    Arguments
    ---------
    classifier: str
        name of the classifier's class, either fully qualified, or relative to this module.

    Returns
    -------
    Class instance or None if the `classifier` class was not found.

    Raises
    ------
    ValueError: if none of the given classifiers could be loaded
    """
    loaded_classifiers = []
    for c in classifiers_config:
        if isinstance(c, str):
            log.debug("str: %s", c)
            if cls := load_classifier_class(c):
                loaded_classifiers.append(cls())
            else:
                log.warning("No such classifier class: %s", c)
        elif isinstance(c, dict):
            log.debug("dict: %s", c)
            if len(c) == 1:
                for k, v in c.items():
                    log.debug("k: %s", k)
                    if cls := load_classifier_class(k):
                        loaded_classifiers.append(cls(**v))
                    else:
                        log.warning("No such classifier class: %s", c)
            else:
                log.error("Classifier config should be a single-key config: %s", c)
        elif isinstance(c, Classifier):
            log.debug("cls: %s", c)
            loaded_classifiers.append(c)
        else:
            log.warning("Don't know how to handle classifier request:", c.__class__, c)
    if not loaded_classifiers:
        raise ValueError("No valid classifiers given:", classifiers_config)

    log.info("Loaded classifiers: %s", loaded_classifiers)
    return loaded_classifiers


def load_classifier_class(classifier: str) -> Type:
    """
    Return class instance of type `classifier`, loading its module if necessary.

    Arguments
    ---------
    classifier: str
        name of the classifier's class, either fully qualified, or relative to this module.

    Returns
    -------
    Class instance or None if the `classifier` class was not found.
    """
    *module_name, class_name = classifier.split(".")
    if module_name:
        try:
            module = importlib.import_module(".".join(module_name))
        except ModuleNotFoundError:
            return None
    else:
        module = sys.modules[__name__]
    if hasattr(module, class_name):
        cls = getattr(module, class_name)
        log.debug("Classifier class %s:", cls)
        return cls


@attrs
class CandidateClassifier:
    """
    Processes `MultiPointingCandidate`s by applying a set of classification algorithms
    and combining their result into an RFI/Astrophysical label.

    Attributes:
    -----------

    active_classifiers (List[sps_multi_pointing.classifier.Classifier]):
        classification algorithms that are applied to the `candidate` and
        their result combined to label the candidate as RFI or Astrophysical.
    """

    active_classifiers: List[Classifier] = attrib(
        default=Factory(lambda: [DummyClassifier()]),
        converter=load_active_classifiers,
    )

    def classify(self, candidate: MultiPointingCandidate) -> MultiPointingCandidate:
        """Add the combined result of each active classifier on the multi-pointing
        `candidate` to the "classification" field of the candidate.
        """

        classifier_grades = [
            classifier.classify(candidate) for classifier in self.active_classifiers
        ]
        candidate.classification = self.combine_grades(classifier_grades)
        return candidate

    def combine_grades(
        self,
        classifier_grades: List[CandidateClassification],
    ) -> CandidateClassification:
        """
        Combines grades of all active classifiers into a single overall grade.

        Currently, we just choose the classification with the highest grade
        """
        return classifier_grades[np.argmax([g.grade for g in classifier_grades])]
