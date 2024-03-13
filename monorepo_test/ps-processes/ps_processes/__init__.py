class FailedChi2TestError(ValueError):
    """
    Exception raised when the power spectra stack failed the chi2 test
    """

    pass


class NoPsStackDbError(ValueError):
    """
    Exception raised when the power spectra stack database does not exist
    """

    pass


class IncompleteMonthlyStackError(ValueError):
    """
    Exception raised when the monthly power spectra stack is inconsistent with the database
    """

    pass


class StackNotInDbError(ValueError):
    """
    Exception raised when the power spectra stack on disk is not in current database
    """

    pass
