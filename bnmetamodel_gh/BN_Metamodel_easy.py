__all__ = ["BN_Metamodel_easy"]

from .BayesianNetwork import BayesianNetwork
from .BNdata import BNdata
from .Helper_functions import BNskelFromCSVpybbn, loadDataFromCSV

from typing import Any, List, Optional


class BN_Metamodel_easy:
    """
    A class used to represent a Bayesian Network Metamodel.

    Parameters
    ----------
    csvdata : str
        The path to the csv file containing the data.
    targets : list
        A list of the target variables.
    verbose : bool, optional
        Whether to print the progress of the learning process. The default is
        False.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(
        self,
        csvdata: str,
        targets: List[Any],  # TODO: fix typing to remove Any
        verbose: Optional[bool] = False,
        **kwargs
    ):
        """
        Constructor for BN_Metamodel_easy class.
        """
        # data can either be specified by file path or by list
        # order:
        # 1) loads csv file
        # 2) builds BN skeleton (topology)
        # 3) specifies bin types /  bin nums
        # 4) prepares data usin BNdata
        # 5) builds BN (data, skel)

        self.verbose = verbose
        self.targets = targets
        self.variables = loadDataFromCSV(csvdata, True)[0]  # load data
        self.binTypeDict = {}
        self.numBinsDict = {}

        # extract skeleton from csv
        BNskel = BNskelFromCSVpybbn(csvdata, targets)

        if "numBinsDict" in kwargs:
            self.numBinsDict = kwargs["numBinsDict"]

        for var in self.variables:
            # TODO: The following if clause seems to have no effect
            if var in targets:
                # default: all distributions are discretized by equal spacing
                self.binTypeDict[var] = "e"

                # default: all distributions have 6 bins by default
                self.numBinsDict[var] = 6
            else:
                # default: all distributions are discretized by equal spacing
                self.binTypeDict[var] = "e"

                # default: all distributions have 6 bins by default
                self.numBinsDict[var] = 6

        data = BNdata(
            csvdata=csvdata,
            targetlist=self.targets,
            binTypeDict=self.binTypeDict,
            numBinsDict=self.numBinsDict,
            verbose=self.verbose,
        )

        self.learnedBaynet = BayesianNetwork(
            BNdata=data, netStructure=BNskel, verbose=self.verbose
        )

    def json(self) -> dict:
        """
        Returns the json representation of the Bayesian Network.

        Returns
        -------
        json_data : dict
            The json representation of the Bayesian Network.
        """
        return self.learnedBaynet.json_data

    def generate(self) -> BayesianNetwork:
        """
        Returns the generated Bayesian Network.

        Returns
        -------
        learnedBaynet : BayesianNetwork
            The generated Bayesian Network.
        """
        return self.learnedBaynet

    # TODO: added @staticmethod but not sure if the method is intended to be
    # static?
    @staticmethod
    def changeNumBinsDict(dic: dict) -> None:
        """
        Update the number of bins dictionary.

        Parameters
        ----------
        dic : dict
            New dictionary specifying the number of bins for each variable.

        Returns
        -------
        None
        """
        BN_Metamodel_easy.numBinsDict = dic

    def inferPD_JT_soft(self, softevidence: dict) -> dict:
        """
        Infer the posterior distribution using Junction Tree with soft
        evidence.

        Parameters
        ----------
        softevidence : dict
            Soft evidence in the form of variable-to-probability mappings.

        Returns
        -------
        dict
            The posterior probabilities of the query variables given the
            evidence.
        """
        posteriors = self.learnedBaynet.inferPD_JT_soft(softevidence)
        self.learnedBaynet.plotPDs(
            xlabel="Ranges ",
            ylabel="Probability",
            maintitle="Posterior Distributions",
            displayplt=True,
            posteriorPD=posteriors,
            evidence=softevidence.keys(),
        )
        return posteriors

    def inferPD_JT_hard(self, hardevidence: dict) -> dict:
        """
        Infer the posterior distribution using Junction Tree with hard
        evidence.

        Parameters
        ----------
        hardevidence : dict
            Hard evidence in the form of variable-to-value mappings.

        Returns
        -------
        dict
            The posterior probabilities of the query variables given the
            evidence.
        """
        posteriors = self.learnedBaynet.inferPD_JT_hard(hardevidence)
        self.learnedBaynet.plotPDs(
            xlabel="Ranges ",
            ylabel="Probability",
            maintitle="Posterior Distributions",
            displayplt=True,
            posteriorPD=posteriors,
            evidence=hardevidence.keys(),
        )
        return posteriors

    def inferPD_VE_hard(self, query: dict, evidence: dict) -> dict:
        """
        Infer the posterior distribution using Variable Elimination with hard
        evidence.

        Parameters
        ----------
        query : dict
            The variables to query.
        evidence : dict
            Hard evidence in the form of variable-to-value mappings.

        Returns
        -------
        dict
            The posterior probabilities of the query variables given the
            evidence.
        """
        _, posteriors = self.learnedBaynet.inferPD(query, evidence)
        self.learnedBaynet.plotPDs(
            xlabel="Ranges ",
            ylabel="Probability",
            maintitle="Posterior Distributions",
            displayplt=True,
            posteriorPD=posteriors,
            evidence=evidence.keys(),
        )
        return posteriors
