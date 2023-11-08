__all__ = [
    "BayesianNetwork",
]

from .Helper_functions import (
    discretize,
    generateErrors,
    kfoldToDF,
    potentials_to_dfs,
    printdist,
    pybbnToLibpgm_posteriors,
    without_keys,
)
from .BNdata import BNdata

import copy

import pandas as pd
import matplotlib.pyplot as plt

from pybbn.graph.jointree import EvidenceBuilder
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.factory import Factory
from pybbn.graph.jointree import EvidenceType

from sklearn.model_selection import KFold

from typing import Any, List, Optional


class BayesianNetwork:
    """
    A class used to represent a Bayesian Network.

    Parameters
    ----------
    BNdata : BNdata, optional
        The BNdata object containing the data to be used to build the
        Bayesian Network, by default None.
    netStructure : dict, optional
        The structure of the Bayesian Network. A dictionary where keys are
        names of children and values are list of parent names.
    modeldata : dict, optional
        The model data of the Bayesian Network, by default None.
    targetlist : list, optional
        The list of targets of the Bayesian Network, by default None.
    binranges : _type_, optional
        The bin ranges of the Bayesian Network, by default None.
    verbose : bool, optional
        Whether to print the progress of the learning process, by default
        False.
    """

    def __init__(
        self,
        BNdata: Optional[BNdata] = None,
        netStructure: Optional[dict] = None,
        modeldata: Optional[dict] = None,
        targetlist: Optional[List] = None,
        binranges: Optional[Any] = None,  # TODO: add type (replace Any)
        verbose: Optional[bool] = False,
        # priors: Optional[Any] = None,  # INFO: removed (not in use)
    ):
        """
        Constructor of the BayesianNetwork class.
        """
        self.verbose = verbose

        if modeldata is not None:
            # Load model from already built BN
            if self.verbose:
                print("model data has been supplied")
            self.json_data = modeldata

            #TODO: load learnedBaynet from data using pybbn
            # self.learnedBaynet = DiscreteBayesianNetwork()
            # self.nodes = modeldata['V']
            # self.edges = modeldata ['E']
            # self.Vdata = modeldata ['Vdata']

            self.targets = targetlist
            self.BinRanges = binranges
        else:
            # Build new model from data supplied via BNdata and netStructure
            if self.verbose:
                print("model data has not been supplied")
            self.BNdata = BNdata
            self.structure = netStructure
            self.targets = BNdata.targets

            if isinstance(self.structure, str):
                # structure is passed as a file path, so load file into
                # skeleton
                # TODO: The line below crashes because no import of GraphSkeleton  # noqa
                skel = GraphSkeleton()
                skel.load(self.structure)
                skel.toporder()
                self.skel = skel
            else:
                # structure is passed as loaded graph skeleton so assigne
                # given structure to skeleton
                self.skel = self.structure

            # learn bayesian network
            if self.verbose:
                print("building bayesian network ...")

            binnedData = BNdata.binnedData.map(str)
            baynet = Factory.from_data(netStructure, binnedData)

            # create join tree (this must be computed once)
            if self.verbose:
                print("--> building junction tree ...")
            self.join_tree = InferenceController.apply(baynet)
            if self.verbose:
                print("--> building junction tree is complete")

            self.learnedBaynet = baynet
            self.nodes = [node.variable.name for node in baynet.get_nodes()]

                #TODO: get these properties from pybbn baynet
                # self.edges = list(baynet.edges.values())
                # self.Vdata = baynet.
                # self.json_data = {'V': self.nodes, 'E': self.edges, 'Vdata': self.Vdata}

            self.BinRanges = self.BNdata.binRanges

            if self.verbose:
                print("building bayesian network complete")

    def getpriors(self) -> dict:
        """
        Get the priors of the Bayesian Network.

        Returns
        -------
        dict
            The priors of the Bayesian Network.
        """
        priorPDs = {}

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        return priorPDs

    def plotPDs(
        self,
        maintitle: str,
        xlabel: str,
        ylabel: str,
        displayplt: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Plots the probability distributions.

        Parameters
        ----------
        maintitle : str
            The title for the figure.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        displayplt : bool, optional
            Whether to display the plot, by default False.
        **kwargs
            Keyword arguments. If "evidence" is passed, then .... If
            "posteriorPD" is passed, then ....

        Returns
        -------
        None
            The probability distributions are displayed as a plot.
        """
        # set the number of columns and rows and dimensions of the figure
        n_totalplots = len(self.nodes)
        if self.verbose:
            print("num of total plots ", n_totalplots)

        if n_totalplots <= 4:
            n_cols = n_totalplots
            n_rows = 1
        else:
            n_cols = 4
            n_rows = n_totalplots % 4
            if self.verbose:
                print(f"num rows {n_rows}")

        if n_rows == 0:
            n_rows = n_totalplots / 4

        # generate the probability distributions for the prior distributions
        binRanges = self.BNdata.binRanges
        priorPDs = {}

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        # instantiate a figure as a placaholder for each distribution (axes)
        fig = plt.figure(
            figsize=((200 * n_cols) / 96, (200 * n_rows) / 96),
            dpi=96,
            facecolor="white",
        )
        fig.suptitle(maintitle, fontsize=8)  # title

        # copy node names into new list
        nodessorted = copy.copy(self.nodes)

        # evidence
        evidenceVars = []
        if "evidence" in kwargs:
            evidenceVars = kwargs["evidence"]
            # sort evidence variables to be in the beginning of the list
            for index, var in enumerate(evidenceVars):
                nodessorted.insert(
                    index,
                    nodessorted.pop(nodessorted.index(evidenceVars[index])),
                )

        i = 0
        for varName in nodessorted:
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_facecolor("whitesmoke")

            xticksv = []
            binwidths = []
            edge = []

            for index, range in enumerate(binRanges[varName]):
                edge.append(range[0])
                binwidths.append(range[1] - range[0])
                xticksv.append(((range[1] - range[0]) / 2) + range[0])
                if index == len(binRanges[varName]) - 1:
                    edge.append(range[1])

            # plot the priors
            ax.bar(
                xticksv,
                priorPDs[varName],
                align="center",
                width=binwidths,
                color="black",
                alpha=0.2,
                linewidth=0.2,
            )

            # filter out evidence and query to color the bars accordingly
            # (evidence-green, query-red)
            if "posteriorPD" in kwargs:
                if len(kwargs["posteriorPD"][varName]) > 1:
                    if varName in evidenceVars:
                        ax.bar(
                            xticksv,
                            kwargs["posteriorPD"][varName],
                            align="center",
                            width=binwidths,
                            color="green",
                            alpha=0.2,
                            linewidth=0.2,
                        )

                    else:
                        ax.bar(
                            xticksv,
                            kwargs["posteriorPD"][varName],
                            align="center",
                            width=binwidths,
                            color="red",
                            alpha=0.2,
                            linewidth=0.2,
                        )

            # TODO #38: Fix xticks in rendering posteriors
            # plt.xlim(edge[0], max(edge))
            plt.xticks([round(e, 4) for e in edge], rotation="vertical")
            plt.ylim(0, 1)
            # plt.show()

            for spine in ax.spines:
                ax.spines[spine].set_linewidth(0)

            ax.grid(
                color="0.2",
                linestyle=":",
                linewidth=0.1,
                dash_capstyle="round",
            )
            # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.set_title(varName, fontweight="bold", size=6)
            ax.set_ylabel(ylabel, fontsize=7)  # Y label
            ax.set_xlabel(xlabel, fontsize=7)  # X label
            ax.xaxis.set_tick_params(labelsize=6, length=0)
            ax.yaxis.set_tick_params(labelsize=6, length=0)

            i += 1

        # Improve appearance a bit
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        if displayplt:
            plt.show()

    def crossValidate_JT(self, targetList: List[str], numFolds: int) -> dict:
        """
        Cross-validates the BN using the Junction Tree algorithm.
        Returns a list of error dataframes, one for each target.

        Parameters
        ----------
        targetList : list[str]
            List of target variables as strings.
        numFolds : int
            Number of folds for cross-validation.

        Returns
        -------
        dict
            Dictionary of error dataframes, one for each target.
        """
        # TODO: note that the method currently is unusable as
        # discrete_mle_estimateparams2 is not imported

        # perhaps use **kwargs, to ask if data not specified, then use
        # self.binnedData

        error_dict = {}
        # create empty dataframes to store errors for each target
        for target in targetList:
            df_columns = [
                "NRMSE",
                "LogLoss",
                "Classification Error",
                "Distance Error",
            ]
            df_indices = ["Fold_%s" % (num + 1) for num in range(numFolds)]
            error_df = pd.DataFrame(index=df_indices, columns=df_columns)
            error_df = error_df.fillna(0.0)
            error_df["Distance Error"] = error_df["Distance Error"].astype(
                object
            )
            error_df["Classification Error"] = error_df[
                "Classification Error"
            ].astype(object)

            error_dict[target] = error_df

        # specify number of k folds
        kf = KFold(n_splits=numFolds)
        kf.get_n_splits((self.BNdata.dataArray))

        fold_counter = 0

        for training_index, testing_index in kf.split(self.BNdata.data):
            # loop through all data and split into training and testing for
            # each fold
            if self.verbose:
                print(f"-------- FOLD NUMBER {fold_counter + 1} ----")

            trainingData = kfoldToDF(training_index, self.BNdata.data)
            testingData = kfoldToDF(testing_index, self.BNdata.data)

            # bin test/train data
            binRanges = self.BinRanges
            binnedTrainingDict, _, _ = discretize(
                trainingData, binRanges, False
            )
            binnedTestingDict, binnedTestingData, _ = discretize(
                testingData, binRanges, False
            )
            binnedTestingData = binnedTestingData.astype(int)

            # estimate BN parameters
            baynet = discrete_mle_estimateparams2(
                self.skel, binnedTrainingDict
            )

            # JOIN TREE USING PYBBN ##############################
            # get topology of bn

            # TODO: The json_data variable can be removed as it is not used
            # -- unless it is what should be passed to
            # Factory.from_libpgm_discrete_dictionary rather than
            # self.json_data below?
            # json_data = {
            #     "V": baynet.V,
            #     "E": baynet.E,
            #     "Vdata": baynet.Vdata,
            # }

            # create BN with pybbn
            pybbn = Factory.from_libpgm_discrete_dictionary(self.json_data)

            # create join tree (this must be computed once)
            # TODO: The jt variable can be removed as it is not used.
            # jt = InferenceController.apply(
            #     pybbn
            # )

            queries = {}
            marginalTargetPosteriorsDict = {}
            for target in targetList:
                # assign bin to zero to query distribution (libpgm convention)
                queries[target] = 0

                # create empty list for each target to populate with predicted
                # target posterior distributions
                marginalTargetPosteriorsDict[target] = []

            for i in range(0, binnedTestingData.shape[0]):
                # In this loop, we predict the posterior distributions for
                # each queried target
                # TODO #39: Adapt loops for storing predicted posteriors
                row = binnedTestingDict[i]
                evidence = without_keys(row, queries.keys())

                result = self.inferPD_JT_hard(evidence)

                if len(queries) > 1:
                    # if more than 1 target was specified
                    posteriors = printdist(result, baynet)
                    for target in targetList:
                        probs = posteriors.groupby(target)["probability"]
                        marginalPosterior = probs.sum()

                        # the line below might need [probability]
                        marginalTargetPosteriorsDict[target].append(
                            marginalPosterior
                        )
                else:
                    # if only 1 target was specified
                    posterior = printdist(result, baynet)

                    # to make sure probabilities are listed in order of bins,
                    # sort by first queried variable:
                    posterior.sort_values([targetList[0]], inplace=True)

                    marginalTargetPosteriorsDict[target].append(
                        posterior["probability"]
                    )

            # generate accuracy measures at one go
            # for each target
            for key in error_dict.keys():
                (
                    rmse,
                    loglossfunction,
                    norm_distance_errors,
                    correct_bin_probabilities,
                ) = generateErrors(
                    marginalTargetPosteriorsDict[key],
                    testingData,
                    binnedTestingData,
                    binRanges,
                    key,
                )

                # add generated measures to error_df (error dataframe)
                error_dict[key]["NRMSE"][fold_counter] = rmse
                error_dict[key]["LogLoss"][fold_counter] = loglossfunction
                error_dict[key]["Distance Error"][
                    fold_counter
                ] = norm_distance_errors
                error_dict[key]["Classification Error"][
                    fold_counter
                ] = correct_bin_probabilities

            fold_counter += 1

        return error_dict

    def validateNew(self, newBNData: BNdata, targetList: List[str]) -> dict:
        """
        Validates the BN using the Junction Tree algorithm.
        returns a list of error dataframes, one for each target

        Parameters
        ----------
        newBNData : BNData
            The new BN data.
        targetList : list[str]
            List of target variables.

        Returns
        -------
        dict
            Dictionary of error dataframes, one for each target.
        """
        # TODO: note that the method currently is unusable as
        # discrete_mle_estimateparams2, TableCPDFactorization, and condprobve2
        # are not imported

        # perhaps use **kwargs, to ask if data not specified, then use
        # self.binnedData

        # create empty dataframes to store errors for each target
        error_dict = {}
        for target in targetList:
            df_columns = [
                "NRMSE",
                "LogLoss",
                "Classification Error",
                "Distance Error",
            ]
            df_indices = [0]
            error_df = pd.DataFrame(index=df_indices, columns=df_columns)
            error_df = error_df.fillna(0.0)
            dist_err = error_df["Distance Error"].astype(object)
            class_err = error_df["Classification Error"].astype(object)
            error_df["Distance Error"] = dist_err
            error_df["Classification Error"] = class_err

            error_dict[target] = error_df

        trainingData = self.BNdata.data
        testingData = newBNData.data

        # bin test/train data
        binRanges = self.BinRanges
        binnedTrainingDict, _, _ = discretize(trainingData, binRanges, False)
        binnedTestingDict, binnedTestingData, _ = discretize(
            testingData, binRanges, False
        )
        binnedTestingData = binnedTestingData.astype(int)

        # estimate BN parameters
        baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

        queries = {}
        marginalTargetPosteriorsDict = {}
        for target in targetList:
            # assign bin to zero to query distribution (libpgm convention)
            queries[target] = 0
            # create empty list for each target to populate with predicted
            # target posterior distributions
            marginalTargetPosteriorsDict[target] = []

        # In this loop we predict the posterior distributions for each queried
        # target
        # TODO #39: Adapt loops for storing predicted posteriors
        for i in range(0, binnedTestingData.shape[0]):
            row = binnedTestingDict[i]
            evidence = without_keys(row, queries.keys())
            fn = TableCPDFactorization(baynet)
            result = condprobve2(fn, queries, evidence)

            if len(queries) > 1:
                # more than 1 target was specified
                posteriors = printdist(result, baynet)
                for target in targetList:
                    marginalPosterior = posteriors.groupby(target)[
                        "probability"
                    ].sum()

                    # the line below might need [probability]
                    marginalTargetPosteriorsDict[target].append(
                        marginalPosterior
                    )
            else:
                # only 1 target was specified
                posterior = printdist(result, baynet)

                # make sure probabilities are listed in order of bins, sorted
                # by first queried variable
                posterior.sort_values([targetList[0]], inplace=True)

                marginalTargetPosteriorsDict[target].append(
                    posterior["probability"]
                )

        # generate accuracy measures at one go
        # for each target
        for key in error_dict.keys():
            (
                rmse,
                loglossfunction,
                norm_distance_errors,
                correct_bin_probabilities,
            ) = generateErrors(
                marginalTargetPosteriorsDict[key],
                testingData,
                binnedTestingData,
                binRanges,
                key,
            )

            # add generated measures to error_df (error dataframe)
            error_dict[key]["NRMSE"][0] = rmse
            error_dict[key]["LogLoss"][0] = loglossfunction
            error_dict[key]["Distance Error"][0] = norm_distance_errors
            error_dict[key]["Classification Error"][
                0
            ] = correct_bin_probabilities

        return error_dict

    def inferPD_JT_hard(self, hardEvidence: dict) -> dict:
        """
        Method to perform inference with hard evidence using join tree.

        Parameters
        ----------
        hardEvidence : dict
            Hard evidence in the form of a dictionary:
            ``{'max_def': 5, 'span': 4}``.

        Returns
        -------
        dict
            Posterior distributions in the form of a dictionary of dataframes.
        """
        # converts libpgm to pybnn then use pybnn to run junction tree and
        # then spitback out results for visualising

        if self.verbose:
            print("performing inference using junction tree algorithm ...")

        # convert libpgm evidence to pybbn evidence
        formattedEvidence = {}
        for var in hardEvidence.keys():
            for i in range(0, len(hardEvidence[var])):
                if hardEvidence[var][i] == 1.0:
                    formattedEvidence[var] = i

        if self.verbose:
            print(f"formatted evidence {formattedEvidence}")

        # formattedEvidence = hardEvidence

        # generate list of pybnn evidence
        evidenceList = []

        for e in formattedEvidence.keys():
            ev = EvidenceBuilder() \
                .with_node(self.join_tree.get_bbn_node_by_name(e)) \
                .with_evidence(str(float(formattedEvidence[e])), 1.0) \
                .build()
            evidenceList.append(ev)

        self.join_tree.unobserve_all()
        self.join_tree.update_evidences(evidenceList)

        posteriors = potentials_to_dfs(self.join_tree, self.verbose)

        # join tree algorithm seems to eliminate bins whose posterior
        # probabilities are zero
        # check for missing bins and add them back

        for posterior in posteriors:
            numbins = len(self.BinRanges[posterior[0]])

            for i in range(0, numbins):
                if float(i) not in posterior[1]["val"].astype(float).tolist():  # if
                    # print 'bin number ', float(i) ,' was missing '
                    posterior[1].loc[len(posterior[1])] = [float(i), 0.0]
                    continue

        print ('posteriors \n ', posteriors)


        posteriorsDict = pybbnToLibpgm_posteriors(posteriors)
        if self.verbose:
            print("inference is complete")
            print("posterior distributions were generated successfully")

        return posteriorsDict

    def inferPD_JT_soft(self, softEvidence: dict) -> dict:
        """
        Method to perform inference with soft evidence (virtual) using join
        tree only.

        Parameters
        ----------
        softEvidence : dict
            Soft evidence in the form of a dictionary:
            ``{'max_def': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}``.

        Returns
        -------
        dict
            Posterior distributions in the form of a dictionary of dataframes.
        """

        # TODO #40: Find way to enter probabilities and convert them to likelihoods in inferPD_JT_soft

        if self.verbose:
            print("performing inference using junction tree algorithm ...")

        evidenceList = []

        for evName in softEvidence.keys():
            ev = EvidenceBuilder().with_node(
                self.join_tree.get_bbn_node_by_name(evName)
            )

            for state, likelihood in enumerate(softEvidence[evName]):
                ev.values[state] = likelihood

            # specify evidence type as virtual (soft)
            # (likelihoods not probabilities)
            ev = ev.with_type(EvidenceType.VIRTUAL).build()
            evidenceList.append(ev)

        self.join_tree.unobserve_all()
        self.join_tree.update_evidences(evidenceList)

        # contains posteriors + evidence distributions
        posteriors = potentials_to_dfs(self.join_tree, self.verbose)

        # join tree algorithm seems to eliminate bins whose posterior
        # probabilities are zero
        # the following checks for missing bins and adds them back

        for posterior in posteriors:
            if self.verbose:
                print("posssssssterior ", posterior)
            numbins = len(self.BinRanges[posterior[0]])

            for i in range(0, numbins):
                if float(i) not in posterior[1]["val"].tolist():
                    # print 'bin number ', float(i) ,' was missing '
                    posterior[1].loc[len(posterior[1])] = [float(i), 0.0]
                    continue

        posteriorsDict = pybbnToLibpgm_posteriors(posteriors)

        if self.verbose:
            print("inference is complete")
            print("posterior distributions were generated successfully")

        # posteriors + evidence distributions (for visualising)
        return posteriorsDict

    def convertEvidence(self, humanEvidence):
        """
        Converts human evidence to libpgm evidence.

        Parameters
        ----------
        humanEvidence : dict
            Human evidence in the form of a dictionary, structured either as
            ranges of interest or hard numbers.

            Example of a range of interest (min, max values in a list):

            .. code-block:: json

                {
                    "v1": [10, 20],
                    "v2": [20, 40]
                }

            Example of hard numbers as values:

            .. code-block:: json

                {
                    "v1": [10],
                    "v2": [30]
                }

        Returns
        -------
        dict
            # TODO

            Example:

            .. code-block:: json

                {
                    "v1": [0.0, 1.0, 0.2],
                    "v2": [0.1, 0.5, 1.0],
                }
        """
        allevidence = {}

        ranges = self.BinRanges

        # loop through variables in list of inputted evidences
        for var in humanEvidence:
            if type(humanEvidence[var]) == list:
                input_range_min = humanEvidence[var][0]
                input_range_max = humanEvidence[var][1]

                # evidence_var = []
                allevidence[var] = [0.0] * len(ranges[var])

                # loop through bin ranges of variable "var"
                for index, binRange in enumerate(ranges[var]):
                    if (
                        input_range_min <= binRange[0] <= input_range_max
                        or input_range_min <= binRange[1] <= input_range_max
                    ):
                        allevidence[var][index] = 1.0

                    if (
                        binRange[0] <= input_range_min <= binRange[1]
                        or binRange[0] <= input_range_max <= binRange[1]
                    ):
                        allevidence[var][index] = 1.0

        if self.verbose:
            for item in allevidence:
                print(f"{item} -- {allevidence[item]}")

        return allevidence
