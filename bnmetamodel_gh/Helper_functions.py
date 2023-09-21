__all__ = [
    "loadDataFromCSV",
    "bins",
    "percentile_bins",
    "printdist",
    "kfoldToDF",
    "without_keys",
    "distribution_distance_error",
    "discretize",
    "getBinRanges",
    "generateErrors",
    "BNskelFromCSVpybbn",
]


import copy
import csv
import itertools
import math
import operator

import numpy as np
import pandas as pd
import sklearn

from typing import Any, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def loadDataFromCSV(csv_file_path: str, header: bool = False) -> List[Any]:
    """
    Load data from a csv file at `csv_file_path`. The csv file must be
    formatted such that the first row contains the column names if header is
    True, and the first row contains data if header is False.

    Arguments
    ---------
    csv_file_path : str
        The path to the csv file.
    header : bool
        Whether the csv file contains a header row.

    Returns
    -------
    list of any data types
        The data from the csv file.
    """
    # TODO #42: Refactor loadFromCSV to include k-fold code
    # (see also #43: Sort out the doubling up of the loadDataFromCSV function)
    dataset = []
    with open(csv_file_path, "rt") as csvfile:
        lines = csv.reader(csvfile)

        for row in lines:
            dataset.append(row)

    data = []
    if header:
        data.append(dataset[0])

    for i in range(0, len(dataset)):
        row = []
        for j in range(0, len(dataset[i])):
            if i == 0:
                row.append(dataset[i][j])
            else:
                item = float(dataset[i][j])
                row.append(item)
        data.append(row)

    return data


def bins(max: float, min: float, numBins: int) -> List[List[float]]:
    """
    Create a list of bin ranges from `max` to `min` with `numBins` bins.

    Arguments
    ---------
    max : float
        The maximum value of the range.
    min : float
        The minimum value of the range.
    numBins : int
        The number of bins.

    Returns
    -------
    list[list[float, float]]
        A list of bin ranges from `max` to `min` with `numBins` bins.

    Example
    -------
    >>> bins(10, 0, 5)
    [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]]

    >>> bins(8, 2, 3)
    [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]

    >>> bins(5, 3, 1)
    [[3.0, 5.0]]
    """
    bin_ranges = []
    increment = (max - min) / float(numBins)

    for i in range(numBins - 1, -1, -1):
        a = round(max - (increment * i), 20)
        b = round(max - (increment * (i + 1)), 20)
        bin_ranges.append([b, a])

    return bin_ranges


def percentile_bins(array: List, numBins: int) -> List[List[float]]:
    """
    Create a list of percentile bin ranges from `max` to `min` with `numBins`
    bins.

    Arguments
    ---------
    array : list
        The array to be binned.
    numBins : int
        The number of bins.

    Returns
    -------
    list[list[float, float]]
        A list of percentile bin ranges from `max` to `min` with `numBins`
        bins.

    Example
    -------
    >>> data = [i for i in range(1, 101)]
    >>> percentile_bins(data, 4)
    [[1.0, 25.75], [25.75, 50.5], [50.5, 75.25], [75.25, 100.0]]
    >>> percentile_bins(data, 5)
    [[1.0, 20.8], [20.8, 40.6], [40.6, 60.4], [60.4, 80.2], [80.2, 100.0]]
    """
    a = np.array(array)

    percentage = 100.0 / numBins
    bin_widths = [0]
    bin_ranges = []
    for i in range(0, numBins):
        p_min = round((np.percentile(a, (percentage * i))), 20)
        # print(f"p_min {p_min}")
        bin_widths.append(p_min)
        p_max = round((np.percentile(a, (percentage * (i + 1)))), 20)
        # print(f"p_max {p_max}")
        bin_ranges.append([round(p_min, 20), round(p_max, 20)])

    return bin_ranges


def printdist(jd: str, bn, normalize: bool = True) -> pd.DataFrame:
    """
    Get the distribution of the node in `bn` with the name `jd`. If
    `normalize` is True, the distribution will be normalized.

    Arguments
    ---------
    jd : str
        The name of the node.
    bn : DiscreteBayesianNetwork  # TODO: Correct type?
        The Bayesian network.
    normalize : bool
        Whether to normalize the distribution.

    Returns
    -------
    pandas.DataFrame
        The distribution of the node in `bn` with the name `jd`. If
        `normalize` is True, the distribution will be normalized.
    """
    x = [bn.Vdata[i]["vals"] for i in jd.scope]
    s = sum(jd.vals)
    zipover = [i / s for i in jd.vals] if normalize else jd.vals

    # creates the cartesian product
    k = [
        a + [b]
        for a, b in zip(
            [list(i) for i in itertools.product(*x[::-1])], zipover
        )
    ]

    df = pd.DataFrame.from_records(
        k, columns=[i for i in reversed(jd.scope)] + ["probability"]
    )

    return df


def kfoldToDF(indexList: List[int], dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rows from the input dataframe based on provided indices and return
    as a new dataframe.

    Parameters
    ----------
    indexList : list of int
        A list of indices pointing to rows in `dataframe` that should be
        extracted.
    dataframe : pandas.DataFrame
        The source dataframe from which rows will be extracted.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing rows from `dataframe` as indicated by
        `indexList`.
    """
    df = pd.DataFrame(
        index=range(0, len(indexList)), columns=dataframe.columns
    )

    for index, dfindex in enumerate(indexList):
        df.iloc[index] = dataframe.iloc[dfindex]

    return df


def without_keys(dic: dict, keys: List[str]) -> dict:
    """
    Return a copy of dict `d` without the keys in `keys`.

    Arguments
    ---------
    dic : dict
        The dict to be copied.
    keys : list
        The keys to be removed.

    Returns
    -------
    dict
        A copy of dict `d` without the keys in `keys`.
    """
    return {k: v for k, v in dic.items() if k not in keys}


def distribution_distance_error(
    correct_bin_locations: List[int],
    predicted_bin_probabilities: List[List[float]],
    actual_values: List[float],
    bin_ranges: List[Tuple[float, float]],
    plot: bool = False,
) -> List[float]:
    """
    Compute the distance error between the mean of the bin with the highest
    predicted probability and the actual values. The distance error can be
    normalized with the range of bin values.

    Parameters
    ----------
    correct_bin_locations : list of int
        A list of indices indicating the correct bin locations for each actual
        value.
    predicted_bin_probabilities : list of list of float
        A 2D list where each inner list represents the predicted probabilities
        of an item being in each bin.
    actual_values : list of float
        A list of actual values for which the distance error needs to be
        computed.
    bin_ranges : list of tuple of float
        A list where each tuple represents the minimum and maximum value for
        each bin.
    plot : bool, optional
        If True, a histogram of normalized distance errors is plotted. Default
        is False.

    Returns
    -------
    list of float
        A list of normalized distance errors for each actual value.

        If plot is True, a histogram of normalized distance errors is also
        plotted.

    Example
    -------
    >>> correct_bins = [1, 2]
    >>> predicted_probs = [[0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    >>> actual_vals = [15, 35]
    >>> bins = [(0, 10), (10, 20), (20, 30)]
    >>> distribution_distance_error(correct_bins, predicted_probs, actual_vals, bins)
    [5.0, 5.0]
    """
    distance_errors = []
    norm_distance_errors = []
    output_bin_means = []
    for i in range(0, len(bin_ranges)):
        max_bound = bin_ranges[i][1]
        min_bound = bin_ranges[i][0]

        output_bin_means.append(((max_bound - min_bound) * 0.5) + min_bound)

    for i in range(len(correct_bin_locations)):
        probabilities = predicted_bin_probabilities[i]
        index, value = max(
            enumerate(probabilities), key=operator.itemgetter(1)
        )  # finds bin with max probability and returns it's value and index

        # bin containing actual value
        # actual_bin = correct_bin_locations[i]  # INFO: not used

        # distance between actual value and bin mean
        distance_error = abs(output_bin_means[index] - actual_values[i])

        norm_distance_error = (distance_error - bin_ranges[0][0]) / (
            bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0]
        )

        distance_errors.append(distance_error)
        norm_distance_errors.append(
            norm_distance_error * 100
        )  # remove 100 to normalise

    if plot:
        plt.hist(norm_distance_errors, bins=15)
        plt.xlim(-1, 1)
        plt.show()

    return norm_distance_errors


def discretize(
    dataframe: pd.DataFrame, binRangesDict: dict, plot: bool = False
) -> Tuple[List[dict], pd.DataFrame, dict]:
    """
    Discretize the data in a dataframe based on given bin ranges and count
    occurrences within each bin.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe containing the continuous data to be discretized.
    binRangesDict : dict
        A dictionary where keys are column names from the dataframe, and the
        values are lists of lists representing bin ranges. Each list consists
        of two elements: a lower and upper limit for the bin.
    plot : bool, optional
        A flag to indicate if any plots should be produced during the
        discretization process (default is False). Note: This argument is
        accepted but not utilized in the provided function code.

    Returns
    -------
    binnedData : list of dict
        A list where each element is a dictionary. The keys are column names,
        and the values represent the bin indices where each data point belongs.
    binnedDf : pandas.DataFrame
        A dataframe where each data point in the original dataframe is replaced
        with its corresponding bin index.
    binCountsDict : dict
        A dictionary where keys are column names from the dataframe. The values
        are lists of lists, where each inner list contains a single integer
        representing the number of data points that fall within the
        corresponding bin.

    Example
    -------
    >>> df = pd.DataFrame({'A': [5, 15, 25, 35], 'B': [50, 60, 70, 80]})
    >>> bins = {'A': [[0, 10], [10, 20], [20, 30], [30, 40]], 'B': [[40, 60], [60, 80]]}
    >>> binnedData, binnedDf, counts = discretize(df, bins)

    Note
    ----
    For values outside the defined bins, they are placed into the nearest edge
    bin.
    """
    # TODO: remove plot argument, as it is not currently used
    binnedDf = pd.DataFrame().reindex_like(dataframe)

    binCountsDict = copy.deepcopy(
        binRangesDict
    )  # copy trainingDfDiscterizedRangesDict
    for key in binCountsDict:
        for bin in binCountsDict[key]:
            del bin[:]
            bin.append(0)

    for varName in binRangesDict.keys():
        # load discretized ranges belonging to varName in order to bin in
        discreteRanges = binRangesDict.get(varName)

        index = 0
        for item1 in dataframe[varName]:
            for i in range(len(discreteRanges)):
                binRange = discreteRanges[i]

                # Bin training data
                if i == 0:
                    # this is first bin so bin numbers larger or equal than min
                    # num and less or equal than max num (basically, include
                    # min num)
                    if binRange[0] <= item1 <= binRange[1]:
                        # print(f"{item1} is binned within {binRange}")
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1
                else:
                    # this is not first bin bin numbers less or equal to max
                    # num
                    if binRange[0] < item1 <= binRange[1]:
                        # print(f"{item1} is binned within {binRange}")
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                # catch values outside of range (smaller than min)
                if i == 0 and binRange[0] > item1:
                    # print(f"the value {item1} is smaller than the minimum \
                    #     bin {binRange[0]}")
                    binnedDf.iloc[index][varName] = i
                    binCountsDict[varName][i][0] += 1

                # catch values outside of range (larger than max)
                if i == len(discreteRanges) - 1 and binRange[1] < item1:
                    # print(f"the value {item1} is larger than the maximum \
                    #     bin {binRange[1]}")
                    binnedDf.iloc[index][varName] = i
                    binCountsDict[varName][i][0] += 1

            index += 1

    binnedData = binnedDf.to_dict(orient="records")  # a list of dictionaries

    return binnedData, binnedDf, binCountsDict


def getBinRanges(
    dataframe: pd.DataFrame, binTypeDict: dict, numBinsDict: dict
) -> dict:
    """
    Discretize DataFrame columns into specified bin ranges.

    This function processes the provided DataFrame columns, discretizing them
    into bin ranges as specified by the given dictionaries `binTypeDict` and
    `numBinsDict`.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame that contains the data to be discretized.
    binTypeDict : dict
        A dictionary mapping from column names (variables) of the DataFrame to
        bin types. Currently supported bin types are:
        - 'p': Percentile-based bins.
        - 'e': Equal-width bins.
    numBinsDict : dict
        A dictionary mapping from column names (variables) of the DataFrame to
        the number of bins required for that column.

    Returns
    -------
    dict
        A dictionary with column names (variables) as keys. The corresponding
        values are lists containing the computed bin ranges for each variable.

    Notes
    -----
    - The function currently assumes that each column of the DataFrame
      corresponds to a Bayesian Network (BN) node.

    Example
    -------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 6, 7, 8, 9]})
    >>> binTypeDict = {'A': 'p', 'B': 'e'}
    >>> numBinsDict = {'A': 3, 'B': 4}
    >>> getBinRanges(df, binTypeDict, numBinsDict)
    # This will return a dictionary with bin ranges for columns 'A' and 'B'
    # based on the specified bin types and number of bins.
    """
    trainingDfDiscterizedRanges = []
    trainingDfDiscterizedRangesDict = {}

    # loop through variables in trainingDf (columns) to discretize into ranges
    # according to trainingDf

    # TODO #47: Refactor getBinRanges to no longer use names from dataframe but from an original list of BN nodes
    for varName in binTypeDict.keys():
        if binTypeDict[varName] == "p":
            current_bins = percentile_bins(
                dataframe[varName], numBinsDict.get(varName)
            )
        elif "e":
            current_bins = bins(
                max(dataframe[varName]),
                min(dataframe[varName]),
                numBinsDict.get(varName),
            )

        # add to a list
        trainingDfDiscterizedRanges.append(current_bins)

        # add to a dictionary
        trainingDfDiscterizedRangesDict[varName] = current_bins

        # TODO #48: Refactor getBinRanges to include a new option `auto(mlp)`

    return trainingDfDiscterizedRangesDict


def generateErrors(
    predictedTargetPosteriors: List[List[Tuple[float, float]]],
    testingData: pd.DataFrame,
    binnedTestingData: pd.DataFrame,
    binRanges: dict,
    target: str,
) -> Tuple[float, float, List[float], List[float]]:
    """
    Compute error metrics between predicted target posteriors and actual target
    values.

    This function calculates the RMSE, log loss, distribution distance error,
    and the probabilities associated with the correct bins of actual target
    values.

    Parameters
    ----------
    predictedTargetPosteriors : list of list of float
        The predicted posterior probabilities for each target value. Each inner
        list contains the posterior probabilities for the target's bins.
    testingData : pd.DataFrame
        The DataFrame containing the actual target values.
    binnedTestingData : pd.DataFrame
        The DataFrame containing the binned actual target values. The binning
        corresponds to the binRanges.
    binRanges : dict
        A dictionary mapping target names to their respective bin ranges.
    target : str
        The name of the target variable for which the errors are to be
        computed.

    Returns
    -------
    tuple
        - RMSE (float): Root mean square error between the expected values of
          the predicted posteriors and the actual target values.
        - Log Loss (float): Logarithmic loss between the binned actual target
          values and the predicted posteriors.
        - Distribution Distance Errors (list of float): Distribution distance
          errors between the predicted and actual distributions for each target
          value.
        - Correct Bin Probabilities (list of float): Probabilities associated
          with the correct bins for each actual target value.

    Example
    -------
    >>> predicted = [[0.1, 0.9], [0.7, 0.3]]
    >>> testingData = pd.DataFrame({'target': [1, 0]})
    >>> binnedTestingData = pd.DataFrame({'target': [1, 0]})
    >>> binRanges = {'target': [[0, 0.5], [0.5, 1]]}
    >>> generateErrors(predicted, testingData, binnedTestingData, binRanges, 'target')
    # This will return RMSE, log loss, distribution distance errors, and
    # correct bin probabilities for the 'target' variable.
    """
    posteriorPDmeans = []
    for posterior in predictedTargetPosteriors:
        posteriorPDmeans.append(expectedValue((binRanges[target]), posterior))

    mse = sklearn.metrics.mean_squared_error(
        testingData[target], posteriorPDmeans
    )
    rmse = math.sqrt(mse)

    loglossfunction = sklearn.metrics.log_loss(
        binnedTestingData[target],
        predictedTargetPosteriors,
        normalize=True,
        labels=range(0, len(binRanges[target])),
    )
    norm_distance_errors = distribution_distance_error(
        binnedTestingData[target],
        predictedTargetPosteriors,
        testingData[target],
        binRanges[target],
        False,
    )

    correct_bin_probabilities = []
    for p in range(len(testingData[target])):
        correct_bin_probabilities.append(
            predictedTargetPosteriors[p][binnedTestingData[target][p]]
        )

    return (
        float(rmse),
        float(loglossfunction),
        norm_distance_errors,
        correct_bin_probabilities,
    )


def BNskelFromCSVpybbn(
    csvdata: Union[str, List[str]], targets: List[str]
) -> dict:
    """
    Create a Bayesian Network structure from a CSV file or data.

    This function constructs a Bayesian Network structure based on the columns
    present in a CSV file or data. Columns not specified as targets are treated
    as input vertices. The structure is a dictionary where keys are the vertex
    names and the values are lists of connected vertices.

    Parameters
    ----------
    csvdata : str or list
        If a string, it is treated as the filepath to a CSV file. The CSV
        should have its first row as headers, which are taken as vertex names
        for the Bayesian Network. If a list, the first element should be a list
        of headers (vertex names).

    targets : list of str
        A list of strings specifying which columns in the CSV data should be
        considered as target vertices. All other columns are treated as input
        vertices.

    Returns
    -------
    dict
        A dictionary representing the structure of the Bayesian Network. The
        dictionary has vertex names as keys, and the values are lists
        containing names of connected vertices.

    Notes
    -----
    - Edges (connections) are formed between all input vertices and target
      vertices. The direction of edges (who points to who) depends on the
      number of inputs vs. targets.

    Example
    -------
    >>> BNskelFromCSVpybbn("data.csv", ["TargetA", "TargetB"])
    # This will create a Bayesian Network structure using columns in "data.csv"
    # with "TargetA" and "TargetB" as target vertices. The output will be a
    # dictionary representing the structure.

    """
    # TODO #49: Refactor BNskelFromCSV to include swapping direction of too many inputs into a node

    # EXTRACT HEADER STRINGS FROM CSV FILE
    # skel = GraphSkeleton() # libpgm dependency
    BNstructure = {}
    inputVerts = []

    # if data is a filepath
    if isinstance(
        csvdata, str
    ):  # previously (csvdata, basestring) python 2.0 compatability
        dataset = []
        with open(csvdata, "rt") as csvfile:
            lines = csv.reader(csvfile)

            for row in lines:
                dataset.append(row)

        allVertices = dataset[0]

    else:  # TODO this is a risky one, which needs to be more explicit
        allVertices = csvdata[0]

    BNstructure["V"] = allVertices
    # skel.V = allVertices

    structure = {}

    for verts in allVertices:
        if verts not in targets:
            inputVerts.append(verts)
            # structure [verts] =

    # target, each input
    # edges = []  # INFO: not used

    if len(inputVerts) < len(targets):
        for input in inputVerts:
            structure[input] = []
        for target in targets:
            structure[target] = []
            for input in inputVerts:
                structure[target].append(input)
                # edge = [target, input]
                # edges.append(edge)

        # BNstructure ['E'] = edges
        # skel.E = edges
    else:
        for target in targets:
            structure[target] = []
        for input in inputVerts:
            structure[input] = []
            for target in targets:
                structure[input].append(target)
                # edge = [input, target]
                # edges.append(edge)
        # BNstructure['E'] = edges
        # skel.E = edges

    # print('edges\n ',edges)

    # skel.toporder()

    return structure


def potential_to_df(p) -> pd.DataFrame:
    """
    Generates a dataframe from a potential.

    Parameters
    ----------
    p : _type_
        _description_

    Returns
    -------
    pandas.DataFrame
        _description_
    """
    data = []
    for pe in p.entries:
        v = list(pe.entries.values())[0]
        p = pe.value
        t = (v, p)
        data.append(t)
    return pd.DataFrame(data, columns=["val", "p"])


def potentials_to_dfs(
    join_tree, verbose: Optional[bool] = False
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Generates a list of dataframes from a join tree.

    Parameters
    ----------
    join_tree : _type_
        _description_

    Returns
    -------
    list[tuple[str, pandas.DataFrame]]
        A list of tuples, consisting of the variable name and the
        adjoining dataframe. TODO: Contains posteriors + evidence
        distributions?
    """
    data = []
    for node in join_tree.get_bbn_nodes():
        name = node.variable.name
        df = potential_to_df(join_tree.get_bbn_potential(node))
        if verbose:
            print(f"df potentials \n{df}")
        t = (name, df)
        data.append(t)
    return data


def pybbnToLibpgm_posteriors(
    pybbnPosteriors: List[Tuple[str, pd.DataFrame]]
) -> dict:
    """
    Converts pybbn posteriors to libpgm posteriors.

    Parameters
    ----------
    pybbnPosteriors : list[tuple[str, pd.DataFrame]]
        _description_

    Returns
    -------
    dict
        A dictionary of dataframes.
    """
    posteriors = {}

    for node in pybbnPosteriors:
        var = node[0]
        df = node[1]
        p = df.sort_values(by=["val"])
        posteriors[var] = p["p"].tolist()

    return posteriors  # returns a dictionary of dataframes
