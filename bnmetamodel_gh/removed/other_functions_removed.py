__all__ = [
    "numericalSort",
    "loadDataset",
    "loadDataset_sk",
    "generate_training_ver_data",
    "list_to_libpgm_dict",
    "ranges_extreme",
    "valstobins",
    "whichBin",
    "disc2",
    "disc3",
    "disc",
]

import pandas as pd
import numpy as np


import copy
import csv
import io
import numbers
import random
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def numericalSort(value):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.
    # Besides, it will not work as it will throw an AttributeError because
    # "module 'numbers' has no attribute 'split'"
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def loadDataset(
    filename: str, split: float, training_data: List = [], ver_data: List = []
) -> None:
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.
    # Besides, it will not work as it doesn't return any data.

    Load a dataset from a CSV file and split it into training and validation
    sets.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    split : float
        Proportion of the data to be assigned to the training set.
        Value should be between 0.0 and 1.0.
    training_data : list, optional
        A list where the training data rows will be appended. Defaults to an
        empty list.
    ver_data : list, optional
        A list where the validation data rows will be appended. Defaults to an
        empty list.

    Returns
    -------
    None
    """
    with open(filename, "rb") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
    training_data.append(dataset[0])
    ver_data.append(dataset[0])

    for x in range(1, len(dataset) - 1):
        for y in range(len(dataset[x])):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            training_data.append(dataset[x])
        else:
            ver_data.append(dataset[x])

    print("Xtrain_old", training_data)
    print("X_test)old", ver_data)


def loadDataset_sk(
    filename: str, training_data: List = [], ver_data: List = []
) -> Tuple[List, List]:
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Load a dataset from a CSV file and split it into training and validation
    sets using sklearn's train_test_split.

    Parameters
    ----------
    filename : str
        Path to the CSV file.
    training_data : list, optional
        A list where the training data rows will be appended. Defaults to an
        empty list.
    ver_data : list, optional
        A list where the validation data rows will be appended. Defaults to an
        empty list.

    Returns
    -------
    training_data : list
        The training data including the header.
    ver_data : list
        The validation data including the header.
    """
    with open(filename, "rb") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
    header = dataset[0]
    del dataset[0]

    n_dataset = []

    for i in range(1, len(dataset)):
        n_dataset.append([float(j) for j in dataset[i]])

    training_data, ver_data = train_test_split(
        n_dataset, test_size=0.33, random_state=None
    )

    training_data.insert(0, header)
    ver_data.insert(0, header)

    """
    for i in range (1,len(training_data)):
        for j in range (len(training_data[i])):
            float(training_data[i][j])

    for i in range(1,len(ver_data)):
        for j in range(len(ver_data[i])):
            float(ver_data[i][j])
    """
    print("len Xtrain", len(training_data))
    print("len X_test", len(ver_data))
    return training_data, ver_data


def generate_training_ver_data(csv_file_path, num_ver_samples):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Split data from a CSV file into 'training' data and 'verification' data.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file.
    num_ver_samples : int
        Number of verification samples to extract.

    Returns
    -------
    training_data : list
        List of rows (including header) designated as the training set.
    ver_data : list
        List of rows (including header) designated as the verification set.

    Raises
    ------
    ValueError
        If `num_ver_samples` is larger than the number of rows in the CSV minus
        the header.
    """
    # READ CSV DATA

    data = []
    with io.open(csv_file_path, "rb") as f:
        reader = csv.reader(f, dialect=csv.excel)

        for row in reader:
            data.append(row)

    # SPLIT DATA INTO 'TRAINING' DATA AND 'VERIFICATION' DATA
    ver_data = []
    training_data = copy.copy(data)

    # generate random_numbers
    random_numbers = random.sample(
        range(1, len(training_data)), num_ver_samples
    )

    ver_data.append(data[0])

    for i in range(0, len(random_numbers)):
        r = random_numbers[i]
        ver_data.append(training_data[r])
        training_data[r] = 0

    training_data = filter(lambda a: a != 0, training_data)
    return training_data, ver_data


def list_to_libpgm_dict(lst):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Convert a nested list into a list of dictionaries in libpgm format.

    Parameters
    ----------
    lst : list of lists
        The input data where the first list is assumed to be the header (keys
        for the dictionaries), and each subsequent list contains the respective
        values for each key. All the non-header values are assumed to be
        convertible to floats.

    Returns
    -------
    data_array : list of dict
        List of dictionaries where each dictionary represents a row from the
        input data. Keys for the dictionaries are derived from the header of
        the input data.

    Examples
    --------
    >>> data = [["Participant", "Age"], [1, 25], [2, 30]]
    >>> result = list_to_libpgm_dict(data)
    >>> print(result)
    [{'Participant': 1.0, 'Age': 25.0}, {'Participant': 2.0, 'Age': 30.0}]

    Notes
    -----
    The function is designed to be used for preparing data for libpgm.
    """
    try:
        [float(y) for x in lst[1:] for y in x]
    except ValueError:
        raise ValueError(
            "All values of the passed data must be convertible to floats."
        )

    data_array = []
    for i in range(1, len(lst)):
        temp_dict = {}
        for j in range(0, len(lst[i])):
            temp_dict[str(lst[0][j])] = float(lst[i][j])

        data_array.append(temp_dict)

    return data_array


def valstobins(csvData, val_dict, numBins):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    # TODO: Looks like this function will crash if you try to call it, as it
    relies on the bins function which isn't accessible here.

    Map values from `val_dict` to their respective bin indices based on the
    bin ranges calculated from the extreme values in the `csvData`.

    Parameters
    ----------
    csvData : list of lists
        Input csv data where the first row is assumed to be the header (names
        of the variables), and each subsequent row contains the respective
        values for each variable.
    val_dict : dict
        Dictionary where keys are variable names and values are the values that
        need to be mapped to their respective bin indices.
    numBins : int
        The number of bins to be used for discretization.

    Returns
    -------
    output_bins : dict
        A dictionary where keys are variable names from `val_dict` and values are the respective
        bin indices.
    """
    # typical val_dict looks like this: {'A':0.1',
    output_bins = {}

    # extract ranges of bins from extreme ranges
    extreme_ranges_dict = ranges_extreme(csvData)
    extreme_ranges = list(extreme_ranges_dict)

    for key in val_dict.keys():
        min = extreme_ranges_dict[key][0]
        max = extreme_ranges_dict[key][1]

        bin_ranges = bins(max, min, numBins)
        print("bin range for", key, bin_ranges)

        for j in range(0, len(bin_ranges)):
            val_check = val_dict[key]
            print("value to check", val_check)
            bin_min = bin_ranges[j][0]
            bin_max = bin_ranges[j][1]

            if (val_check >= bin_min) and (val_check <= bin_max):
                output_bins[str(key)] = j

    return output_bins


def whichBin(values_list, ranges_list, indexOnly=False):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Maps each value in the `values_list` to its corresponding bin defined by
    the intervals in `ranges_list`.

    Parameters
    ----------
    values_list : list of float
        List of numerical values to be binned.
    ranges_list : list of tuple of float
        List of tuples, where each tuple has two elements representing the
        range (lower and upper bound) of a bin.
    indexOnly : bool, optional (default=False)
        If True, the function returns a list of bin indices for each value in
        `values_list`. If False, the function returns a list of lists where
        each inner list is a binary list indicating the bin of the
        corresponding value from `values_list`.

    Returns
    -------
    bin_index_list : list of int
        List of bin indices corresponding to the `values_list` when
        `indexOnly=True`.
    binned_list : list of list of float
        List of lists, where each inner list is a binary list indicating the
        bin of the corresponding value from `values_list` when
        `indexOnly=False`.

    Examples
    --------
    >>> values = [2.5, 3.5, 4.5]
    >>> ranges = [(0, 3), (3, 4), (4, 5)]
    >>> print(whichBin(values, ranges, True))
    ranges [(0, 3), (3, 4), (4, 5)]
    bin index list [0, 1, 2]
    [0, 1, 2]
    """
    binned_list = []
    bin_index_list = [0] * len(values_list)

    print("ranges", ranges_list)

    for i in range(len(values_list)):
        binned = []
        for k in range(len(ranges_list)):
            binned.append(0.0)

        for j in range(len(ranges_list)):
            if (values_list[i] >= ranges_list[j][0]) & (
                values_list[i] <= ranges_list[j][1]
            ):
                binned[j] = 1.0
                bin_index_list[i] = j

        binned_list.append(binned)

    print("bin index list", bin_index_list)
    if indexOnly == True:
        return bin_index_list
    else:
        return binned_list


def disc2(csv_data, data, alldata, numBins, minmax):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Discretizes continuous data into bins.

    Parameters
    ----------
    csv_data : list of list
        Raw data loaded from CSV, where each inner list represents a row.
    data : list of dict
        The data to be binned, where each dictionary represents a row and
        keys are column names.
    alldata : list of list
        Entire dataset.
    numBins : int
        Number of bins to be used for discretization.
    minmax : dict
        Dictionary with variable names as keys and their respective
        minimum and maximum values as a tuple.

    Returns
    -------
    binned_data : list of dict
        List of dictionaries where each dictionary represents binned data
        for each row in `data`. Keys represent column names and values
        indicate bin index.

    Examples
    --------
    # TODO: This example fails with a ValueError because "min() arg is an
    # empty sequence"
    >>> csv_data = [['A', 'B'], [2.5, 4], [3.5, 5], [4.5, 6]]
    >>> data = [{'A': 2.5, 'B': 4}, {'A': 3.5, 'B': 5}, {'A': 4.5, 'B': 6}]
    >>> alldata = [['A', 'B'], [2.5, 4], [3.5, 5], [4.5, 6]]
    >>> minmax = {'A': (2.5, 4.5), 'B': (4, 6)}
    >>> numBins = 2
    >>> print(disc2(csv_data, data, alldata, numBins, minmax))
    [{'A': 0, 'B': 0}, {'A': 1, 'B': 1}, {'A': 1, 'B': 1}]
    """
    assert (
        isinstance(data, list) and data and isinstance(data[0], dict)
    ), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)

    binned_data = []
    print("csv_data", csv_data)

    df = pd.DataFrame(csv_data)
    df.columns = df.iloc[0]
    df = df[1:]

    print("all data", alldata)

    alldf = pd.DataFrame(alldata)
    alldf.columns = alldf.iloc[0]
    alldf = alldf[1:]

    all_ranges = []
    output_ranges = []
    all_key_strings = df.columns.get_values()
    all_key_strings = all_key_strings.tolist()

    for i in range(len(df.columns)):
        # [[0.5901, 1.072859], [1.072859, 2.220474], [2.220474, 4.197012], [4.197012, 6.620893], [6.620893, 9.349943], [9.349943, 13.694827], [13.694827, 18.286964], [18.286964, 24.310064],
        all_ranges.append(percentile_bins(alldf[alldf.columns[i]], numBins))
        if i == 0:
            output_ranges.append(
                percentile_bins(alldf[alldf.columns[i]], numBins)
            )

    print("all ranges", all_ranges)

    for i in range(0, len(cdata)):
        output_bins = {}
        counter = 0
        for key in cdata[i].keys():
            min = minmax[key][0]
            max = minmax[key][1]

            index = all_key_strings.index(key)

            # TODO #50: Refactor disc2 to no longer hardcode `max_def`
            if key == "max_def":
                # using equal distance discretisation
                bin_ranges = bins(max, min, numBins)
            else:
                # using percentile discretisation
                bin_ranges = all_ranges[index]

            counter = counter + 1

            for k in range(0, len(bin_ranges)):
                val_check = round(cdata[i][key], 6)
                bin_min = bin_ranges[k][0]
                bin_max = bin_ranges[k][1]

                if (val_check >= bin_min) and (val_check <= bin_max):
                    if key not in output_bins:
                        output_bins[str(key)] = k

                if (k == 0) and (val_check < bin_min):
                    output_bins[str(key)] = k

        binned_data.append(output_bins)

    print("binned data", binned_data)
    return binned_data


def disc3(csv_data, data, numBins):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Discretizes continuous data by replacing values with the midpoint of the
    bin they fall into.

    Parameters
    ----------
    csv_data : list of list
        Raw data loaded from CSV, where each inner list represents a row.
    data : list of dict
        The data to be binned, where each dictionary represents a row and
        keys are column names.
    numBins : int
        Number of bins to be used for discretization.

    Returns
    -------
    binned_data : list of dict
        List of dictionaries where each dictionary represents binned data
        for each row in `data`. The original continuous values are replaced
        by the midpoint of the bin they fall into. Keys represent column
        names and values are the midpoint of the bin.

    Examples
    --------
    # TODO: This example fails with a ValueError because "min() arg is an
    # empty sequence"
    >>> csv_data = [['A', 'B'], [2.5, 4], [3.5, 5], [4.5, 6]]
    >>> data = [{'A': 2.5, 'B': 4}, {'A': 3.5, 'B': 5}, {'A': 4.5, 'B': 6}]
    >>> numBins = 2
    >>> print(disc3(csv_data, data, numBins))
    [{'A': 3.0, 'B': 4.5}, {'A': 4.0, 'B': 5.5}, {'A': 4.0, 'B': 5.5}]
    """
    assert (
        isinstance(data, list) and data and isinstance(data[0], dict)
    ), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)

    extreme_ranges_dict = ranges_extreme(csv_data)

    binned_data = []

    for i in range(0, len(cdata)):
        output_bins = {}
        for key in cdata[i].keys():
            min = extreme_ranges_dict[key][0]
            max = extreme_ranges_dict[key][1]

            bin_ranges = bins(max, min, numBins)

            for k in range(0, len(bin_ranges)):
                val_check = round(cdata[i][key], 6)
                bin_min = bin_ranges[k][0]
                bin_max = bin_ranges[k][1]

                if (val_check >= bin_min) and (val_check <= bin_max):
                    output_bins[str(key)] = ((bin_max - bin_min) / 2) + bin_min

        binned_data.append(output_bins)

    return binned_data


def disc(data, bins):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Discretizes continuous data into specified number of bins and replaces the
    original values with bin indices.

    Parameters
    ----------
    data : list of dict
        The data to be binned, where each dictionary represents a row and
        keys are variable names.
    bins : int
        Number of bins to be used for discretization.

    Returns
    -------
    cdata : list of dict
        List of dictionaries where each dictionary represents binned data for
        each row in `data`. The original continuous values are replaced by bin
        indices. Keys represent variable names and values are bin indices.

    Examples
    --------
    >>> data = [{'A': 2.5, 'B': 4}, {'A': 3.5, 'B': 5}, {'A': 4.5, 'B': 6}]
    >>> bins = 2
    >>> print(disc(data, bins))
    [{'A': 0, 'B': 0}, {'A': 0, 'B': 0}, {'A': 1, 'B': 1}]
    """
    assert (
        isinstance(data, list) and data and isinstance(data[0], dict)
    ), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)

    # establish ranges
    ranges = dict()
    for variable in cdata[0].keys():
        ranges[variable] = [float("infinity"), float("infinity") * -1]

    for sample in cdata:
        for var in sample.keys():
            if sample[var] < ranges[var][0]:
                ranges[var][0] = sample[var]
            if sample[var] > ranges[var][1]:
                ranges[var][1] = sample[var]

    for sample in cdata:
        for i in range(bins):
            for var in sample.keys():
                if sample[var] >= (
                    ranges[var][0]
                    + (ranges[var][1] - ranges[var][0]) * i / float(bins)
                ) and (
                    sample[var]
                    <= (
                        ranges[var][0]
                        + (ranges[var][1] - ranges[var][0])
                        * (i + 1)
                        / float(bins)
                    )
                ):
                    sample[var] = i

    return cdata


def ranges_extreme(csvData):
    """
    # TODO: This function is not used anywhere in the codebase. Can be removed.

    Calculate the minimum and maximum values for each variable in the given
    csv data.

    Parameters
    ----------
    csvData : list of lists
        Input csv data where the first row is assumed to be the header (names
        of the variables), and each subsequent row contains the respective
        values for each variable. All the non-header values are assumed to be
        convertible to floats.

    Returns
    -------
    ranges : dict
        A dictionary where keys are variable names and values are lists with
        two elements: the minimum and the maximum value of the respective
        variable.
    """
    ranges = {}

    data = copy.deepcopy(csvData)
    data = zip(*data)

    for i in range(0, len(data)):
        var_name = data[i][0]
        data[i] = list(data[i])
        data[i].remove(data[i][0])
        data[i] = map(float, data[i])
        ranges[str(var_name)] = [
            float(min(list(data[i]))),
            float(max(list(data[i]))),
        ]

    return ranges


def discretize(data, vars_to_discretize, n_bins):
    """
    Discretizes selected variables in the data based on the specified method.

    Parameters
    ----------
    data : dict or DataFrame
        Input data where keys/variables represent columns and values represent
        the data.
    vars_to_discretize : dict
        Dictionary mapping variables to their discretization method.
        The method can be 'Equal', 'Freq', or 'Bins'.
    n_bins : dict
        Dictionary mapping variables to either the number of bins (for 'Equal'
        and 'Freq') or explicit bin edges (for 'Bins').

    Returns
    -------
    data_subset : DataFrame
        The input data with the selected variables discretized.
    bins : dict
        A dictionary containing bin definitions for each of the discretized
        variables.

    Examples
    --------
    >>> data = {'A': [1,2,3,4,5], 'B': [10,20,30,40,50]}
    >>> vars_to_discretize = {'A': 'Equal', 'B': 'Freq'}
    >>> n_bins = {'A': 2, 'B': 3}
    >>> discretized_data, bins_definitions = discretize(data, vars_to_discretize, n_bins)
    A
    B
    >>> print(discretized_data)
       A  B
    0  0  0
    1  0  0
    2  0  1
    3  1  2
    4  1  2
    >>> print(bins_definitions)
    {'A': array([0.996, 3.   , 5.   ]), 'B': array([10.        , 23.33333333, 36.66666667, 50.        ])}

    Notes
    -----
    - When using 'Equal', it discretizes by splitting into equal intervals.
    - When using 'Freq', it discretizes by frequency.
    - When using 'Bins', it discretizes based on provided bin margins.
    """
    data_subset = pd.DataFrame(data).copy()
    bins = {}
    for i in vars_to_discretize:
        out = None
        binning = None

        # discretize by splitting into equal intervals
        if vars_to_discretize[i] == "Equal":
            out, binning = pd.cut(
                data_subset.loc[:, i],
                bins=n_bins[i],
                labels=False,
                retbins=True,
            )

        # discretize by frequency
        elif vars_to_discretize[i] == "Freq":
            nb = n_bins[i]
            while True:
                try:
                    out, binning = pd.qcut(
                        data_subset.loc[:, i], q=nb, labels=False, retbins=True
                    )
                    break
                except:
                    nb -= 1

        # discretize based on provided bin margins
        elif vars_to_discretize[i] == "Bins":
            out = np.digitize(data_subset.loc[:, i], n_bins[i], right=True) - 1
            binning = n_bins[i]

        data_subset.loc[:, i] = out

        # replace NA variables with and special index (1+max) -
        # if it has not been done so automatically an in np.digitize
        data_subset.loc[:, i][data_subset.loc[:, i].isnull()] = (
            data_subset.loc[:, i].max() + 1
        )
        bins[i] = binning

    return data_subset, bins
