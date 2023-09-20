__all__ = ["BNdata"]

from .Helper_functions import discretize, getBinRanges

import csv
import copy

import pandas as pd

from typing import Any, List, Optional, Union

# TODO #41: Implement MDRM Sensitivity analysis as class


class BNdata:
    """
    Class for loading and discretizing data for use in Bayesian Networks.

    Arguments
    ---------
    csvdata : str | list[str]
        _description_
    targetlist : any  # TODO: fix type
        _description_
    binTypeDict : dict
        _description_
    numBinsDict : dict
        _description_
    verbose : bool, optional
        Whether to print the progress of the learning process.
    """

    def __init__(
        self,
        csvdata: Union[str, List[str]],
        targetlist: Any,  # TODO: fix type
        binTypeDict: dict,
        numBinsDict: dict,
        verbose: Optional[bool] = False,
    ):
        """
        Constructor of the BNdata class.
        """
        # data can either be specified by file path or by list
        if not isinstance(csvdata, str) or isinstance(csvdata, list):
            raise SyntaxError(
                "Passed csvdata must be string or list of strings."
            )

        self.verbose = verbose
        self.targets = targetlist
        self.numBinsDict = numBinsDict
        self.binTypeDict = binTypeDict

        if isinstance(csvdata, str):
            # data is a filepath
            dataset = []
            with open(csvdata, "rt") as csvfile:
                lines = csv.reader(csvfile)

                for row in lines:
                    dataset.append(row)

            csvD = []
            for i in range(0, len(dataset)):
                row = []
                for j in range(0, len(dataset[i])):
                    if i == 0:
                        row.append(dataset[i][j])
                    else:
                        item = float(dataset[i][j])
                        row.append(item)
                csvD.append(row)

            self.dataArray = csvD
            self.data = pd.DataFrame(data=csvD[1:], columns=csvD[0])

        elif isinstance(csvdata, list):
            # data is a list of lists
            self.data = pd.DataFrame(csvdata, header=0)
            self.dataArray = csvdata

        # range discretization using equal or percentile binning
        # binRanges will contain dict with bin ranges
        self.binRanges = getBinRanges(
            self.data, self.binTypeDict, self.numBinsDict
        )
        # range discretization using minimum length description method

        # Binning data
        self.binnedDict, self.binnedData, self.bincountsDict = discretize(
            self.data, self.binRanges, True
        )

    def loadFromCSV(self, header: bool = False) -> List:
        """
        Load data from a CSV file.

        This function reads the specified CSV file and returns its contents in
        a list. If the `header` parameter is set to True, the first row
        (header) of the CSV file will be considered as column names.

        Parameters
        ----------
        header : bool, optional (default=False)
            Whether the CSV has a header row. If True, the first row will be
            treated as column names.

        Returns
        -------
        list
            List of lists containing the data from the CSV file. Each inner
            list corresponds to a row in the CSV. If the `header` is set to
            True, the first inner list will be column names.

        Attributes Updated
        ------------------
        data : list
            List of lists containing the data from the CSV file.

        Examples
        --------
        >>> model = BNdata('path_to_file.csv', targetlist=..., binTypeDict=..., numBinsDict=...)
        >>> data = model.loadFromCSV(header=True)
        >>> print(data[0])  # This should print the column names if header=True
        """
        # TODO #42: Refactor loadFromCSV to include k-fold code
        # (see also #43: Sort out the doubling up of the loadDataFromCSV function)
        dataset = []
        with open(self.file, "rb") as csvfile:
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

        self.data = data

        return self.data

    def discretize(
        self, binRangesDict: dict, plot: bool = False
    ) -> List[dict]:
        """
        Discretizes the data based on specified bin ranges.

        This method takes the data from the instance attribute and bins it
        using the bin ranges specified in the binRangesDict. Once the data has
        been binned, the method updates the binnedData attribute of the
        instance with a list of dictionaries where each dictionary corresponds
        to a row in the discretized DataFrame.

        Parameters
        ----------
        binRangesDict : dict
            A dictionary specifying the bin ranges for each variable. The keys
            of the dictionary correspond to variable names and the values are
            lists of ranges (where each range is itself a list of two values:
            the start and end of the range).

        plot : bool, optional (default=False)
            If set to True, plots will be generated during the discretization
            process.

            NOTE: Although the parameter exists in the method signature, its
            current implementation does not generate any plots.

        Returns
        -------
        List[dict]
            List of dictionaries where each dictionary corresponds to a row in
            the discretized DataFrame.

        Attributes Updated
        ------------------
        binnedData : List[dict]
            Updated with the discretized data.

        Examples
        --------
        >>> model = SomeClassWithThisMethod('path_to_data.csv')
        >>> bin_ranges = {
        ...     "variable1": [[0, 10], [10, 20]],
        ...     "variable2": [[0, 0.5], [0.5, 1]],
        ... }
        >>> discretized_data = model.discretize(bin_ranges)
        """
        # TODO: plot argument is not used, remove it?

        binnedDf = pd.DataFrame().reindex_like(self.data)

        # copy trainingDfDiscterizedRangesDict
        binCountsDict = copy.deepcopy(binRangesDict)

        for key in binCountsDict:
            for bin in binCountsDict[key]:
                del bin[:]
                bin.append(0)

        for varName in list(self.data):
            # load discretized ranges belonging to varName in order to bin in
            discreteRanges = binRangesDict.get(varName)

            index = 0
            for item in self.data[varName]:
                for i in range(len(discreteRanges)):
                    binRange = discreteRanges[i]

                    # Bin training data

                    if binRange[0] <= item <= binRange[1]:
                        # print (f"{item} lies within {binRange}")
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                    if i == 0 and binRange[0] > item:
                        # print(f"the value {item} is smaller than the \
                        #     minimum bin {binRange[0]}")
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                    if i == len(discreteRanges) - 1 and binRange[1] < item:
                        # print(f"the value {item} is larger than the \
                        #     maximum bin {binRange[1]}")
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                index += 1

        # a list of dictionaries
        self.binnedData = binnedDf.to_dict(orient="records")

        if self.verbose:
            # debug messages
            print(f"train binCountdict: {binCountsDict}")
            print(f"binned_trainingData: {self.binnedData}")

        return self.binnedData
