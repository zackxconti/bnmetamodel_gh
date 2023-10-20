class BNdata:

    # def getBinRanges (self, binTypeDict, numBinsDict):
    #     trainingDfDiscterizedRanges = []
    #     trainingDfDiscterizedRangesDict = {}
    #
    #     # loop through variables in trainingDf (columns) to discretize into
    #     # ranges according to trainingDf
    #     for varName in list(self.data):
    #         # if true, discretise variable i, using percentiles, if false,
    #         # discretise using equal bins
    #         if binTypeDict[varName] == 'percentile':
    #             # add to a list
    #             trainingDfDiscterizedRanges.append(
    #                 percentile_bins(self.data[varName],
    #                 numBinsDict.get(varName)))
    #             # add to a dictionary
    #             trainingDfDiscterizedRangesDict[varName] = percentile_bins(
    #                 self.data[varName], numBinsDict.get(varName))
    #         elif 'equal':
    #             # add to a list
    #             trainingDfDiscterizedRanges.append(
    #                 bins(max(self.data[varName]), min(self.data[varName]),
    #                 numBinsDict.get(varName)))
    #             # add to a dictionary
    #             trainingDfDiscterizedRangesDict[varName] = bins(
    #                 max(self.data[varName]), min(self.data[varName]),
    #                 numBinsDict.get(varName))
    #
    #     # update class attribute, while you're at it
    #     self.bin_ranges = trainingDfDiscterizedRangesDict
    #
    #     return trainingDfDiscterizedRangesDict
