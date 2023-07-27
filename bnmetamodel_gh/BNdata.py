import csv
import pandas as pd
from Helper_functions import getBinRanges, discretize
import copy

# TODO #41: Implement MDRM Sensitivity analysis as class

class BNdata:

    def __init__(self, csvdata, targetlist, binTypeDict, numBinsDict ): # data can either be specified by file path or by list
        self.targets = targetlist
        self.numBinsDict = numBinsDict
        self.binTypeDict = binTypeDict

        print 'importing data from csv file ...'

        # if data is a filepath
        if isinstance(csvdata, basestring):
            dataset = []
            with open(csvdata, 'rb') as csvfile:
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

        # else, if data is a list of lists
        elif isinstance (csvdata, list):
            self.data = pd.DataFrame(csvdata, header=0)
            self.dataArray = csvdata

        print 'importing data from csv file completed'

        ## range discretization using equal or percentile binning
        binRanges = getBinRanges(self.data,self.binTypeDict, self.numBinsDict) #returns dict with bin ranges
        self.binRanges = binRanges
        ## range discretization using minimum length description method

        print 'binning data ...'
        datadf, datadict, bincountsdict = discretize(self.data,self.binRanges,True)
        print 'binning data complete'

        self.binnedDict, self.binnedData, self.bincountsDict = datadf, datadict, bincountsdict

    def getBinRanges (self, binTypeDict, numBinsDict):
        trainingDfDiscterizedRanges = []
        trainingDfDiscterizedRangesDict = {}

        # loop through variables in trainingDf (columns) to discretize into ranges according to trainingDf
        for varName in list(self.data):
            # if true, discretise variable i, using percentiles, if false, discretise using equal bins
            if binTypeDict[varName] == 'percentile':
                trainingDfDiscterizedRanges.append(percentile_bins(self.data[varName], numBinsDict.get(varName)))  # adds to a list
                trainingDfDiscterizedRangesDict[varName] = percentile_bins(self.data[varName], numBinsDict.get(varName))  # adds to a dictionary
            elif 'equal':
                trainingDfDiscterizedRanges.append(bins(max(self.data[varName]), min(self.data[varName]),numBinsDict.get(varName)))  # adds to a list
                trainingDfDiscterizedRangesDict[varName] = bins(max(self.data[varName]), min(self.data[varName]),numBinsDict.get(varName))  # adds to a dictionary

        # update class attribute, while you're at it
        self.bin_ranges = trainingDfDiscterizedRangesDict

        return trainingDfDiscterizedRangesDict

    def discretize (self, binRangesDict, plot=False):
        binnedDf = pd.DataFrame().reindex_like(self.data)

        binCountsDict = copy.deepcopy(binRangesDict)  # copy trainingDfDiscterizedRangesDict
        for key in binCountsDict:
            for bin in binCountsDict[key]:
                del bin[:]
                bin.append(0)

        for varName in list(self.data):
            # load discretized ranges belonging to varName in order to bin in
            discreteRanges = binRangesDict.get(varName)

            index = 0
            for item1 in self.data[varName]:
                for i in range(len(discreteRanges)):
                    binRange = discreteRanges[i]

                    ############ bin training data #############

                    if binRange[0] <= item1 <= binRange[1]:
                        # print item1,' lies within ',binRange
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                    if i == 0 and binRange[0] > item1:
                        # print 'the value ', item1, 'is smaller than the minimum bin', binRange[0]
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                    if i == len(discreteRanges) - 1 and binRange[1] < item1:
                        # print 'the value ', item1, 'is larger than the maximum bin', binRange[1]
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                index += 1

        binnedData = binnedDf.to_dict(orient='records') # a list of dictionaries
        self.binnedData = binnedData

        print 'train binCountdict ', binCountsDict
        print 'binned_trainingData ', binnedData
        return binnedData
