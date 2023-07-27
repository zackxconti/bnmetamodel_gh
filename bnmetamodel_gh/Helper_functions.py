# IMPORTED LIBRARIES
# sklearn imports
import sklearn
from sklearn.metrics import mean_squared_error

import csv

# libpgm imports
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization
from libpgm.pgmlearner import PGMLearner

import io
import copy
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import operator
import re
import networkx as nx



def discrete_estimatebn( learner, data, skel, pvalparam=.05, indegree=0.5):
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."

    # learn parameters
    bn = learner.discrete_mle_estimateparams(skel, data)

    return bn


def alphanum_key(s):
    key = re.split(r"(\d+)", s)
    key[1::2] = map(int, key[1::2])
    return key


def len_csvdata(csv_file_path):
    data = []
    with io.open(csv_file_path, 'rb') as f:
        reader = csv.reader(f, dialect=csv.excel)

        for row in reader:
            data.append(row)

    length = len(data)

    return length


def loadDataFromCSV (csv_file_path, header=False):
    # TODO #42: Refactor loadFromCSV to include k-fold code
    # (see also #43: Sort out the doubling up of the loadDataFromCSV function)
    dataset = []
    with open(csv_file_path, 'rb') as csvfile:
        lines = csv.reader(csvfile)

        for row in lines:
            dataset.append(row)

    data = []
    if (header==True): data.append(dataset[0])

    for i in range(0, len(dataset)):
        row = []
        for j in range (0, len(dataset[i])):
            if i==0: row.append(dataset[i][j])
            else:
                item = float(dataset[i][j])
                row.append(item)
        data.append(row)

    return data


def ranges(data):
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."

    cdata = copy.deepcopy(data)

    # establish ranges
    ranges = dict()
    for variable in cdata[0].keys():
        ranges[variable] = [float("infinity"), float("infinity") * -1]

    for sample in cdata:
        for var in sample.keys():
            if sample[var] < ranges[var][0]:
                ranges[var][0] = round(sample[var], 1)
            if sample[var] > ranges[var][1]:
                ranges[var][1] = round(sample[var], 1)

    return ranges


def bins(max, min, numBins):
    bin_ranges = []
    increment = (max - min) / float(numBins)

    for i in range(numBins - 1, -1, -1):
        a = round(max - (increment * i), 20)
        b = round(max - (increment * (i + 1)), 20)
        bin_ranges.append([b, a])

    return bin_ranges


def percentile_bins(array, numBins):
    a = np.array(array)

    percentage = 100.0 / numBins
    bin_widths = [0]
    bin_ranges = []
    for i in range(0, numBins):
        p_min = round ((np.percentile(a, (percentage * i))),20)
        # print 'p_min ', p_min
        bin_widths.append(p_min)
        p_max = round((np.percentile(a, (percentage * (i + 1)))), 20)
        # print 'p_max ', p_max
        bin_ranges.append([round(p_min, 20), round(p_max, 20)])

    return bin_ranges


def draw_barchartpd(binranges, probabilities):
    xticksv = []
    widths = []
    edge = []
    for index, range in enumerate(binranges):
        print 'range ', range
        edge.append(range[0])
        widths.append(range[1]-range[0])
        xticksv.append(((range[1]-range[0])/2)+range[0])
        if index ==len(binranges)-1: edge.append(range[1])

    print 'xticks ', xticksv
    print 'probabilities ', probabilities
    print 'edge ', edge

    b = plt.bar(xticksv, probabilities, align='center', width=widths, color='black', alpha=0.2)

    # plt.xlim(edge[0], max(edge))
    plt.xticks(edge)
    plt.ylim(0, 1)
    plt.show()

    return b


def draw_histograms(df, binwidths, n_rows, n_cols, maintitle, xlabel, ylabel, displayplt = False, saveplt =False ,**kwargs ):
    fig=plt.figure(figsize=((750*n_cols)/220, (750*n_rows)/220  ), dpi=220)
    t = fig.suptitle(maintitle, fontsize=4)
    # t.set_poition(0.5, 1.05)

    #TODO #44: Replace df with probabilities / write bar function

    i = 0
    for var_name in list(df):
        ax=fig.add_subplot(n_rows,n_cols,i+1)

        if isinstance(binwidths, int) == True:
            print 'binwidths ', binwidths
            df[var_name].hist(bins=binwidths, ax=ax, color='black')
        else:
            df[var_name].hist(bins=binwidths[var_name],ax=ax, color='black' )

        ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round' )
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_title(var_name, fontweight="bold", size=6)
        ax.set_ylabel(ylabel, fontsize=4)  # Y label
        ax.set_xlabel(xlabel, fontsize=4)  # X label
        ax.xaxis.set_tick_params(labelsize=4)
        ax.yaxis.set_tick_params(labelsize=4)

        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

        i+=1

    fig.tight_layout()  # Improves appearance a bit.
    fig.subplots_adjust(top=0.85) #white spacing between plots and title
    # if you want to set backgrond of figure to transpearaent do it here. Use facecolor='none' as argument in savefig ()
    if displayplt == True:plt.show()

    if saveplt == True: fig.savefig('/Users/zack_sutd/Dropbox/SUTD/PhD/Thesis/Phase 2/Simple_truss/Plots/'+str(maintitle)+'.png', dpi=400)


def printdist(jd, bn, normalize=True):
    x = [bn.Vdata[i]["vals"] for i in jd.scope]
    s = sum(jd.vals)
    zipover = [i / s for i in jd.vals] if normalize else jd.vals

    # creates the cartesian product
    k = [a + [b] for a, b in zip([list(i) for i in itertools.product(*x[::-1])], zipover)]

    df = pd.DataFrame.from_records(k, columns=[i for i in reversed(jd.scope)] + ['probability'])

    return df


def kfoldToList (indexList, csvData, header):
    list = []
    list.append(header)
    for i in range (0, len(indexList)):
        list.append(csvData[indexList[i]])

    return list


def kfoldToDF (indexList, dataframe):
    df = pd.DataFrame(index = range(0, len(indexList)),columns=dataframe.columns)

    for index, dfindex in enumerate(indexList):
        df.iloc[index] = dataframe.iloc[dfindex]

    return df

def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

def distribution_distance_error(correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges, plot=False):
    distance_errors = []
    norm_distance_errors = []
    output_bin_means = []
    for i in range(0, len(bin_ranges)):
        max_bound = bin_ranges[i][1]
        min_bound = bin_ranges[i][0]

        output_bin_means.append(((max_bound - min_bound) * 0.5) + min_bound)

    for i in range(len(correct_bin_locations)):
        probabilities = predicted_bin_probabilities[i]
        index, value = max(enumerate(probabilities), key=operator.itemgetter(1))  # finds bin with max probability and returns it's value and index
        actual_bin = correct_bin_locations[i]  # bin containing actual value

        # distance between actual value and bin mean
        distance_error = abs(output_bin_means[index] - actual_values[i])

        norm_distance_error = (distance_error - bin_ranges[0][0]) / (
        bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])

        distance_errors.append(distance_error)
        norm_distance_errors.append(norm_distance_error*100) # remove 100 to normalise

    if plot == True:
        plt.hist(norm_distance_errors, bins=15)
        plt.xlim(-1, 1)
        plt.show()

    return norm_distance_errors


def graph_to_pdf(nodes, edges, name):
    '''
    save a plot of the Bayes net graph in pdf
    '''
    G=nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.drawing.nx_pydot.write_dot(G,name + ".dot")
    os.system("dot -Tpdf %s > %s" % (name+'.dot', name+'.pdf'))

def discrete_mle_estimateparams2(graphskeleton, data):
    '''
    Estimate parameters for a discrete Bayesian network with a structure given by *graphskeleton* in order to maximize the probability of data given by *data*. This function takes the following arguments:

        1. *graphskeleton* -- An instance of the :doc:`GraphSkeleton <graphskeleton>` class containing vertex and edge data.
        2. *data* -- A list of dicts containing samples from the network in {vertex: value} format. Example::

                [
                    {
                        'Grade': 'B',
                        'SAT': 'lowscore',
                        ...
                    },
                    ...
                ]

    This function normalizes the distribution of a node's outcomes for each combination of its parents' outcomes. In doing so it creates an estimated tabular conditional probability distribution for each node. It then instantiates a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance based on the *graphskeleton*, and modifies that instance's *Vdata* attribute to reflect the estimated CPDs. It then returns the instance. 

    The Vdata attribute instantiated is in the format seen in :doc:`unittestdict`, as described in :doc:`discretebayesiannetwork`.

    Usage example: this would learn parameters from a set of 200 discrete samples::

        import json

        from libpgm.nodedata import NodeData
        from libpgm.graphskeleton import GraphSkeleton
        from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
        from libpgm.pgmlearner import PGMLearner

        # generate some data to use
        nd = NodeData()
        nd.load("../tests/unittestdict.txt")    # an input file
        skel = GraphSkeleton()
        skel.load("../tests/unittestdict.txt")
        skel.toporder()
        bn = DiscreteBayesianNetwork(skel, nd)
        data = bn.randomsample(200)

        # instantiate my learner 
        learner = PGMLearner()

        # estimate parameters from data and skeleton
        result = learner.discrete_mle_estimateparams(skel, data)

        # output
        print json.dumps(result.Vdata, indent=2)

    '''
    assert (isinstance(graphskeleton, GraphSkeleton)), "First arg must be a loaded GraphSkeleton class."
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Second arg must be a list of dicts."

    # instantiate Bayesian network, and add parent and children data
    bn = DiscreteBayesianNetwork()
    graphskeleton.toporder()
    bn.V = graphskeleton.V
    bn.E = graphskeleton.E
    bn.Vdata = dict()
    for vertex in bn.V:
        bn.Vdata[vertex] = dict()
        bn.Vdata[vertex]["children"] = graphskeleton.getchildren(vertex)
        bn.Vdata[vertex]["parents"] = graphskeleton.getparents(vertex)

        # make placeholders for vals, cprob, and numoutcomes
        bn.Vdata[vertex]["vals"] = []
        if (bn.Vdata[vertex]["parents"] == []):
            bn.Vdata[vertex]["cprob"] = []
        else:
            bn.Vdata[vertex]["cprob"] = dict()

        bn.Vdata[vertex]["numoutcomes"] = 0

    # STEP 1

    # determine which outcomes are possible for each node
    for sample in data:
        for vertex in bn.V:
            if (sample[vertex] not in bn.Vdata[vertex]["vals"]):
                bn.Vdata[vertex]["vals"].append(sample[vertex])
                bn.Vdata[vertex]["numoutcomes"] += 1

    # lay out probability tables, and put a [num, denom] entry in all spots:

    # define helper function to recursively set up cprob table
    def addlevel(vertex, _dict, key, depth, totaldepth):
        if depth == totaldepth:
            _dict[str(key)] = []
            for _ in range(bn.Vdata[vertex]["numoutcomes"]):
                _dict[str(key)].append([0, 0])
            return
        else:
            for val in bn.Vdata[bn.Vdata[vertex]["parents"][depth]]["vals"]:
                ckey = key[:]
                ckey.append(str(val))
                addlevel(vertex, _dict, ckey, depth + 1, totaldepth)

    # STEP 2
    # put [0, 0] at each entry of cprob table
    for vertex in bn.V:
        if (bn.Vdata[vertex]["parents"]):
            root = bn.Vdata[vertex]["cprob"]
            numparents = len(bn.Vdata[vertex]["parents"])
            addlevel(vertex, root, [], 0, numparents)
        else:
            for _ in range(bn.Vdata[vertex]["numoutcomes"]):
                bn.Vdata[vertex]["cprob"].append([0, 0])

    # STEP 3
    # fill out entries with samples:
    for sample in data:
        for vertex in bn.V:
            # compute index of result
            rindex = bn.Vdata[vertex]["vals"].index(sample[vertex])

            # go to correct place in Vdata
            if bn.Vdata[vertex]["parents"]:
                pvals = [str(sample[t]) for t in bn.Vdata[vertex]["parents"]]
                lev = bn.Vdata[vertex]["cprob"][str(pvals)]
            else:
                lev = bn.Vdata[vertex]["cprob"]

            # increase all denominators for the current condition
            for entry in lev:
                entry[1] += 1

            # increase numerator for current outcome
            lev[rindex][0] += 1

    # STEP 4
    ########################### LAPLACE SMOOTHING TO AVOID ZERO DIVISION ERROR WHEN WE HAVE EMPTY BINS #############################
    for vertex in bn.V:
        numBins = bn.Vdata[vertex]['numoutcomes']

        if not (bn.Vdata[vertex]["parents"]):  # has no parents
            for counts in bn.Vdata[vertex]['cprob']:
                counts[0] += 1  # numerator (count)
                counts[1] += numBins  # denomenator (total count)
        else:
            countdict = bn.Vdata[vertex]['cprob']

            for key in countdict.keys():
                for counts in countdict[key]:
                    counts[0]+=1
                    counts[1]+=numBins

            # STEP 5
            """
            # OPTIONAL: converts cprob from dict into df, does laplace smoothing, then (missing) maps back to dict
            bincounts = pd.DataFrame.from_dict(bn.Vdata[vertex]['cprob'], orient='index')

            for columnI in range (0, bincounts.shape[1]):
                for rowI in range (0,bincounts.shape[0]):
                    bincounts[columnI][rowI]=[bincounts[columnI][rowI][0]+1,bincounts[columnI][rowI][1]+numBins]
            """

    # STEP 6
    ######################################################################################

    # convert arrays to floats
    for vertex in bn.V:
        if not bn.Vdata[vertex]["parents"]:
            bn.Vdata[vertex]["cprob"] = [x[0] / float(x[1]) for x in bn.Vdata[vertex]["cprob"]]
        else:
            for key in bn.Vdata[vertex]["cprob"].keys():
                try:
                    bn.Vdata[vertex]["cprob"][key] = [x[0] / float(x[1]) for x in bn.Vdata[vertex]["cprob"][key]]

                # default to even distribution if no data points
                except ZeroDivisionError:

                    bn.Vdata[vertex]["cprob"][key] = [1 / float(bn.Vdata[vertex]["numoutcomes"]) for x in bn.Vdata[vertex]["cprob"][key]]

    # return cprob table with estimated probability distributions
    return bn


def condprobve2(self, query, evidence):
    '''
    Eliminate all variables in *factorlist* except for the ones queried. Adjust all distributions for the evidence given. Return the probability distribution over a set of variables given by the keys of *query* given *evidence*. 

    Arguments:
        1. *query* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what outcome to calculate the probability of. 
        2. *evidence* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what is known about the system.

    Attributes modified:
        1. *factorlist* -- Modified to be one factor representing the probability distribution of the query variables given the evidence.

    The function returns *factorlist* after it has been modified as above.

    Usage example: this code would return the distribution over a queried node, given evidence::

        import json

        from libpgm.graphskeleton import GraphSkeleton
        from libpgm.nodedata import NodeData
        from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
        from libpgm.tablecpdfactorization import TableCPDFactorization

        # load nodedata and graphskeleton
        nd = NodeData()
        skel = GraphSkeleton()
        nd.load("../tests/unittestdict.txt")
        skel.load("../tests/unittestdict.txt")

        # toporder graph skeleton
        skel.toporder()

        # load evidence
        evidence = dict(Letter='weak')
        query = dict(Grade='A')

        # load bayesian network
        bn = DiscreteBayesianNetwork(skel, nd)

        # load factorization
        fn = TableCPDFactorization(bn)

        # calculate probability distribution
        result = fn.condprobve(query, evidence)

        # output
        print json.dumps(result.vals, indent=2)
        print json.dumps(result.scope, indent=2)
        print json.dumps(result.card, indent=2)
        print json.dumps(result.stride, indent=2)

    '''
    assert (isinstance(query, dict) and isinstance(evidence, dict)), "First and second args must be dicts."
    ## need to modify and add 1 to the zeros here but need frequency count

    eliminate = self.bn.V[:]
    for key in query.keys():
        eliminate.remove(key)

    for key in evidence.keys():
        eliminate.remove(key)

    # modify factors to account for E = e
    for key in evidence.keys():
        for x in range(len(self.factorlist)):
            if (self.factorlist[x].scope.count(key) > 0):
                self.factorlist[x].reducefactor(key, evidence[key])
        for x in reversed(range(len(self.factorlist))):
            if (self.factorlist[x].scope == []):
                del (self.factorlist[x])

    # eliminate all necessary variables in the new factor set to produce result
    self.sumproductve(eliminate)

    # normalize result
    summ = 0.0
    lngth = len(self.factorlist.vals)
    for x in range(lngth):
        summ += self.factorlist.vals[x]

    for x in range(lngth):
        a = float(self.factorlist.vals[x])
        a = a / summ

    # return table
    return self.factorlist


def inferPosteriorDistribution(queries, evidence, baynet):
    # TODO #45: Extend inferPosteriorDistribution to handle multiple query nodes
    fn = TableCPDFactorization(baynet)

    result = condprobve2(fn, queries, evidence)  # written here
    print 'result.vals ', result.vals
    probabilities = printdist(result, baynet)
    probabilities.sort_values(['max_def'], inplace=True)  # make sure probabilities are listed in order of bins

    return probabilities


def laplacesmooth(bn):
    # TODO #46: Update laplacesmooth to align with condprobve/lmeestimateparams
    for vertex in bn.V:
        print 'vertex ', vertex
        numBins = bn.Vdata[vertex]['numoutcomes']

        if not (bn.Vdata[vertex]["parents"]):  # has no parents
            for i in range(len(bn.Vdata[vertex]['cprob'])):
                bn.Vdata[vertex]['cprob'][i][0] += 1  # numerator (count)
                bn.Vdata[vertex]['cprob'][i][1] += numBins  # denomenator (total count)
        else:
            for i in range(numBins):
                binindex = [str(float(i))]
                bincounts = bn.Vdata[vertex]['cprob'][str(binindex)]
                for j in range(len(bincounts)):
                    bincounts[j][0] += 1  # numerator (count)
                    bincounts[j][1] += numBins  # denomenator (total count)

    return bn

def buildBN(trainingData, binstyleDict, numbinsDict, **kwargs): # need to modify to accept skel or skelfile
    discretized_training_data, bin_ranges = discretizeTrainingData(trainingData, binstyleDict, numbinsDict, True)
    print 'discret training ',discretized_training_data

    if 'skel'in kwargs:
        # load file into skeleton
        if isinstance(kwargs['skel'], basestring):
            skel = GraphSkeleton()
            skel.load(kwargs['skel'])
            skel.toporder()
        else:
            skel = kwargs['skel']

    # learn bayesian network
    learner = PGMLearner()
    baynet = discrete_mle_estimateparams2(skel,discretized_training_data)  # using discrete_mle_estimateparams2 written as function in this file, not calling from libpgm

    return baynet


def expectedValue (binRanges, probabilities):
    expectedV = 0.0
    for index, binrange in enumerate(binRanges):
        v_max = binrange[0]
        v_min = binrange[1]

        meanBinvalue = ((v_max - v_min) / 2) + v_min

        expectedV += meanBinvalue * probabilities[index]

    return expectedV


def discretize (dataframe, binRangesDict, plot=False):
    binnedDf = pd.DataFrame().reindex_like(dataframe)

    binCountsDict = copy.deepcopy(binRangesDict)  # copy trainingDfDiscterizedRangesDict
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

                ############ bin training data #############
                if i==0: # if this is first bin then bin numbers larger or equal than min num and less or equal than max num (basically, include min num)
                    if binRange[0] <= item1 <= binRange[1]:
                        # print item1,' is binned within ',binRange
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                else: # if not first bin bin numbers less or equal to max num
                    if binRange[0] < item1 <= binRange[1]:
                        # print item1,' is binned within ',binRange
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                # catch values outside of range (smaller than min)
                if i == 0 and binRange[0] > item1:
                    # print 'the value ', item1, 'is smaller than the minimum bin', binRange[0]
                    binnedDf.iloc[index][varName] = i
                    binCountsDict[varName][i][0] += 1

                # catch values outside of range (larger than max)
                if i == len(discreteRanges) - 1 and binRange[1] < item1:
                    # print 'the value ', item1, 'is larger than the maximum bin', binRange[1]
                    binnedDf.iloc[index][varName] = i
                    binCountsDict[varName][i][0] += 1

            index += 1

    binnedData = binnedDf.to_dict(orient='records') # a list of dictionaries

    return binnedData, binnedDf, binCountsDict


def getBinRanges (dataframe, binTypeDict, numBinsDict):
    trainingDfDiscterizedRanges = []
    trainingDfDiscterizedRangesDict = {}

    # loop through variables in trainingDf (columns) to discretize into ranges according to trainingDf

    # TODO #47: Refactor getBinRanges to no longer use names from dataframe but from an original list of BN nodes
    for varName in binTypeDict.keys():
        if binTypeDict[varName] == 'p':
            trainingDfDiscterizedRanges.append(percentile_bins(dataframe[varName], numBinsDict.get(varName)))  # adds to a list
            trainingDfDiscterizedRangesDict[varName] = percentile_bins(dataframe[varName], numBinsDict.get(varName))  # adds to a dictionary
        elif 'e':
            trainingDfDiscterizedRanges.append(bins(max(dataframe[varName]), min(dataframe[varName]),numBinsDict.get(varName)))  # adds to a list
            trainingDfDiscterizedRangesDict[varName] = bins(max(dataframe[varName]), min(dataframe[varName]),numBinsDict.get(varName))  # adds to a dictionary

        # TODO #48: Refactor getBinRanges to include a new option `auto(mlp)`

    return trainingDfDiscterizedRangesDict


def generateErrors (predictedTargetPosteriors, testingData, binnedTestingData, binRanges, target):
    posteriorPDmeans = []
    for posterior in predictedTargetPosteriors:
        posteriorPDmeans.append(expectedValue((binRanges[target]), posterior))

    mse = mean_squared_error(testingData[target], posteriorPDmeans)
    rmse = math.sqrt(mse)

    loglossfunction = sklearn.metrics.log_loss(binnedTestingData[target], predictedTargetPosteriors,normalize=True, labels=range(0, len(binRanges[target])))
    norm_distance_errors = distribution_distance_error(binnedTestingData[target], predictedTargetPosteriors,testingData[target], binRanges[target], False)

    correct_bin_probabilities = []
    for p in range(len(testingData[target])):
        correct_bin_probabilities.append(predictedTargetPosteriors[p][binnedTestingData[target][p]])

    return float(rmse),float(loglossfunction),norm_distance_errors,correct_bin_probabilities


def BNskelFromCSV (csvdata, targets):
    # TODO #49: Refactor BNskelFromCSV to include swapping direction of too many inputs into a node

    ######## EXTRACT HEADER STRINGS FROM CSV FILE ########
    skel = GraphSkeleton()
    BNstructure = {}
    inputVerts = []

    # if data is a filepath
    if isinstance(csvdata, basestring):
        dataset = []
        with open(csvdata, 'rb') as csvfile:
            lines = csv.reader(csvfile)

            for row in lines:
                dataset.append(row)

        allVertices = dataset [0]

    else:
        allVertices = csvdata[0]

    BNstructure ['V'] = allVertices
    skel.V = allVertices

    for verts in allVertices:
        if verts not in targets:
            inputVerts.append(verts)

    # target, each input
    edges = []
    if len(inputVerts) > len(targets):
        for target in targets:

            for input in inputVerts:
                edge = [target, input]
                edges.append(edge)

        BNstructure ['E'] = edges
        skel.E = edges
    else:
        for input in inputVerts:
            for target in targets:
                edge = [input, target]
                edges.append(edge)
        BNstructure['E'] = edges
        skel.E = edges

    skel.toporder()

    return skel
