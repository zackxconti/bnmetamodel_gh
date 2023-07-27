from Helper_functions import *
import pandas
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.factory import Factory
from pybbn.graph.jointree import EvidenceType
import pandas as pd
import copy
from sklearn.model_selection import KFold

class BayesianNetwork:

    def __init__(self, BNdata=None, netStructure=None, modeldata=None, targetlist=None, binranges=None, priors=None):
        # ***************** #
        # CONVENTION: data and binnedData are stored in dataframes
        # ***************** #

        ## CASE: load model from already built BN
        if modeldata != None:
            print " model data has been supplied "
            #### modeldata should be in json dict format ####
            self.json_data = modeldata
            self.learnedBaynet = DiscreteBayesianNetwork()
            self.nodes = modeldata['V']
            self.edges = modeldata ['E']
            self.Vdata = modeldata ['Vdata']

            self.targets = targetlist
            self.BinRanges = binranges;

        ## CASE: build new model from data supplied via BNdata and netstructure
        else:
            print "model data has not been supplied"
            self.BNdata = BNdata
            self.structure = netStructure
            self.targets = BNdata.targets

            if isinstance(self.structure, basestring): # if structure is passed as a file path
                # load file into skeleton
                skel = GraphSkeleton()
                skel.load(self.structure)
                skel.toporder()
                self.skel = skel
            else:                                      # if structure is passed as loaded graph skeleton
                # given skel
                self.skel = self.structure

            # learn bayesian network
            print 'building bayesian network ...'
            baynet = discrete_mle_estimateparams2(self.skel, BNdata.binnedDict)  # using discrete_mle_estimateparams2 written as function in this file, not calling from libpgm
            # TODO: baynet might be redundant since we are building a junction tree. # Issue #7 --> extract to separate issue

            print 'this is what the libpgm algorithm spits out all data ', self.skel.alldata

            self.learnedBaynet = baynet
            self.nodes = baynet.V
            self.edges = baynet.E
            self.Vdata = baynet.Vdata
            self.json_data = {'V': self.nodes, 'E': self.edges, 'Vdata': self.Vdata}

            self.BinRanges = self.BNdata.binRanges

            print 'building bayesian network complete'

        print 'json data ', self.json_data

        # create BN with pybbn
        bbn = Factory.from_libpgm_discrete_dictionary(self.json_data)

        print 'building junction tree ...'
        # create join tree (this must be computed once)
        self.join_tree = InferenceController.apply(bbn)
        print "building junction tree is complete"

    def generate(self):  # need to modify to accept skel or skelfile
        baynet = discrete_mle_estimateparams2(self.skel, self.binnedData)  # using discrete_mle_estimateparams2 written as function in this file, not calling from libpgm

        self.nodes = baynet.V
        self.edges = baynet.E
        self.Vdata = baynet.Vdata

        return baynet

    def getpriors (self):
        priorPDs = {}

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        return priorPDs

    def inferPD(self, query, evidence, plot=False):
        print 'performing inference ...'
        print 'building conditional probability table ...'

        fn = TableCPDFactorization(self.learnedBaynet)
        print 'conditional probability table is completed'
        print 'performing inference with specified hard evidence ...'
        result = condprobve2(fn, query, evidence)

        print 'result ofrom condprobve2 ', result

        queriedMarginalPosteriors = []
        postInferencePDs = {}

        if len(query) > 1:
            probabilities = printdist(result, self.learnedBaynet)
            print 'probabilities from printdist2 ', probabilities

            for varName in query.keys():

                marginalPosterior = probabilities.groupby(varName,as_index=False)['probability'].sum()
                marginalPosterior.sort_values([varName], inplace=True)
                queriedMarginalPosteriors.append(marginalPosterior)
                postInferencePDs[varName] = marginalPosterior['probability'].tolist()

        else:
            marginalPosterior = printdist(result, self.learnedBaynet)
            marginalPosterior.sort_values([query.keys()[0]], inplace=True)
            queriedMarginalPosteriors.append(marginalPosterior)  # to make sure probabilities are listed in order of bins, sorted by first queried variable
            postInferencePDs[query.keys()[0]] = marginalPosterior['probability'].tolist()

        for varName in evidence.keys():
            e = []
            for i in range (0, len(self.BNdata.binRanges[varName])):
                e.append(0.0)

            e[evidence[varName]]=1.0
            postInferencePDs[varName] = e

        print 'inference is complete'
        return queriedMarginalPosteriors, postInferencePDs

    def inferPD_2(self, query, evidence, plot=False): # this function will allow inference with soft evidence
        # evidence is provided in the form of a dict { 'x1': [0.2, 0.1, 0.4, 0.0, 0.3], 'x2': [1.0, 0.0, 0.0, 0.0, 0.0], ...}

        for varName in evidence.keys(): #for each evidence variable
            var = varName
            allStatesQueriedMarginalPosteriors = []
            num_states = len(evidence[var])
            for i in range (0, num_states): # for each state
                e = {var:i}

                print 'performing inference ...'
                print 'building conditional probability table ...'

                #query is list of variables that are being queried
                fn = TableCPDFactorization(self.learnedBaynet)

                print 'conditional probability table is completed'
                print 'performing inference with specified soft evidence ...'

                result = condprobve2(fn, query, e)

                queriedMarginalPosteriors = []

                if len(query) > 1:
                    probabilities = printdist(result, self.learnedBaynet)

                    for varName in query.keys():
                        marginalPosterior = probabilities.groupby(varName, as_index=False)['probability'].sum()
                        marginalPosterior.sort_values([varName], inplace=True)
                        queriedMarginalPosteriors.append(marginalPosterior)  # returns a list of dataframes, each with probability distribution for each queried variable

                else:
                    marginalPosterior = printdist(result, self.learnedBaynet)
                    marginalPosterior.sort_values([query.keys()[0]], inplace=True)
                    queriedMarginalPosteriors.append(marginalPosterior)  # to make sure probabilities are listed in order of bins, sorted by first queried variable

                allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)

        # loop through each state
        assembledPosteriors = [] # convert to dataframe
        assembledP = allStatesQueriedMarginalPosteriors[0] # dummy list of queried PD dicts

        for varName in evidence.keys():  # for each evidence variable

            evidencePD = evidence[varName]
            postInferencePDs = {}
            assembledPosterior = []

            for i, queryVarName in enumerate (query.keys()):
                num_states = len(allStatesQueriedMarginalPosteriors[0][i]['probability'].tolist())

                for j in range (0, num_states):
                    sum = 0
                    for k in range  (0,len(evidencePD)):
                        sum+= allStatesQueriedMarginalPosteriors[k][i]['probability'].tolist()[j]* evidencePD[k]

                    assembledP[i].set_value(j, 'probability', sum) # data frame
                    assembledPosterior.append(sum) # list

                assembledPosteriors.append(assembledPosterior)
                postInferencePDs.update({queryVarName: assembledP[i]['probability'].tolist()})

        #TODO: here need to update BN PDS and set them as priors for infernece with the next evidence variable # Issue #7 --> extract to separate issue

        # for visualising evidence PDs
        for evidenceVarName in evidence.keys():
            postInferencePDs[evidenceVarName] = evidence[evidenceVarName]

        print 'inference is complete'
        return assembledP, postInferencePDs

    def inferPD_3(self, query, evidence, plot=False): # this function will allow inference with soft evidence
        # evidence is provided in the form of a dict { 'x1': [0.2, 0.1, 0.4, 0.0, 0.3], 'x2': [1.0, 0.0, 0.0, 0.0, 0.0], ...}

        ######## GENERATE SEQUENCE DICTIONARY : ALL POSSIBLE COMBINATIONS OF STATES FROM EACH EVIDENCE VARIABLES #######
        allstates = []
        for ev in evidence.keys():
            states = []
            for j in range (len(evidence[ev])):
                states.append(j)
            allstates.append(states)

        sequence = list(itertools.product(*allstates))
        sequenceDict = {}
        for name in evidence.keys():
            sequenceDict[name] = []

        for i in range(0, len(sequence)):
            for j, name in enumerate(evidence.keys()):
                sequenceDict[name].append(sequence[i][j])

        ################################################################################################################

        ################################ PERFORM INFERENCE TO GENERATE QUERIED PDs #####################################
        ################################## FOR EACH SEQUENCE OF HARD EVIDENCE (SHAOWEI' METHOD)#########################
        allStatesQueriedMarginalPosteriors = []

        # access list of states

        #combinations = [[ {var:0}, {var:0}, {var:0}, {var:0}, {var:0}], [0, 0, 0, 0, 1], ...... ]
        #combinations = {var: [0, 1, 2, 3, 4, .............. , 1], var: [0, 1, 2, 3, 4, ....... , 1], ... }

        #For each combination of evidence states
        for i in range (0, len(sequence)):
            e={}
            for var in evidence.keys(): e[var]=sequenceDict[var][i] # dictionary

            #query is list of variables that are being queried
            fn = TableCPDFactorization(self.learnedBaynet)
            result = condprobve2(fn, query, e)

            queriedMarginalPosteriors = []

            if len(query) > 1:
                probabilities = printdist(result, self.learnedBaynet)

                for varName in query.keys():
                    marginalPosterior = probabilities.groupby(varName, as_index=False)['probability'].sum()
                    marginalPosterior.sort_values([varName], inplace=True)
                    queriedMarginalPosteriors.append(marginalPosterior)  # returns a list of dataframes, each with probability distribution for each queried variable
            else:
                marginalPosterior = printdist(result, self.learnedBaynet)
                marginalPosterior.sort_values([query.keys()[0]], inplace=True)
                queriedMarginalPosteriors.append(marginalPosterior)  # to make sure probabilities are listed in order of bins, sorted by first queried variable

            allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)

        # loop through each state
        assembledPosteriors = [] # convert to dataframe
        assembledP = allStatesQueriedMarginalPosteriors[0] # dummy list of queried PD dicts

        postInferencePDs = {}
        assembledPosterior = []

        for i, queryVarName in enumerate (query.keys()): # for each queried PD
            num_states = len(allStatesQueriedMarginalPosteriors[0][i]['probability'].tolist()) #queried states

            for j in range (0, num_states): # for each state in each queried PD
                sum = 0
                for k in range (0, len(sequence)):
                    # sequence (0, 0), (0, 1), (0, 2), ....
                    # sequenceDict = {var: [0, 1, 2, 3, 4, .............. , 1], var: [0, 1, 2, 3, 4, ....... , 1], ... }

                    ev = [] # to hold list of probabilities to be multiplied by the conditional probability

                    for var in evidence.keys():
                        index = sequenceDict[var][k] # index of evidence state
                        ev.append(evidence[var][index]) # calling the inputted probabilities by index

                    if all(v == 0 for v in ev) : continue

                    ######## ATTEMPT TO TRY STOPPING LOOP WHEN MULTIPLIER IS ZERO PROBABILITY TO SAVE SPEED########
                    multipliers = []
                    for e in ev:
                        if e!= 0.0:
                            multipliers.append(e)
                    ########################

                    sum+= (allStatesQueriedMarginalPosteriors[k][i]['probability'].tolist()[j]* (reduce(lambda x, y: x * y, ev)))

                assembledP[i].set_value(j, 'probability', sum) # data frame
                assembledPosterior.append(sum) # list

            # this is a cheating step to order probabilities by index of df ... should be fixed somwehre before. Compare results with pybbn and bayesialab
            for d in assembledP:
                d.sort_index(inplace=True)

            assembledPosteriors.append(assembledPosterior)
            postInferencePDs.update({queryVarName: assembledP[i]['probability'].tolist()})

        # for visualising evidence PDs
        for evidenceVarName in evidence.keys():
            postInferencePDs[evidenceVarName] = evidence[evidenceVarName]

        return assembledP, postInferencePDs

    def inferPD_4(self, query, evidence, plot=False): # this method performs inference with soft evidence using shaowei's method with join tree
        # evidence is provided in the form of a dict { 'x1': [0.2, 0.1, 0.4, 0.0, 0.3], 'x2': [1.0, 0.0, 0.0, 0.0, 0.0], ...}

        ######## GENERATE SEQUENCE DICTIONARY : ALL POSSIBLE COMBINATIONS OF STATES FROM EACH EVIDENCE VARIABLES #######
        allstates = []
        for ev in evidence.keys():
            states = []
            for j in range (len(evidence[ev])):
                states.append(j)
            allstates.append(states)

        sequence = list(itertools.product(*allstates))

        sequenceDict = {}
        for name in evidence.keys():
            sequenceDict[name] = []

        for i in range(0, len(sequence)):
            for j, name in enumerate(evidence.keys()):
                sequenceDict[name].append(sequence[i][j])

        print ' ______________________________________________  sequence dict',sequenceDict

        ################################################################################################################

        ################################ PERFORM INFERENCE TO GENERATE QUERIED PDs #####################################
        ################################## FOR EACH SEQUENCE OF HARD EVIDENCE ####################################
        allStatesQueriedMarginalPosteriors = []

        # access list of states

        #combinations = [[ {var:0}, {var:0}, {var:0}, {var:0}, {var:0}], [0, 0, 0, 0, 1], ...... ]
        #combinations = {var: [0, 1, 2, 3, 4, .............. , 1], var: [0, 1, 2, 3, 4, ....... , 1], ... }

        #For each combination of evidence states

        for i in range (0, len(sequence)):
            e={}
            for var in evidence.keys(): e[var]=sequenceDict[var][i] # dictionary

            queriedMarginalPosteriors = self.inferWithJunctionTree(e)

            allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)

        # loop through each state
        assembledPosteriors = [] # convert to dataframe
        assembledP = allStatesQueriedMarginalPosteriors[0] # dummy list of queried PD dicts

        postInferencePDs = {}
        assembledPosterior = []

        for i, queryVarName in enumerate (query.keys()): # for each queried PD
            num_states = len(allStatesQueriedMarginalPosteriors[0][i]['probability'].tolist()) #queried states

            for j in range (0, num_states): # for each state in each queried PD
                sum = 0
                for k in range (0, len(sequence)):
                    # sequence (0, 0), (0, 1), (0, 2), ....
                    # sequenceDict = {var: [0, 1, 2, 3, 4, .............. , 1], var: [0, 1, 2, 3, 4, ....... , 1], ... }

                    ev = [] # to hold list of probabilities to be multiplied by the conditional probability

                    for var in evidence.keys():
                        index = sequenceDict[var][k] # index of evidence state
                        ev.append(evidence[var][index]) # calling the inputted probabilities by index

                    sum+= (allStatesQueriedMarginalPosteriors[k][i]['probability'].tolist()[j]* (reduce(lambda x, y: x * y, ev)))

                assembledP[i].set_value(j, 'probability', sum) # data frame
                assembledPosterior.append(sum) # list

            # this is a cheating step to order probabilities by index of df ... should be fixed somwehre before. Compare results with pybbn and bayesialab
            for d in assembledP:
                d.sort_index(inplace=True)

            assembledPosteriors.append(assembledPosterior)

            postInferencePDs[list(assembledP[i])[0]] = assembledP[i]['probability'].tolist()

        # for visualising evidence PDs
        for evidenceVarName in evidence.keys():
            postInferencePDs[evidenceVarName] = evidence[evidenceVarName]

        return assembledP, postInferencePDs

    def plotPDs (self, maintitle, xlabel, ylabel, displayplt = False, **kwargs ): # plots the probability distributions
        # code to automatically set the number of columns and rows and dimensions of the figure

        n_totalplots = len(self.nodes)
        print 'num of total plots ', n_totalplots

        if n_totalplots <= 4:
            n_cols = n_totalplots
            n_rows = 1
        else:
            n_cols = 4
            n_rows = n_totalplots % 4
            print 'num rows ', n_rows

        if n_rows == 0: n_rows = n_totalplots/4

        # generate the probability distributions for the prior distributions
        binRanges = self.BNdata.binRanges
        priorPDs = {}

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0])/float(total))

            priorPDs[varName] = priors

        # instantiate a figure as a placaholder for each distribution (axes)
        fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
        fig.suptitle(maintitle, fontsize=8) # title

        # copy node names into new list
        nodessorted = copy.copy(self.nodes)

        # evidence
        evidenceVars = []
        if 'evidence' in kwargs:
            evidenceVars = kwargs['evidence']

            #sort evidence variables to be in the beginning of the list
            for index, var in enumerate (evidenceVars):
                nodessorted.insert(index, nodessorted.pop(nodessorted.index(evidenceVars[index])))

        i = 0
        for varName in nodessorted:
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_axis_bgcolor("whitesmoke")

            xticksv = []
            binwidths = []
            edge = []

            for index, range in enumerate(binRanges[varName]):
                edge.append(range[0])
                binwidths.append(range[1] - range[0])
                xticksv.append(((range[1] - range[0]) / 2) + range[0])
                if index == len(binRanges[varName]) - 1: edge.append(range[1])

            # plot the priors
            ax.bar(xticksv, priorPDs[varName], align='center', width=binwidths, color='black', alpha=0.2, linewidth=0.2)

            # filter out evidence and query to color the bars accordingly (evidence-green, query-red)
            if 'posteriorPD' in kwargs:


                if len(kwargs['posteriorPD'][varName]) > 1:
                    if varName in evidenceVars:
                        ax.bar(xticksv, kwargs['posteriorPD'][varName], align='center', width=binwidths, color='green', alpha=0.2, linewidth=0.2)

                    else:
                        ax.bar(xticksv, kwargs['posteriorPD'][varName], align='center', width=binwidths, color='red', alpha=0.2, linewidth=0.2)

            # TODO: fix xticks .... not plotting all # Issue #7 --> extract to separate issue
            # plt.xlim(edge[0], max(edge))
            plt.xticks([round(e, 4) for e in edge], rotation='vertical')
            plt.ylim(0, 1)
            # plt.show()

            for spine in ax.spines:
                ax.spines[spine].set_linewidth(0)

            ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
            # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.set_title(varName, fontweight="bold", size=6)
            ax.set_ylabel(ylabel, fontsize=7)  # Y label
            ax.set_xlabel(xlabel, fontsize=7)  # X label
            ax.xaxis.set_tick_params(labelsize=6, length =0)
            ax.yaxis.set_tick_params(labelsize=6, length = 0)

            i += 1

        fig.tight_layout()  # Improves appearance a bit.
        fig.subplots_adjust(top=0.85)  # white spacing between plots and title
        # if you want to set backgrond of figure to transpearaent do it here. Use facecolor='none' as argument in savefig ()

        if displayplt == True: plt.show()

    def crossValidate (self, targetList, numFolds):  # returns a list of error dataframes, one for each target
        #perhaps use **kwargs, to ask if data not specified, then use self.binnedData

        error_dict = {}
        # create empty dataframes to store errors for each target
        for target in targetList:
            df_columns = ['NRMSE', 'LogLoss', 'Classification Error', 'Distance Error']
            df_indices = ['Fold_%s' % (num + 1) for num in range(numFolds)]
            error_df = pandas.DataFrame(index=df_indices, columns=df_columns)
            error_df = error_df.fillna(0.0)
            error_df['Distance Error'] = error_df['Distance Error'].astype(object)
            error_df['Classification Error'] = error_df['Classification Error'].astype(object)

            error_dict[target] = error_df

        # specify number of k folds
        kf = KFold(n_splits=numFolds)
        kf.get_n_splits((self.BNdata.dataArray))

        fold_counter = 0
        listRMSE = 0.0
        listLogLoss = 0.0

        # loop through all data and split into training and testing for each fold
        for training_index, testing_index in kf.split(self.BNdata.data):
            print '--------------------- FOLD NUMBER ', fold_counter+1, '  ---------------------'

            trainingData = kfoldToDF(training_index,self.BNdata.data)
            testingData = kfoldToDF(testing_index, self.BNdata.data)

            # bin test/train data
            binRanges = self.BinRanges
            binnedTrainingDict, binnedTrainingData, binCountsTr =discretize(trainingData,binRanges,False)
            binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData,binRanges,False)
            binnedTestingData=binnedTestingData.astype(int)

            # estimate BN parameters
            baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

            queries ={}
            marginalTargetPosteriorsDict = {}
            for target in targetList:
                # assign bin to zero to query distribution (libpgm convention)
                queries[target] = 0
                # create empty list for each target to populate with predicted target posterior distributions
                marginalTargetPosteriorsDict[target] = []

            # In this loop we predict the posterior distributions for each queried target
            # TODO: need to adapt this loop for storing predicted posteriors for each target in the list, and eventually calc error_df for each Target (or into one DF with multiple indices) # Issue #7 --> extract to separate issue

            for i in range (0,binnedTestingData.shape[0]):
                row = binnedTestingDict[i]
                evidence = without_keys(row, queries.keys())
                fn = TableCPDFactorization(baynet)
                result = condprobve2(fn, queries, evidence)

                # if more than 1 target was specified
                if len(queries) > 1:
                    posteriors = printdist(result, baynet)
                    for target in targetList:
                        marginalPosterior = posteriors.groupby(target)['probability'].sum()
                        marginalTargetPosteriorsDict[target].append(marginalPosterior) #might need [probability]

                # if only 1 target was specified
                else:

                    posterior = printdist(result, baynet)
                    posterior.sort_values([targetList[0]],inplace=True) # to make sure probabilities are listed in order of bins, sorted by first queried variable
                    marginalTargetPosteriorsDict[target].append(posterior['probability'])

            # generate accuracy measures at one go
            # for each target
            for key in error_dict.keys():
                rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(marginalTargetPosteriorsDict[key], testingData, binnedTestingData, binRanges, key)

                # add generated measures to error_df (error dataframe)
                error_dict[key]['NRMSE'][fold_counter] = rmse
                error_dict[key]['LogLoss'][fold_counter] = loglossfunction
                error_dict[key]['Distance Error'][fold_counter] = norm_distance_errors
                error_dict[key]['Classification Error'][fold_counter] = correct_bin_probabilities

            fold_counter +=1

        return error_dict

    def crossValidate_JT(self, targetList, numFolds):  # returns a list of error dataframes, one for each target
        # perhaps use **kwargs, to ask if data not specified, then use self.binnedData

        error_dict = {}
        # create empty dataframes to store errors for each target
        for target in targetList:
            df_columns = ['NRMSE', 'LogLoss', 'Classification Error', 'Distance Error']
            df_indices = ['Fold_%s' % (num + 1) for num in range(numFolds)]
            error_df = pandas.DataFrame(index=df_indices, columns=df_columns)
            error_df = error_df.fillna(0.0)
            error_df['Distance Error'] = error_df['Distance Error'].astype(object)
            error_df['Classification Error'] = error_df['Classification Error'].astype(object)

            error_dict[target] = error_df

        # specify number of k folds
        kf = KFold(n_splits=numFolds)
        kf.get_n_splits((self.BNdata.dataArray))

        fold_counter = 0
        listRMSE = 0.0
        listLogLoss = 0.0

        # loop through all data and split into training and testing for each fold
        for training_index, testing_index in kf.split(self.BNdata.data):
            print '--------------------- FOLD NUMBER ', fold_counter + 1, '  ---------------------'

            trainingData = kfoldToDF(training_index, self.BNdata.data)
            testingData = kfoldToDF(testing_index, self.BNdata.data)

            # bin test/train data
            binRanges = self.BinRanges
            binnedTrainingDict, binnedTrainingData, binCountsTr = discretize(trainingData, binRanges, False)
            binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData, binRanges, False)
            binnedTestingData = binnedTestingData.astype(int)

            # estimate BN parameters
            baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

            ###################################### JOIN TREE USING PYBBN ##############################
            # get topology of bn
            json_data = {'V': baynet.V, 'E': baynet.E, 'Vdata': baynet.Vdata}
            # create BN with pybbn
            pybbn = Factory.from_libpgm_discrete_dictionary(self.json_data)
            # create join tree (this must be computed once)
            jt = InferenceController.apply(pybbn)
            ###########################################################################################

            queries = {}
            marginalTargetPosteriorsDict = {}
            for target in targetList:
                # assign bin to zero to query distribution (libpgm convention)
                queries[target] = 0
                # create empty list for each target to populate with predicted target posterior distributions
                marginalTargetPosteriorsDict[target] = []

            # In this loop we predict the posterior distributions for each queried target
            # TODO: need to adapt this loop for storing predicted posteriors for each target in the list, and eventually calc error_df for each Target (or into one DF with multiple indices) # Issue #7 --> extract to separate issue

            for i in range(0, binnedTestingData.shape[0]):
                row = binnedTestingDict[i]
                evidence = without_keys(row, queries.keys())

                result = self.inferPD_JT_hard(evidence)

                # if more than 1 target was specified
                if len(queries) > 1:
                    posteriors = printdist(result, baynet)
                    for target in targetList:
                        marginalPosterior = posteriors.groupby(target)['probability'].sum()
                        marginalTargetPosteriorsDict[target].append(marginalPosterior)  # might need [probability]

                # if only 1 target was specified
                else:
                    posterior = printdist(result, baynet)
                    posterior.sort_values([targetList[0]], inplace=True)  # to make sure probabilities are listed in order of bins, sorted by first queried variable
                    marginalTargetPosteriorsDict[target].append(posterior['probability'])

            # generate accuracy measures at one go
            # for each target
            for key in error_dict.keys():
                rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(
                    marginalTargetPosteriorsDict[key], testingData, binnedTestingData, binRanges, key)

                # add generated measures to error_df (error dataframe)
                error_dict[key]['NRMSE'][fold_counter] = rmse
                error_dict[key]['LogLoss'][fold_counter] = loglossfunction
                error_dict[key]['Distance Error'][fold_counter] = norm_distance_errors
                error_dict[key]['Classification Error'][fold_counter] = correct_bin_probabilities

            fold_counter += 1

        return error_dict

    def validateNew (self, newBNData, targetList):  # returns a list of error dataframes, one for each target
        # perhaps use **kwargs, to ask if data not specified, then use self.binnedData

        error_dict = {}
        # create empty dataframes to store errors for each target
        for target in targetList:
            df_columns = ['NRMSE', 'LogLoss', 'Classification Error', 'Distance Error']
            df_indices = [0]
            error_df = pandas.DataFrame(index=df_indices, columns=df_columns)
            error_df = error_df.fillna(0.0)
            error_df['Distance Error'] = error_df['Distance Error'].astype(object)
            error_df['Classification Error'] = error_df['Classification Error'].astype(object)

            error_dict[target] = error_df

        listRMSE = 0.0
        listLogLoss = 0.0
        trainingData = self.BNdata.data
        testingData = newBNData.data

        # bin test/train data
        binRanges = self.BinRanges
        binnedTrainingDict, binnedTrainingData, binCountsTr =discretize(trainingData,binRanges,False)
        binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData,binRanges,False)
        binnedTestingData=binnedTestingData.astype(int)

        # estimate BN parameters
        baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

        queries ={}
        marginalTargetPosteriorsDict = {}
        for target in targetList:
            # assign bin to zero to query distribution (libpgm convention)
            queries[target] = 0
            # create empty list for each target to populate with predicted target posterior distributions
            marginalTargetPosteriorsDict[target] = []

        # In this loop we predict the posterior distributions for each queried target
        # TODO: need to adapt this loop for storing predicted posteriors for each target in the list, and eventually calc error_df for each Target (or into one DF with multiple indices) # Issue #7 --> extract to separate issue
        for i in range (0,binnedTestingData.shape[0]):
            row = binnedTestingDict[i]
            evidence = without_keys(row, queries.keys())
            fn = TableCPDFactorization(baynet)
            result = condprobve2(fn, queries, evidence)

            # if more than 1 target was specified
            if len(queries) > 1:
                posteriors = printdist(result, baynet)
                for target in targetList:
                    marginalPosterior = posteriors.groupby(target)['probability'].sum()
                    marginalTargetPosteriorsDict[target].append(marginalPosterior) #might need [probability]

            # if only 1 target was specified
            else:

                posterior = printdist(result, baynet)
                posterior.sort_values([targetList[0]],inplace=True) # to make sure probabilities are listed in order of bins, sorted by first queried variable
                marginalTargetPosteriorsDict[target].append(posterior['probability'])

        # generate accuracy measures at one go
        # for each target
        for key in error_dict.keys():
            rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(marginalTargetPosteriorsDict[key], testingData, binnedTestingData, binRanges, key)

            # add generated measures to error_df (error dataframe)
            error_dict[key]['NRMSE'][0] = rmse
            error_dict[key]['LogLoss'][0] = loglossfunction
            error_dict[key]['Distance Error'][0] = norm_distance_errors
            error_dict[key]['Classification Error'][0] = correct_bin_probabilities

        return error_dict

    def inferPD_JT_hard (self, hardEvidence): # method to perform inference with hard evidence using join tree
        # hardEvidence is supplied in the form {'max_def': 5, 'span': 4}
        # converts libpgm to pybnn then use pybnn to run junction tree and then spitback out results for visualising

        print 'performing inference using junction tree algorithm ...'

        # convert soft evidence to hard

        formattedEvidence = {}
        for var in hardEvidence.keys():
            for i in range(0, len(hardEvidence[var])):
                if hardEvidence[var][i] == 1.0: formattedEvidence[var]=i

        print 'formatted evidence ',formattedEvidence

        def potential_to_df(p):
            data = []
            for pe in p.entries:
                v = pe.entries.values()[0]
                p = pe.value
                t = (v, p)
                data.append(t)
            return pd.DataFrame(data, columns=['val', 'p'])

        def potentials_to_dfs(join_tree):
            data = []
            for node in join_tree.get_bbn_nodes():
                name = node.variable.name
                df = potential_to_df(join_tree.get_bbn_potential(node))
                t = (name, df)
                data.append(t)
            return data

        def pybbnToLibpgm_posteriors(pybbnPosteriors):
            posteriors = {}

            for node in pybbnPosteriors:
                var = node[0]
                df = node[1]
                p = df.sort_values(by=['val'])
                posteriors[var] = p['p'].tolist()

            return posteriors # returns a dictionary of dataframes

        # generate list of pybnn evidence
        evidenceList = []

        for e in formattedEvidence.keys():
            ev = EvidenceBuilder() \
                .with_node(self.join_tree.get_bbn_node_by_name(e)) \
                .with_evidence(formattedEvidence[e], 1.0) \
                .build()

            evidenceList.append(ev)

        self.join_tree.unobserve_all()
        self.join_tree.update_evidences(evidenceList)

        posteriors = potentials_to_dfs(self.join_tree)

        # join tree algorithm seems to eliminate bins whose posterior probabilities are zero
        # check for missing bins and add them back

        for posterior in posteriors:
            numbins = len(self.BinRanges[posterior[0]])

            for i in range (0,numbins):
                if float (i) not in posterior[1]['val'].tolist(): # if
                    #print 'bin number ', float(i) ,' was missing '
                    posterior[1].loc[len(posterior[1])] = [float(i), 0.0]
                    continue

        posteriorsDict = pybbnToLibpgm_posteriors(posteriors)
        print 'inference is complete ... posterior distributions were generated successfully'

        return posteriorsDict

    def inferPD_JT_soft (self, softEvidence ): # method to perform inference with soft evidence (virtual) using join tree only
        # TODO: currently you can only enter likelihoods. Need to find way to enter probabilities and convert them to likelihoods. # Issue #7 --> extract to separate issue

        print 'performing inference using junction tree algorithm ...'

        def potential_to_df(p):
            data = []
            for pe in p.entries:
                v = pe.entries.values()[0]
                p = pe.value
                t = (v, p)
                data.append(t)
            return pd.DataFrame(data, columns=['val', 'p'])

        def potentials_to_dfs(join_tree):
            data = []
            for node in join_tree.get_bbn_nodes():
                name = node.variable.name
                df = potential_to_df(join_tree.get_bbn_potential(node))
                t = (name, df)

                data.append(t)
            return data

        def pybbnToLibpgm_posteriors(pybbnPosteriors):
            posteriors = {}

            for node in pybbnPosteriors:
                var = node[0]
                df = node[1]
                p = df.sort_values(by=['val'])
                posteriors[var] = p['p'].tolist()

            return posteriors # returns a dictionary of dataframes

        evidenceList = []

        for evName in softEvidence.keys():
            ev = EvidenceBuilder().with_node(self.join_tree.get_bbn_node_by_name(evName))

            for state, likelihood in enumerate(softEvidence[evName]):
                ev.values[state] = likelihood

            ev = ev.with_type(EvidenceType.VIRTUAL).build() # specify evidenc type as virtual (soft) (likelihoods not probabilities)
            evidenceList.append(ev)

        self.join_tree.unobserve_all()
        self.join_tree.update_evidences(evidenceList)

        posteriors = potentials_to_dfs(self.join_tree) #contains posteriors + evidence distributions

        # join tree algorithm seems to eliminate bins whose posterior probabilities are zero
        # the following checks for missing bins and adds them back

        for posterior in posteriors:
            print 'posssssssterior ', posterior
            numbins = len(self.BinRanges[posterior[0]])

            for i in range (0,numbins):
                if float (i) not in posterior[1]['val'].tolist(): # if
                    #print 'bin number ', float(i) ,' was missing '
                    posterior[1].loc[len(posterior[1])] = [float(i), 0.0]
                    continue

        posteriorsDict = pybbnToLibpgm_posteriors(posteriors)

        print 'inference is complete ... posterior distributions were generated successfully'

        return posteriorsDict  # posteriors + evidence distributions (for visualising)

    def convertEvidence (self, humanEvidence):
        #humanEvidence can either be entered as ranges of interest {v1: [min, max], v2: [min, max]} or hard numbers {v1: [val], v2: [val]}
        #need to return a dict {v1:[0.0, 1.0, 0.2], v2:[0.1, 0.5, 1.0], ...}

        allevidence = {}

        ranges = self.BinRanges

        # loop through variables in list of inputted evidences
        for var in humanEvidence:
            if type(humanEvidence[var]) == list:

                input_range_min = humanEvidence[var][0]
                input_range_max = humanEvidence[var][1]

                #evidence_var = []
                allevidence[var] = [0.0]*len(ranges[var])

                # loop through bin ranges of variable "var"
                for index, binRange in enumerate(ranges[var]):
                    if input_range_min <= binRange[0] <= input_range_max or input_range_min <= binRange[1] <= input_range_max:
                        allevidence[var][index]=1.0

                    if binRange[0] <= input_range_min <= binRange[1] or binRange[0] <= input_range_max <= binRange[1]:
                        allevidence[var][index] = 1.0

        for item in allevidence: print item, ' -- ', allevidence[item]

        return allevidence
