# from .Helper_functions import condprobve2, discrete_mle_estimateparams2

class BayesianNetwork:
    def __init__(
        self,
        BNdata = None,
        netStructure = None,
        modeldata = None,
        targetlist = None,
        binranges = None,
        priors = None,
    ):
        if modeldata is not None:
            pass
        else:
            # using discrete_mle_estimateparams2 written as function in this
            # file, not calling from libpgm
            # baynet = discrete_mle_estimateparams2(
            #   self.skel,
            #   BNdata.binnedDict
            # )
            # # TODO #36: Baynet might be redundant since we are building a junction tree

        # print ('json data ', self.json_data)
        # create BN with pybbn
        # bbn = Factory.from_libpgm_discrete_dictionary(self.json_data)

    # def generate(self):  # need to modify to accept skel or skelfile
    #     # using discrete_mle_estimateparams2 written as function in this
    #     # file, not calling from libpgm
    #     baynet = discrete_mle_estimateparams2(self.skel, self.binnedData)
    #
    #     self.nodes = baynet.V
    #     self.edges = baynet.E
    #     self.Vdata = baynet.Vdata
    #
    #     return baynet

    # def inferPD(self, query, evidence, plot=False):
    #     print ('performing inference ...')
    #     print ('building conditional probability table ...')
    #
    #     fn = TableCPDFactorization(self.learnedBaynet)
    #     print ('conditional probability table is completed')
    #     print ('performing inference with specified hard evidence ...')
    #     result = condprobve2(fn, query, evidence)
    #
    #     print ('result ofrom condprobve2 ', result)
    #
    #     queriedMarginalPosteriors = []
    #     postInferencePDs = {}
    #
    #     if len(query) > 1:
    #         probabilities = printdist(result, self.learnedBaynet)
    #         print ('probabilities from printdist2 ', probabilities)
    #
    #         for varName in query.keys():
    #
    #             grouped = probabilities.groupby(varName,as_index=False)
    #             marginalPosterior = grouped['probability'].sum()
    #             marginalPosterior.sort_values([varName], inplace=True)
    #             queriedMarginalPosteriors.append(marginalPosterior)
    #             probs = marginalPosterior['probability']
    #             postInferencePDs[varName] = probs.tolist()
    #
    #     else:
    #         marginalPosterior = printdist(result, self.learnedBaynet)
    #         marginalPosterior.sort_values([query.keys()[0]], inplace=True)
    #         # to make sure probabilities are listed in order of bins, sorted
    #         # by first queried variable
    #         queriedMarginalPosteriors.append(marginalPosterior)
    #         probs = marginalPosterior['probability']
    #         postInferencePDs[query.keys()[0]] = probs.tolist()
    #
    #     for varName in evidence.keys():
    #         e = []
    #         for i in range (0, len(self.BNdata.binRanges[varName])):
    #             e.append(0.0)
    #
    #         e[evidence[varName]]=1.0
    #         postInferencePDs[varName] = e
    #
    #     print ('inference is complete')
    #     return queriedMarginalPosteriors, postInferencePDs
    #
    # def inferPD_2(self, query, evidence, plot=False):
    #     # this function will allow inference with soft evidence
    #     # evidence is provided in the form of a dict:
    #     # {
    #     #     "x1": [0.2, 0.1, 0.4, 0.0, 0.3],
    #     #     "x2": [1.0, 0.0, 0.0, 0.0, 0.0],
    #     #     ...
    #     # }
    #
    #     for varName in evidence.keys(): #for each evidence variable
    #         var = varName
    #         allStatesQueriedMarginalPosteriors = []
    #         num_states = len(evidence[var])
    #         for i in range (0, num_states): # for each state
    #             e = {var:i}
    #
    #             print ('performing inference ...')
    #             print ('building conditional probability table ...')
    #
    #             #query is list of variables that are being queried
    #             fn = TableCPDFactorization(self.learnedBaynet)
    #
    #             print('conditional probability table is completed')
    #             print('performing inference with specified soft evidence...')
    #
    #             result = condprobve2(fn, query, e)
    #
    #             queriedMarginalPosteriors = []
    #
    #             if len(query) > 1:
    #                 probabilities = printdist(result, self.learnedBaynet)
    #
    #                 for varName in query.keys():
    #                     grouped = probabilities.groupby(
    #                         varName, as_index=False
    #                     )
    #                     marginalPosterior = grouped['probability'].sum()
    #                     marginalPosterior.sort_values(
    #                         [varName], inplace=True
    #                     )
    #                     # returns a list of dataframes, each with probability
    #                     # distribution for each queried variable
    #                     queriedMarginalPosteriors.append(marginalPosterior)
    #
    #             else:
    #                 marginalPosterior = printdist(result, self.learnedBaynet)
    #                 marginalPosterior.sort_values(
    #                     [query.keys()[0]], inplace=True
    #                 )
    #                 # to make sure probabilities are listed in order of bins,
    #                 # sorted by first queried variable
    #                 queriedMarginalPosteriors.append(marginalPosterior)
    #
    #             allStatesQueriedMarginalPosteriors.append(
    #                 queriedMarginalPosteriors
    #             )
    #
    #     # loop through each state
    #     assembledPosteriors = [] # convert to dataframe
    #
    #     # dummy list of queried PD dicts
    #     assembledP = allStatesQueriedMarginalPosteriors[0]
    #
    #     for varName in evidence.keys():  # for each evidence variable
    #
    #         evidencePD = evidence[varName]
    #         postInferencePDs = {}
    #         assembledPosterior = []
    #
    #         for i, queryVarName in enumerate (query.keys()):
    #             probs = allStatesQueriedMarginalPosteriors[0][i]['probability']  # noqa
    #             num_states = len(probs.tolist())
    #
    #             for j in range(0, num_states):
    #                 sum = 0
    #                 for k in range(0,len(evidencePD)):
    #                     probs = allStatesQueriedMarginalPosteriors[k][i]['probability']  # noqa
    #                     sum += probs.tolist()[j]* evidencePD[k]
    #
    #                 # data frame
    #                 assembledP[i].set_value(j, "probability", sum)
    #
    #                 # list
    #                 assembledPosterior.append(sum)
    #
    #             assembledPosteriors.append(assembledPosterior)
    #
    #             probs = assembledP[i]['probability']
    #             postInferencePDs.update({queryVarName: probs.tolist()})
    #
    #     # TODO #37: Update BN PDS and set them as priors for inference with the next evidence variable
    #
    #     # for visualising evidence PDs
    #     for evidenceVarName in evidence.keys():
    #         postInferencePDs[evidenceVarName] = evidence[evidenceVarName]
    #
    #     print ('inference is complete')
    #     return assembledP, postInferencePDs
    #
    # def inferPD_3(self, query, evidence, plot=False):
    #     # this function will allow inference with soft evidence
    #     # evidence is provided in the form of a dict:
    #     # {
    #     #     "x1": [0.2, 0.1, 0.4, 0.0, 0.3],
    #     #     "x2": [1.0, 0.0, 0.0, 0.0, 0.0],
    #     #     ...
    #     # }
    #
    #     # GENERATE SEQUENCE DICTIONARY:
    #     # ALL POSSIBLE COMBINATIONS OF STATES FROM EACH EVIDENCE VARIABLE
    #     allstates = []
    #     for ev in evidence.keys():
    #         states = []
    #         for j in range (len(evidence[ev])):
    #             states.append(j)
    #         allstates.append(states)
    #
    #     sequence = list(itertools.product(*allstates))
    #     sequenceDict = {}
    #     for name in evidence.keys():
    #         sequenceDict[name] = []
    #
    #     for i in range(0, len(sequence)):
    #         for j, name in enumerate(evidence.keys()):
    #             sequenceDict[name].append(sequence[i][j])
    #
    #     ##################################################################
    #
    #     # PERFORM INFERENCE TO GENERATE QUERIED PDs
    #     # FOR EACH SEQUENCE OF HARD EVIDENCE (SHAOWEI' METHOD)
    #     allStatesQueriedMarginalPosteriors = []
    #
    #     # access list of states
    #
    #     # combinations = [
    #     #     [
    #     #         {var:0},
    #     #         {var:0},
    #     #         {var:0},
    #     #         {var:0},
    #     #         {var:0}
    #     #     ],
    #     #     [
    #     #         0, 0, 0, 0, 1
    #     #     ],
    #     #     ...
    #     # ]
    #     # combinations = {
    #     #     var: [0, 1, 2, 3, 4, ... , 1],
    #     #     var: [0, 1, 2, 3, 4, ....... , 1],
    #     #     ...
    #     # }
    #
    #     for i in range (0, len(sequence)):
    #         # Loop through each combination of evidence states
    #         e = {}
    #
    #         # dictionary
    #         for var in evidence.keys(): e[var]=sequenceDict[var][i]
    #
    #         # query is list of variables that are being queried
    #         fn = TableCPDFactorization(self.learnedBaynet)
    #         result = condprobve2(fn, query, e)
    #
    #         queriedMarginalPosteriors = []
    #
    #         if len(query) > 1:
    #             probabilities = printdist(result, self.learnedBaynet)
    #
    #             for varName in query.keys():
    #                 probs = probabilities.groupby(varName, as_index=False)
    #                 marginalPosterior = probs['probability'].sum()
    #                 marginalPosterior.sort_values([varName], inplace=True)
    #                 # returns a list of dataframes, each with probability
    #                 # distribution for each queried variable
    #                 queriedMarginalPosteriors.append(marginalPosterior)
    #         else:
    #             marginalPosterior = printdist(result, self.learnedBaynet)
    #
    #             # to make sure probabilities are listed in order of bins,
    #             # sorted by first queried variable
    #             marginalPosterior.sort_values(
    #                 [query.keys()[0]],
    #                 inplace=True
    #             )
    #
    #             queriedMarginalPosteriors.append(marginalPosterior)
    #
    #         allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)
    #
    #     # loop through each state
    #     assembledPosteriors = [] # convert to dataframe
    #
    #     # dummy list of queried PD dicts
    #     assembledP = allStatesQueriedMarginalPosteriors[0]
    #
    #     postInferencePDs = {}
    #     assembledPosterior = []
    #
    #     for i, queryVarName in enumerate (query.keys()):
    #         # loop through each queried PD
    #         probs = allStatesQueriedMarginalPosteriors[0][i]['probability']
    #
    #         # queried states
    #         num_states = len(probs.tolist())
    #
    #         for j in range (0, num_states):
    #             # loop through each state in each queried PD
    #             sum = 0
    #             for k in range (0, len(sequence)):
    #                 # sequence (0, 0), (0, 1), (0, 2), ....
    #                 # sequenceDict = {
    #                 #     var: [0, 1, 2, 3, 4, .............. , 1],
    #                 #     var: [0, 1, 2, 3, 4, ....... , 1],
    #                 #     ...
    #                 # }
    #
    #                 # to hold list of probabilities to be multiplied by the
    #                 # conditional probability
    #                 ev = []
    #
    #                 for var in evidence.keys():
    #                     index = sequenceDict[var][k] # ix of evidence state
    #                     # calling the inputted probabilities by index:
    #                     ev.append(evidence[var][index])
    #
    #                 if all(v == 0 for v in ev) : continue
    #
    #                 # ATTEMPT TO STOP LOOP WHEN MULTIPLIER IS ZERO
    #                 # PROBABILITY TO SAVE SPEED
    #                 multipliers = []
    #                 for e in ev:
    #                     if e!= 0.0:
    #                         multipliers.append(e)
    #
    #                 probs = allStatesQueriedMarginalPosteriors[k][i]['probability']  # noqa
    #                 red = reduce(lambda x, y: x * y, ev))
    #                 sum += (probs.tolist()[j] * red
    #
    #             # data frame
    #             assembledP[i].set_value(j, 'probability', sum)
    #             # list
    #             assembledPosterior.append(sum)
    #
    #         # this is a cheating step to order probabilities by index of df
    #         # ... should be fixed somwehre before. Compare results with pybbn
    #         # and bayesialab
    #         for d in assembledP:
    #             d.sort_index(inplace=True)
    #
    #         assembledPosteriors.append(assembledPosterior)
    #         postInferencePDs.update({
    #             queryVarName: assembledP[i]['probability'].tolist()
    #         })
    #
    #     # for visualising evidence PDs
    #     for evidenceVarName in evidence.keys():
    #         postInferencePDs[evidenceVarName] = evidence[evidenceVarName]
    #
    #     return assembledP, postInferencePDs
    #
    # def inferPD_4(self, query, evidence, plot=False):
    #     # this method performs inference with soft evidence using shaowei's
    #     # method with join tree
    #     # evidence is provided in the form of a dict:
    #     # {
    #     #     "x1": [0.2, 0.1, 0.4, 0.0, 0.3],
    #     #     "x2": [1.0, 0.0, 0.0, 0.0, 0.0],
    #     #     ...
    #     # }
    #
    #     # GENERATE SEQUENCE DICTIONARY : ALL POSSIBLE COMBINATIONS OF STATES
    #     # FROM EACH EVIDENCE VARIABLES
    #     allstates = []
    #     for ev in evidence.keys():
    #         states = []
    #         for j in range (len(evidence[ev])):
    #             states.append(j)
    #         allstates.append(states)
    #
    #     sequence = list(itertools.product(*allstates))
    #
    #     sequenceDict = {}
    #     for name in evidence.keys():
    #         sequenceDict[name] = []
    #
    #     for i in range(0, len(sequence)):
    #         for j, name in enumerate(evidence.keys()):
    #             sequenceDict[name].append(sequence[i][j])
    #
    #     print('_______________________________ sequence dict', sequenceDict)
    #
    #     # PERFORM INFERENCE TO GENERATE QUERIED PDs
    #     # FOR EACH SEQUENCE OF HARD EVIDENCE
    #     allStatesQueriedMarginalPosteriors = []
    #
    #     # access list of states
    #
    #     # combinations = [
    #     #     [
    #     #         {var:0},
    #     #         {var:0},
    #     #         {var:0},
    #     #         {var:0},
    #     #         {var:0}
    #     #      ],
    #     #      [0, 0, 0, 0, 1],
    #     #      ...
    #     # ]
    #     # combinations = {
    #     #     var: [0, 1, 2, 3, 4, ... , 1],
    #     #     var: [0, 1, 2, 3, 4, ....... , 1],
    #     #     ...
    #     # }
    #
    #     for i in range (0, len(sequence)):
    #     # Loop through each combination of evidence states
    #         e = {}
    #
    #         # dictionary
    #         for var in evidence.keys():
    #             e[var] = sequenceDict[var][i]
    #
    #         queriedMarginalPosteriors = self.inferWithJunctionTree(e)
    #
    #         allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)
    #
    #     # loop through each state
    #     assembledPosteriors = [] # convert to dataframe
    #
    #     # dummy list of queried PD dicts
    #     assembledP = allStatesQueriedMarginalPosteriors[0]
    #
    #     postInferencePDs = {}
    #     assembledPosterior = []
    #
    #     for i, queryVarName in enumerate (query.keys()):
    #         # loop through each queried PD
    #         probs = allStatesQueriedMarginalPosteriors[0][i]['probability']
    #         num_states = len(probs.tolist()) #queried states
    #
    #         for j in range (0, num_states):
    #             # loop through each state in each queried PD
    #             sum = 0
    #             for k in range (0, len(sequence)):
    #                 # sequence (0, 0), (0, 1), (0, 2), ....
    #                 # sequenceDict = {
    #                 #     var: [0, 1, 2, 3, 4, ... , 1],
    #                 #     var: [0, 1, 2, 3, 4, ....... , 1],
    #                 #     ...
    #                 # }
    #
    #                 # to hold list of probabilities to be multiplied by the
    #                 # conditional probability
    #                 ev = []
    #
    #                 for var in evidence.keys():
    #                     index = sequenceDict[var][k] # ix of evidence state
    #                     # calling the inputted probabilities by index
    #                     ev.append(evidence[var][index])
    #
    #                 probs = allStatesQueriedMarginalPosteriors[k][i]['probability']  # noqa
    #                 red = reduce(lambda x, y: x * y, ev))
    #                 sum += (probs.tolist()[j] * red
    #
    #             # data frame
    #             assembledP[i].set_value(j, 'probability', sum)
    #             # list
    #             assembledPosterior.append(sum)
    #
    #         # this is a cheating step to order probabilities by index of df
    #         # ... should be fixed somwehre before. Compare results with pybbn
    #         # and bayesialab
    #         for d in assembledP:
    #             d.sort_index(inplace=True)
    #
    #         assembledPosteriors.append(assembledPosterior)
    #
    #         probs = assembledP[i]['probability']
    #         postInferencePDs[list(assembledP[i])[0]] = probs.tolist()
    #
    #     # for visualising evidence PDs
    #     for evidenceVarName in evidence.keys():
    #         postInferencePDs[evidenceVarName] = evidence[evidenceVarName]
    #
    #     return assembledP, postInferencePDs


    # def crossValidate (self, targetList, numFolds):
    #     # returns a list of error dataframes, one for each target
    #     # perhaps use **kwargs, to ask if data not specified, then use
    #     # self.binnedData
    #
    #     # create empty dataframes to store errors for each target
    #     error_dict = {}
    #
    #     for target in targetList:
    #         df_columns = [
    #             "NRMSE",
    #             "LogLoss",
    #             "Classification Error",
    #             "Distance Error"
    #         ]
    #         df_indices = ['Fold_%s' % (num + 1) for num in range(numFolds)]
    #         error_df = pandas.DataFrame(index=df_indices, columns=df_columns)
    #         error_df = error_df.fillna(0.0)
    #         error_df['Distance Error'] = error_df['Distance Error'].astype(object)  # noqa
    #         error_df['Classification Error'] = error_df['Classification Error'].astype(object)  # noqa
    #
    #         error_dict[target] = error_df
    #
    #     # specify number of k folds
    #     kf = KFold(n_splits=numFolds)
    #     kf.get_n_splits((self.BNdata.dataArray))
    #
    #     fold_counter = 0
    #     listRMSE = 0.0
    #     listLogLoss = 0.0
    #
    #     for training_index, testing_index in kf.split(self.BNdata.data):
    #         # loop through all data and split into training and testing for
    #         # each fold
    #         print(f"------------ FOLD NUMBER {fold_counter + 1} --------")
    #
    #         trainingData = kfoldToDF(training_index,self.BNdata.data)
    #         testingData = kfoldToDF(testing_index, self.BNdata.data)
    #
    #         # bin test/train data
    #         binRanges = self.BinRanges
    #         binnedTrainingDict, binnedTrainingData, binCountsTr = discretize(trainingData, binRanges, False)  # noqa
    #         binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData, binRanges, False)  # noqa
    #         binnedTestingData=binnedTestingData.astype(int)
    #
    #         # estimate BN parameters
    #         baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)  # noqa
    #
    #         queries ={}
    #         marginalTargetPosteriorsDict = {}
    #         for target in targetList:
    #             # assign bin to zero to query distribution
    #             # (libpgm convention)
    #             queries[target] = 0
    #
    #             # create empty list for each target to populate with
    #             # predicted target posterior distributions
    #             marginalTargetPosteriorsDict[target] = []
    #
    #         # In this loop we predict the posterior distributions for each
    #         # queried target
    #         # TODO #39: Adapt loops for storing predicted posteriors
    #
    #         for i in range (0,binnedTestingData.shape[0]):
    #             row = binnedTestingDict[i]
    #             evidence = without_keys(row, queries.keys())
    #             fn = TableCPDFactorization(baynet)
    #             result = condprobve2(fn, queries, evidence)
    #
    #             if len(queries) > 1:
    #                 # if more than 1 target was specified
    #                 posteriors = printdist(result, baynet)
    #                 for target in targetList:
    #                     probs = posteriors.groupby(target)['probability']
    #                     marginalPosterior = probs.sum()
    #                     # the line below might need [probability]
    #                     marginalTargetPosteriorsDict[target].append(
    #                         marginalPosterior
    #                     )
    #             else:
    #                 # if only 1 target was specified
    #                 posterior = printdist(result, baynet)
    #                 # to make sure probabilities are listed in order of bins,
    #                 # sorted by first queried variable
    #                 posterior.sort_values([targetList[0]],inplace=True)
    #                 marginalTargetPosteriorsDict[target].append(
    #                     posterior['probability']
    #                 )
    #
    #         # generate accuracy measures at one go
    #         # for each target
    #         for key in error_dict.keys():
    #             (
    #               rmse,
    #               loglossfunction,
    #               norm_distance_errors,
    #               correct_bin_probabilities,
    #             ) = generateErrors(
    #                 marginalTargetPosteriorsDict[key],
    #                 testingData,
    #                 binnedTestingData,
    #                 binRanges,
    #                 key
    #             )
    #
    #             # add generated measures to error_df (error dataframe)
    #             error_dict[key]['NRMSE'][fold_counter] = rmse
    #             error_dict[key]['LogLoss'][fold_counter] = loglossfunction
    #             error_dict[key]['Distance Error'][fold_counter] = norm_distance_errors  # noqa
    #             error_dict[key]['Classification Error'][fold_counter] = correct_bin_probabilities  # noqa
    #
    #         fold_counter +=1
    #
    #     return error_dict