from BayesianNetwork import *
from BNdata import *
from Helper_functions import loadDataFromCSV

class BN_Metamodel_easy:


    def __init__(self, csvdata, targets,**kwargs ): # data can either be specified by file path or by list

        #order:
        # 1) loads csv file
        # 2) builds BN skeleton (topology)
        # 3) specifies bin types /  bin nums
        # 4) perpares data usin BNdata
        # 5) builds BN (data, skel)

        self.targets = targets
        self.variables = loadDataFromCSV(csvdata,True)[0] # load data from
        self.binTypeDict = {} # declare as empty attribute
        self.numBinsDict = {} # declare as empty attribute

        # extract skeleton from csv
        BNskel = BNskelFromCSV(csvdata, targets)

        # if bool(BN_Metamodel_easy.numBinsDict) == False { } # if numBinsDict is empty

        #binTypeDict = {}
        #numBinsDict = {}

        if 'numBinsDict' in kwargs:
            self.numBinsDict = kwargs['numBinsDict']

       # if 'binTypeDict' in kwargs:
        #"""
        for var in self.variables:
            if var in targets:
                self.binTypeDict [var]= 'e' # default: all distributions are discretized by equal spacing
                self.numBinsDict [var] = 7 # default: all distributions have 6 bins by default
            else:
                self.binTypeDict [var]= 'e' # default: all distributions are discretized by equal spacing
                self.numBinsDict [var] = 7 # default: all distributions have 6 bins by default
        #"""
        #data = BNdata(csvdata, self.binTypeDict, self.numBinsDict)
        data = BNdata(csvdata=csvdata, targetlist=self.targets, binTypeDict=self.binTypeDict, numBinsDict=self.numBinsDict)
        #data = BNdata(csvdata, self.targets)

        #self.binTypeDict = data.binTypeDict
        #self.numBinsDict = data.numBinsDict

        self.learnedBaynet = BayesianNetwork(BNdata = data, netStructure=BNskel)



    def json (self):

        return self.learnedBaynet.json_data


    def generate (self):

        return self.learnedBaynet


    def changeNumBinsDict (dict):

        BN_Metamodel_easy.numBinsDict = dict


    def inferPD_JT_soft(self, query, softevidence):

        posteriors = self.learnedBaynet.inferPD_JT_soft(softevidence)

        self.learnedBaynet.plotPDs(xlabel='Ranges ', ylabel='Probability',maintitle='Posterior Distributions',displayplt=True, posteriorPD=posteriors, evidence=softevidence.keys())

        return posteriors


    def inferPD_JT_hard(self, query, hardevidence):

        posteriors = self.learnedBaynet.inferPD_JT_hard(hardevidence)

        self.learnedBaynet.plotPDs(xlabel='Ranges ', ylabel='Probability',maintitle='Posterior Distributions',displayplt=True, posteriorPD=posteriors, evidence=hardevidence.keys())

        return posteriors


    def inferPD_VE_hard(self, query, evidence):

        a, posteriors = self.learnedBaynet.inferPD(query, evidence)

        self.learnedBaynet.plotPDs(xlabel='Ranges ', ylabel='Probability',maintitle='Posterior Distributions', displayplt=True, posteriorPD=posteriors,evidence=evidence.keys())

        return posteriors








