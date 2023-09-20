# from six import string_types  # INFO: not used
# import json # INFO: not used
# from itertools import product  # INFO: not used
# from networkx.algorithms.dag import topological_sort  # INFO: not used

# from pybbn.graph.dag import Bbn  # INFO: not used
# from pybbn.graph.edge import Edge, EdgeType  # INFO: not used
# from pybbn.graph.node import BbnNode  # INFO: not used
# from pybbn.graph.variable import Variable  # INFO: not used

# import itertools  # INFO: not used
# import pandas as pd  # INFO: not used


# class GraphSkeleton(Dictionary):
#     """
#     This class represents a graph skeleton, meaning a vertex set and a
#     directed edge set. It contains the attributes *V* and *E*, and the
#     methods *load*, *getparents*, *getchildren*, and *toporder*.
#     """
#
#     def __init__(self):
#         self.V = None
#         """A list of names of vertices."""
#         self.E = None
#         """A list of [origin, destination] pairs of vertices that constitute edges.""" # noqa
#         self.alldata = None
#         """(Inherited from dictionary) A variable that stores a key-indexable dictionary once it is loaded from a file."""  # noqa
#
#     def load(self, path):
#         """
#         Load the graph skeleton from a text file located at *path*.
#
#         Text file must be a plaintext .txt file with a JSON-style
#         representation of a dict.  Dict must contain the top-level keys "V"
#         and "E" with the following formats::
#
#             {
#                 'V': ['<vertex_name_1>', ... , '<vertex_name_n'],
#                 'E': [['vertex_of_origin', 'vertex_of_destination'], ... ]
#             }
#
#         Arguments
#         ---------
#         path
#             The path to the file containing input data (e.g.,
#             "mydictionary.txt").
#
#         Attributes modified
#         -------------------
#         *V*
#             The set of vertices.
#         *E*
#              The set of edges.
#         """
#         self.dictload(path)
#         self.V = self.alldata["V"]
#         self.E = self.alldata["E"]
#
#         # free unused memory
#         del self.alldata
#
#     def getparents(self, vertex):
#         """
#         Return the parents of *vertex* in the graph skeleton.
#
#         Arguments
#         ---------
#         vertex
#             The name of the vertex whose parents the function finds.
#
#         Returns
#         -------
#         list
#             A list containing the names of the parents of the vertex.
#         """
#         if not vertex in self.V:
#             raise SyntaxError(
#                 "The graph skeleton does not contain this vertex."
#             )
#
#         parents = []
#         for pair in self.E:
#             if (pair[1] == vertex):
#                 parents.append(pair[0])
#         return parents
#
#     def getchildren(self, vertex):
#         """
#         Return the children of *vertex* in the graph skeleton.
#
#         Arguments
#         ---------
#         vertex
#             The name of the vertex whose children the function finds.
#
#         Returns
#         -------
#         list
#             A list containing the names of the children of the vertex.
#         """
#         if not vertex in self.V:
#             raise SyntaxError(
#                 "The graph skeleton does not contain this vertex."
#             )
#
#         children = []
#         for pair in self.E:
#             if (pair[0] == vertex):
#                 children.append(pair[1])
#         return children
#
#     def toporder(self):
#         """
#         Modify the vertices of the graph skeleton such that they are in
#         topological order.
#
#         A topological order is an order of vertices such that if there is an
#         edge from *u* to *v*, *u* appears before *v* in the ordering. It
#         works only for directed ayclic graphs.
#
#         Attributes modified
#         -------------------
#         *V*
#             The names of the vertices are put in topological order.
#
#         The function also checks for cycles in the graph, and returns an
#         error if one is found.
#         """
#         Ecopy = [x[:] for x in self.E]
#         roots = []
#         toporder = []
#
#         for vertex in self.V:
#             # find roots
#             if (self.getparents(vertex) == []):
#                 roots.append(vertex)
#
#         while roots != []:
#             n = roots.pop()
#             toporder.append(n)
#             for edge in reversed(Ecopy):
#                 if edge[0] == n:
#                     m = edge[1]
#                     Ecopy.remove(edge)
#                     yesparent = False
#                     for e in Ecopy:
#                         if e[1] == m:
#                             yesparent = True
#                             break
#                     if yesparent == False:
#                         roots.append(m)
#         assert (not Ecopy), ("Graph contains a cycle", Ecopy)
#         self.V = toporder


# def discrete_mle_estimateparams2(graphskeleton, data):
#     """
#     Estimate parameters for a discrete Bayesian network with a structure
#     given by *graphskeleton* in order to maximize the probability of data
#     given by *data*.
#
#     Parameters
#     ----------
#     graphskeleton
#         An instance of the :doc:`GraphSkeleton <graphskeleton>` class
#         containing vertex and edge data.
#     data
#         A list of dicts containing samples from the network in
#         {vertex: value} format. Example::
#
#                 [
#                     {
#                         'Grade': 'B',
#                         'SAT': 'lowscore',
#                         ...
#                     },
#                     ...
#                 ]
#
#     This function normalizes the distribution of a node's outcomes for each
#     combination of its parents' outcomes. In doing so it creates an estimated
#     tabular conditional probability distribution for each node. It then
#     instantiates a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>`
#     instance based on the *graphskeleton*, and modifies that instance's
#     *Vdata* attribute to reflect the estimated CPDs. It then returns the
#     instance.
#
#     The Vdata attribute instantiated is in the format seen in
#     :doc:`unittestdict`, as described in :doc:`discretebayesiannetwork`.
#
#     Example
#     -------
#     This would learn parameters from a set of 200 discrete samples::
#
#         import json
#
#         from libpgm.nodedata import NodeData
#         from libpgm.graphskeleton import GraphSkeleton
#         from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
#         from libpgm.pgmlearner import PGMLearner
#
#         # generate some data to use
#         nd = NodeData()
#         nd.load("../tests/unittestdict.txt")    # an input file
#         skel = GraphSkeleton()
#         skel.load("../tests/unittestdict.txt")
#         skel.toporder()
#         bn = DiscreteBayesianNetwork(skel, nd)
#         data = bn.randomsample(200)
#
#         # instantiate my learner
#         learner = PGMLearner()
#
#         # estimate parameters from data and skeleton
#         result = learner.discrete_mle_estimateparams(skel, data)
#
#         # output
#         print json.dumps(result.Vdata, indent=2)
#     """
#     if not isinstance(graphskeleton, GraphSkeleton):
#         raise SyntaxError("First arg must be a loaded GraphSkeleton class.")
#
#     if (
#         not isinstance(data, list) or
#         (data and not isinstance(data[0], dict))
#     ):
#         raise SyntaxError("Second arg must be a list of dicts.")
#
#     # instantiate Bayesian network, and add parent and children data
#     bn = DiscreteBayesianNetwork()
#     graphskeleton.toporder()
#     bn.V = graphskeleton.V
#     bn.E = graphskeleton.E
#     bn.Vdata = dict()
#     for vertex in bn.V:
#         bn.Vdata[vertex] = dict()
#         bn.Vdata[vertex]["children"] = graphskeleton.getchildren(vertex)
#         bn.Vdata[vertex]["parents"] = graphskeleton.getparents(vertex)
#
#         # make placeholders for vals, cprob, and numoutcomes
#         bn.Vdata[vertex]["vals"] = []
#         if (bn.Vdata[vertex]["parents"] == []):
#             bn.Vdata[vertex]["cprob"] = []
#         else:
#             bn.Vdata[vertex]["cprob"] = dict()
#
#         bn.Vdata[vertex]["numoutcomes"] = 0
#
#     # STEP 1
#
#     # determine which outcomes are possible for each node
#     for sample in data:
#         for vertex in bn.V:
#             if (sample[vertex] not in bn.Vdata[vertex]["vals"]):
#                 bn.Vdata[vertex]["vals"].append(sample[vertex])
#                 bn.Vdata[vertex]["numoutcomes"] += 1
#
#     # lay out probability tables, and put a [num, denom] entry in all spots:
#
#     # define helper function to recursively set up cprob table
#     def addlevel(vertex, _dict, key, depth, totaldepth):
#         if depth == totaldepth:
#             _dict[str(key)] = []
#             for _ in range(bn.Vdata[vertex]["numoutcomes"]):
#                 _dict[str(key)].append([0, 0])
#             return
#         else:
#             for val in bn.Vdata[bn.Vdata[vertex]["parents"][depth]]["vals"]:
#                 ckey = key[:]
#                 ckey.append(str(val))
#                 addlevel(vertex, _dict, ckey, depth + 1, totaldepth)
#
#     # STEP 2
#     # put [0, 0] at each entry of cprob table
#     for vertex in bn.V:
#         if (bn.Vdata[vertex]["parents"]):
#             root = bn.Vdata[vertex]["cprob"]
#             numparents = len(bn.Vdata[vertex]["parents"])
#             addlevel(vertex, root, [], 0, numparents)
#         else:
#             for _ in range(bn.Vdata[vertex]["numoutcomes"]):
#                 bn.Vdata[vertex]["cprob"].append([0, 0])
#
#     # STEP 3
#     # fill out entries with samples:
#     for sample in data:
#         for vertex in bn.V:
#             # compute index of result
#             rindex = bn.Vdata[vertex]["vals"].index(sample[vertex])
#
#             # go to correct place in Vdata
#             if bn.Vdata[vertex]["parents"]:
#                 pvals = [str(sample[t]) for t in bn.Vdata[vertex]["parents"]]
#                 lev = bn.Vdata[vertex]["cprob"][str(pvals)]
#             else:
#                 lev = bn.Vdata[vertex]["cprob"]
#
#             # increase all denominators for the current condition
#             for entry in lev:
#                 entry[1] += 1
#
#             # increase numerator for current outcome
#             lev[rindex][0] += 1
#
#     # STEP 4
#     # LAPLACE SMOOTHING TO AVOID ZERO DIVISION ERROR WHEN WE HAVE EMPTY BINS
#     for vertex in bn.V:
#         numBins = bn.Vdata[vertex]['numoutcomes']
#
#         if not (bn.Vdata[vertex]["parents"]):  # has no parents
#             for counts in bn.Vdata[vertex]['cprob']:
#                 counts[0] += 1  # numerator (count)
#                 counts[1] += numBins  # denomenator (total count)
#         else:
#             countdict = bn.Vdata[vertex]['cprob']
#
#             for key in countdict.keys():
#                 for counts in countdict[key]:
#                     counts[0]+=1
#                     counts[1]+=numBins
#
#             # STEP 5
#             """
#             # OPTIONAL: converts cprob from dict into df, does laplace
#             # smoothing, then (missing) maps back to dict
#             bincounts = pd.DataFrame.from_dict(
#                 bn.Vdata[vertex]['cprob'], orient='index'
#             )
#
#             for columnI in range (0, bincounts.shape[1]):
#                 for rowI in range (0,bincounts.shape[0]):
#                     row_data = bincounts[columnI][rowI]
#                     bincounts[columnI][rowI] = [
#                         row_data[0] + 1, row_data[1] + numBins
#                     ]
#             """
#
#     # STEP 6
#     ########################################################################
#
#     # convert arrays to floats
#     for vertex in bn.V:
#         if not bn.Vdata[vertex]["parents"]:
#             bn.Vdata[vertex]["cprob"] = [
#                 x[0] / float(x[1]) for x in bn.Vdata[vertex]["cprob"]
#             ]
#         else:
#             for key in bn.Vdata[vertex]["cprob"].keys():
#                 try:
#                     bn.Vdata[vertex]["cprob"][key] = [
#                         x[0] / float(x[1])
#                         for x in bn.Vdata[vertex]["cprob"][key]
#                     ]
#
#                 # default to even distribution if no data points
#                 except ZeroDivisionError:
#
#                     bn.Vdata[vertex]["cprob"][key] = [
#                         1 / float(bn.Vdata[vertex]["numoutcomes"])
#                         for x in bn.Vdata[vertex]["cprob"][key]
#                     ]
#
#     # return cprob table with estimated probability distributions
#     return bn


# def condprobve2(self, query, evidence):
#     """
#     Eliminate all variables in *factorlist* except for the ones queried.
#     Adjust all distributions for the evidence given. Return the probability
#     distribution over a set of variables given by the keys of *query* given
#     *evidence*.
#
#     Arguments
#     ---------
#     query
#         A dict containing (key: value) pairs reflecting (variable: value)
#         that represents what outcome to calculate the probability of.
#     evidence
#         A dict containing (key: value) pairs reflecting (variable: value)
#         that represents what is known about the system.
#
#     Attributes modified
#     -------------------
#     *factorlist*
#         Modified to be one factor representing the probability distribution
#         of the query variables given the evidence.
#
#     The function returns *factorlist* after it has been modified as above.
#
#     Example
#     -------
#     This code would return the distribution over a queried node, given
#     evidence::
#
#         import json
#
#         from libpgm.graphskeleton import GraphSkeleton
#         from libpgm.nodedata import NodeData
#         from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
#         from libpgm.tablecpdfactorization import TableCPDFactorization
#
#         # load nodedata and graphskeleton
#         nd = NodeData()
#         skel = GraphSkeleton()
#         nd.load("../tests/unittestdict.txt")
#         skel.load("../tests/unittestdict.txt")
#
#         # toporder graph skeleton
#         skel.toporder()
#
#         # load evidence
#         evidence = dict(Letter='weak')
#         query = dict(Grade='A')
#
#         # load bayesian network
#         bn = DiscreteBayesianNetwork(skel, nd)
#
#         # load factorization
#         fn = TableCPDFactorization(bn)
#
#         # calculate probability distribution
#         result = fn.condprobve(query, evidence)
#
#         # output
#         print json.dumps(result.vals, indent=2)
#         print json.dumps(result.scope, indent=2)
#         print json.dumps(result.card, indent=2)
#         print json.dumps(result.stride, indent=2)
#
#     '''
#     if not isinstance(query, dict) or not isinstance(evidence, dict):
#         raise SyntaxError("First and second args must be dicts.")
#
#     ## need to modify and add 1 to the zeros here but need frequency count
#
#     eliminate = self.bn.V[:]
#     for key in query.keys():
#         eliminate.remove(key)
#
#     for key in evidence.keys():
#         eliminate.remove(key)
#
#     # modify factors to account for E = e
#     for key in evidence.keys():
#         for x in range(len(self.factorlist)):
#             if (self.factorlist[x].scope.count(key) > 0):
#                 self.factorlist[x].reducefactor(key, evidence[key])
#         for x in reversed(range(len(self.factorlist))):
#             if (self.factorlist[x].scope == []):
#                 del (self.factorlist[x])
#
#     # eliminate all necessary variables in the new factor set to produce
#     # result
#     self.sumproductve(eliminate)
#
#     # normalize result
#     summ = 0.0
#     lngth = len(self.factorlist.vals)
#     for x in range(lngth):
#         summ += self.factorlist.vals[x]
#
#     for x in range(lngth):
#         a = float(self.factorlist.vals[x])
#         a = a / summ
#
#     # return table
#     return self.factorlist
#
#
# def inferPosteriorDistribution(queries, evidence, baynet):
#     # TODO #45: Extend inferPosteriorDistribution to handle multiple query nodes
#     fn = TableCPDFactorization(baynet)
#
#     result = condprobve2(fn, queries, evidence)  # written here
#     print 'result.vals ', result.vals
#     probabilities = printdist(result, baynet)
#
#     # make sure probabilities are listed in order of bins
#     probabilities.sort_values(['max_def'], inplace=True)
#
#     return probabilities

# def buildBN(trainingData, binstyleDict, numbinsDict, **kwargs):
#     # need to modify to accept skel or skelfile
#     discretized_training_data, bin_ranges = discretizeTrainingData(
#         trainingData, binstyleDict, numbinsDict, True
#     )
#     print('discret training', discretized_training_data)
#
#     if 'skel'in kwargs:
#         # load file into skeleton
#         if isinstance(kwargs['skel'], basestring):
#             skel = GraphSkeleton()
#             skel.load(kwargs['skel'])
#             skel.toporder()
#         else:
#             skel = kwargs['skel']
#
#     # learn bayesian network
#     learner = PGMLearner()
#
#     # using discrete_mle_estimateparams2 written as function in this file,
#     # not calling from libpgm
#     baynet = discrete_mle_estimateparams2(skel,discretized_training_data)
#
#     return baynet


# def from_data(structure, df):
#     """
#     Creates a BBN.
#
#     Parameters
#     ----------
#     structure : _type_
#         A dictionary where keys are names of children and values are list of
#         parent names.
#     df : pandas.DataFrame
#         A dataframe.
#
#     Returns
#     -------
#     BBN.
#     """
#
#     def get_profile(df):
#         profile = {}
#         for c in df.columns:
#             values = sorted(list(df[c].value_counts().index))
#             profile[c] = values
#         return profile
#
#     def get_n2i(parents):
#         g = nx.DiGraph()
#         for k in parents:
#             g.add_node(k)
#         for ch, pas in parents.items():
#             for pa in pas:
#                 g.add_edge(pa, ch)
#         nodes = list(topological_sort(g))
#         return {n: i for i, n in enumerate(nodes)}
#
#     def get_cpt(name, parents, n2v, df):
#         parents = sorted(parents)
#         n2v = {k: sorted(v) for k, v in n2v.items()}
#
#         n = df.shape[0]
#
#         cpts = []
#         if len(parents) == 0:
#             for v in n2v[name]:
#                 c = df[df[name] == v].shape[0]
#                 p = c / n
#                 cpts.append(p)
#         else:
#             domains = [(n, d) for n, d in n2v.items() if n in parents]
#             domains = sorted(domains, key=lambda tup: tup[0])
#             domain_names = [tup[0] for tup in domains]
#             domain_values = [tup[1] for tup in domains]
#             domains = list(product(*domain_values))
#
#             for values in domains:
#                 probs = []
#                 denom_q = ' and '.join([
#                     f'{n}=="{v}"' for n, v in zip(domain_names, values)
#                 ])
#                 for v in n2v[name]:
#                     numer_q = f'{name}=="{v}" and {denom_q}'
#
#                     numer = df.query(numer_q).shape[0] / n
#                     denom = df.query(denom_q).shape[0] / n
#
#                     if denom == 0:
#                         prob = 1e-5
#                     else:
#                         prob = numer / denom
#                     probs.append(prob)
#                 probs = pd.Series(probs)
#                 probs = probs / probs.sum()
#                 probs = list(probs)
#                 cpts.extend(probs)
#
#         return cpts
#
#     n2v = get_profile(df)
#     n2i = get_n2i(df)
#     n2c = {n: get_cpt(n, structure[n], n2v, df) for n in structure}
#
#     bbn = Bbn()
#
#     nodes = {}
#     for name in n2v:
#         idx = n2i[name]
#         values = n2v[name]
#         cpts = n2c[name]
#
#         v = Variable(idx, name, values)
#         node = BbnNode(v, cpts)
#         nodes[name] = node
#         bbn.add_node(node)
#
#     for ch, parents in structure.items():
#         ch_node = nodes[ch]
#         for pa in parents:
#             pa_node = nodes[pa]
#
#             edge = Edge(pa_node, ch_node, EdgeType.DIRECTED)
#             bbn.add_edge(edge)
#
#     return bbn
