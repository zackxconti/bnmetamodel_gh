# bnmetamodel_gh
Repo for bnmetamodel lib version for the Grasshopper plug-in.

bnmetamodel is a python library (in progress), to build Bayesian Network metamodels from csv data. The library makes use of 'libpgm', which is a python library written by [authors] for building bayesian networks and performing inference. The secondary aim for this repository is to serve as a backbone to a potential Grasshopper plugin for generating Bayesian network metamodels. 

This repository hosts three main folders: classes, functions and casestudy_examples. 

(1) classes - three main classes that are needed by  to create a BN 
(2) functions - containts a list of functions that used 'internally'. They perform different tasks such as data structure conversions, plotting, etc. I intend to sort them out everntually. 
(3) casestudy_examples - here I will store different examples of Bayesian network metamodels

# Simple example using the BN_Metamodel_easy wrapper

``` ruby
from BN_Metamodel_easy import *

# STEP 1: Specify csv file
csvfilepath = '/Users/zack_sutd/Dropbox/SUTD/PhD/Thesis/Phase 2/Simple_truss/Truss Designs/Symmetric Cantilever Beam Truss/FEA Results/Span_Depth/_1000_symmetric_cantilever_trussbeam_geominputs.csv'

# STEP 2: Instantiate a BN_Metamodel wrapper
b = BN_Metamodel_easy(csvfilepath, ['max_def'])

# STEP 3: Generate a BN_Metamodel as a function of the wrapper
bn = b.generate()

# STEP 4: Specify evidence and query distribution for each variable (as list of normalised percentages)
evidence = {'span':[0.5, 0.5, 0.0, 0.0, 0.0, 0.0 ], 'depth':[0.0, 0.0, 0.0, 0.0, 0.5, 0.5 ]}
query = {'max_def':0}

# STEP 5: Perform inference to 'update' distributions (using Bayesian inference in the background)
a, posteriors = bn.inferPD_3(query, evidence) #a is a dummy variable

# STEP 6: Visualise posterior distributions
##### TODO: I will internalise the following lines #####
rows = 1
columns = len(query.keys())+len(evidence.keys())
if (len(query.keys())+len(evidence.keys()))>4:
    columns=4
    rows = math.ceil((len(query.keys())+len(evidence.keys()))/4)
    
bn.plotPDs(rows, columns,xlabel='Ranges ', ylabel='Probability',maintitle='Posterior Distributions',displayplt=True, posteriorPD=posteriors, evidence=evidence.keys())

```
![](images/bn_example1.jpg)


Notes:

(1) Multimodal posteriors might be a sign of insufficient samples. 
