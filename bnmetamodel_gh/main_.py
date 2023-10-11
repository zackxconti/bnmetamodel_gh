from bnmetamodel_gh import BN_Metamodel_easy
import ast
import math

# Specify filepath to csv file
#csvfilepath = '/Users/zack_sutd/Dropbox/SUTD/PhD/Thesis/Phase 2/Simple_truss/Truss Designs/Symmetric Cantilever Beam Truss/FEA Results/Span_Depth/_1000_symmetric_cantilever_trussbeam_geominputs.csv'
#csvfilepath = '/Users/zack_sutd/Dropbox/SUTD/PhD/Thesis/Phase 1/FEA/Data/Sobol/CSV/09.12.16/Sobol_CSV_data_4000.csv'
csvfilepath = '/Users/zxuerebconti/Dropbox/Independent Academic Work/Routledge article/Case Studies/Simple Beam/beam_demo_CSVdata_2targets.csv'
csvfilepath = '/Users/zxuerebconti/Downloads/nervi_fea_meamodel_CSVdata.csv'


# Instantiate a bnmetamodel wrapper
# b = BN_Metamodel_easy.BN_Metamodel_easy(csvfilepath,['Maximum Deflection', 'Weight'])
b = BN_Metamodel_easy.BN_Metamodel_easy(csvfilepath, ['deflection'])

# Generate a bnmetamodel as a function of the wrapper
bn = b.generate()

# evidence= {'Height':0, 'Amplitude':0, 'Beam_tip_depth':0, 'Beam_start_depth':0, 'bc_X_position':0, 'Span':0}
# evidence = {'Height':[1.0, 0.0, 0.0, 0.0, 0.0], 'Amplitude':[1.0, 0.0, 0.0, 0.0, 0.0], 'Beam_tip_depth':[1.0, 0.0, 0.0, 0.0, 0.0], 'Beam_start_depth':[1.0, 0.0, 0.0, 0.0, 0.0], 'bc_X_position':[1.0, 0.0, 0.0, 0.0, 0.0], 'Span':[1.0, 0.0, 0.0, 0.0, 0.0]}
#evidence = {'deflection':[0.5, 0.5, 0.0, 0.0, 0.0], 'weight':[0.5, 0.5, 0.0, 0.0, 0.0] }
# query = {'deflection': 0 }
# evidence = {'deflection':0 }
# evidence = {'max_def':[1, 0.0, 0.0, 0.0], 'weight':[1, 0.0, 0.0, 0.0]  }
evidence = {'deflection':[1, 0.0, 0.0, 0.0, 0.0]}


#Perform inference to 'update' distributions (using Bayesian inference in the background)
posteriors = bn.inferPD_JT_hard(evidence)
# posteriors = bn.inferPD_JT_soft(evidence)

for item in posteriors: print (item,'  ', posteriors[item])

#Visualise posterior distributions
bn.plotPDs(xlabel='Ranges ', ylabel='Probability',maintitle='Posterior Distributions',displayplt=True, posteriorPD=posteriors,evidence=list(evidence.keys()))

#TODO: order inputs and outputs plots, automatically select grid dims