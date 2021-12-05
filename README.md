# Individual Advantage and Group Identities in Agent-Based Opinion Formation

## About the project
In  this  project,  we  aim  to  gain  a  better  understanding of opinion  spreading  in  medical communities  and  its  dependency  on  relations  between distinct  agents  in  respective  city clusters.  We model physicians as agents and based on the adoption dates of a new drug and adjacency matrices obtained from data concerning relationships between physicians inseveral cities train a machine learning model consisting of neural network layers with an algorithm that is physically inspired by a process of diffusion.  

The goal of it is to fit four parameters we have identified to be influential in opinion spreading, namely three factors of relationship strength between physicians (friendship, advise and discussion relations) and profit constant obtained from adopting an innovative medication.  Upon obtaining fitted parameters on different clusters of physicians, we discuss the applicability of the results inprediction of opinion percentages in other clusters over time concerning adoption of new medication by simulating the algorithm  with  the  obtained  parameters on  other clusters and analyzing the results.


## Dependencies
The code has been tested in python 3.8 with the following dependencies:

```
tqdm
torch/1.7.0
graph-tool/2.4.3
matplotlib/3.4.3
numpy/1.21.2
pandas/1.3.4
seaborn/0.11.0
```

For graph-tool, please follow the official installation instructions on https://graph-tool.skewed.de/

## Run Random Experiments
 We  provide  the  opportunity  to  experiment with different initialization schemes in our code (see ```experiment.py```) and to visualise the simulation 
 dynamically using the graph-tool module (see ```visualization.py```).  
 
 To test the simulation with custom parameters, it is possible to modify the default settings in the two scripts mentioned above. Afterwards, one can 
 adjust the global configurations and simulation settings in ```main.py``` and simply run the main script after having installed the dependencies.
 
 The functions used for initialization are
 - experiment.py
   - initialise_clusters
   - initialise_driving_forces
   - initialise_agents
   - initialise_opinions
   - change_probs
 
 - visualization.py
   - run_simulation
 
 ## Data-Driven Experiments
 The scripts for the data-driven experiments described in the report are:
 ```
 experiment_data.py
 model_data.py
 dataio.py
 ```
 
 These are refactored from ```process.ipynb```
