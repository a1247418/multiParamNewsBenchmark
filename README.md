# multiParamNewsBenchmark
## Introduction
Semi-synthetic dataset based on the NY Times corpus available at UCI (Newman, 2008) and the simulation described in 
"Learning Representations for Counterfactual Inference"[1].

The dataset realizes an arbitraty number of binary or parametric 
treatment options.

## BEFORE RUNNING
Before simulating the data you need to download the [NY Times corpus] (https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/) 
and placing it in the 'data' folder.

## How to run
After downloading the corpus, run preprocess.py to generate an LDA representation of the data.

Then run simulate.py to generate the simulated dataset.

The jupyter notebook analysis.ipynb is provided as a starting-point to explore the dataset.

The config.py file contains many options that can be set. Among others, how many treatments should be generated.

## Result file
The simulation outcome is currently saved as dictionary in a numpy file containing the
following keys:
	'centroids_z': 	A list of the treatment centroids in topic space.
	'centroids_x': 	Same in word space.
	'z': 			All sampled documents in topic space.
	'x': 			Same in word space.
	't': 			The treatment given for each sample. 0 represents the control group.
	'mu': 			The deterministic (true) outcome based on z and t for each sample and treatment.
	'y': 			The measured (noisy) outcome based on mu + noise for each sample and treatment.
	's': 			The treatment 'strength' for each sample and treatment. Is 1 for binary treatment and in [0,1] for parametric treatments.
	'param': 		A boolean list signifying whether a treatment is binary (0) or parametric (1).

## Generation process
As in [1] the NYT corpus is used as a basis. Each unit x represents a document as a vector of word counts.
On this data, LDA is run to generate for each document a vector z in topic space. The dimensionality of Z is a parameter.
The vectors in X are reduced to only keep the dimensions corresponding to the union of most probable words in the topics identified by LDA.
For each treatment, a centroid is defined in Z space. For t=0 it is the median of all documents. For all others it is a randomly chosen document.
The probability of treatment assignment for each unit and treatment depends on parameter k and the proximity of the unit to the treatment centroids in Z space.
The treatment strength is determined as function of treatment, and proximity to treatment centroid.
The treatment outcome is based on treatment, treatment strength, and proximity to treatment centroid.
Both outcome y and treatment strength s are augmented with reandom noise.
	
## References
[1] Fredrik D. Johansson, Uri Shalit & David Sontag. Learning Representations for Counterfactual Inference. 33rd International Conference on Machine Learning (ICML), June 2016.
