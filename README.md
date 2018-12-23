# multiParamNewsBenchmark
## Introduction
This repository hosts a semi-synthetic benchmark based on the UCI NY-Times corpus (Newman, 2008) and the simulation described in 
"Learning Representations for Counterfactual Inference"[1].

The benchmark realizes an arbitrary number of binary or parametric (continuous)
treatment options. Its covariates are based on real-world data, but the 
treatment assignments and outcomes are simulated.

## BEFORE RUNNING
Before simulating the data, you need to **download the NY Times corpus** ( https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/ ) 
and place it in the 'data' folder. Download both the docword and vocab file.

## How to run
After downloading the corpus, **run preprocess.py** to generate an LDA representation of the data.<br />
Then **run simulate.py** to generate the simulated dataset.<br />
The jupyter notebook analysis.ipynb is provided as a starting point to explore the dataset.<br />
The config.py file contains options for the simulation process. Among others, how many treatments should be simulated and how many datasets should be created.<br />

## Result files
For each simulated dataset a separate file is created and the simulation outcome is saved as dictionary in a numpy file containing the
following keys:<br />
	'centroids_z': 		A list of the treatment centroids in topic space.<br />
	'centroids_x': 		Same in word space.<br />
	'z': 			All sampled documents in topic space. [n_samples, n_topics]<br />
	'x': 			Same in word space. [n_samples, n_words]<br />
	't': 			The treatment given for each sample. 0 represents the control group. [n_samples]<br />
	'mu': 			The deterministic (true) outcome based on z and t for each sample and treatment. [n_samples, n_treatments]<br />
	'y': 			The measured (noisy) outcome based on mu + noise for each sample and treatment. [n_samples, n_treatments]<br />
	's': 			The treatment 'strength' for each sample and treatment. Is 1 for binary treatment and in [0,1] for parametric treatments. [n_samples, n_treatments]<br />
	'treatment_types': 	A boolean list signifying whether a treatment is binary (0) or parametric (1). [n_treatments]<br />
To be able to calculate a more accurate counterfactual error for the parametric treatment options, additional counterfactual samles are provided in:<br />
	'mu_pcf': 		[n_samples, n_parametric_treatments, n_additional_samples]<br />
	'y_pcf': 		[n_samples, n_parametric_treatments, n_additional_samples]<br />
	's_pcf': 		Uniform random numbers in [0,1]. [n_samples, n_parametric_treatments, n_additional_samples]<br /><br />
In general, you only need x,t,y,s,y_pcf, and s_pcf to train and evaluate your model.<br />

Note: Currently the data can also be saved in binary format. To this end, multi-dimensional matrices are flattened to 2D and the format changes - using this option
is so far not recommended.<br />

## Generation process
As in [1] the NYT corpus is used as a basis for the simulation. Each unit x represents a document as a vector of word counts.<br />
On this data, LDA is run to generate for each document a vector z in topic space Z. The dimensionality of Z is a parameter.<br />
The vectors in X are reduced to only keep the dimensions corresponding to the union of most probable words in the topics identified by LDA.<br />
For each treatment, a centroid is defined in Z space. For t=0 it is the mean vector of all documents. For all others it is a randomly chosen document.<br />
The probability of treatment assignment for each unit and treatment depends on parameter k and the proximity of the unit to the treatment centroids in Z space.<br />
The treatment strength is determined as function of treatment, and proximity to treatment centroid.<br />
The treatment outcome is based on treatment, treatment strength, and proximity to treatment centroid.<br />
Both outcome y and treatment strength s are augmented with random noise.<br />
For each simulation run, centroids are rerandomized and treatment assignments and outcomes are calculared.
	
## References
[1] Fredrik D. Johansson, Uri Shalit & David Sontag. Learning Representations for Counterfactual Inference. 33rd International Conference on Machine Learning (ICML), June 2016.
