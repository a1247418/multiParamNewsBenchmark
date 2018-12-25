import os

# Simulation
seed = 0  # Random seed
k = 15 # Treatment assignment bias parameter (0 = no bias)
C = 50  # Outcome generation parameter
# Treatment strength/dose generation parameters
str_mean = 0.1
str_std = 0.05
out_std = 5
n_documents = 7000  # Nr topics per sample (this is the number of simulated datapoints)
sets = ['train', 'test']  # Dataset names to generate
n_simulations = {  # Number of datasets to generate for each dataset
    sets[0]: 1,
    sets[1]: 1
}
treatment_types = [0] + [1] * 3  # Treatment options to simulate. binary = 0, parametric = 1
n_cf_samples = 10  # Number of additional counterfactual samples for parametric treatments

# Preprocessing
do_lda = True  # Set to false if there already exists a LDA file you want to reuse
n_topics = 50  # Dimensionality of topic space space
n_top_words_per_topic = 100  # The number of top words to keep in the dictionary for each topic
normalize_outputs = False  # Normalize x and z

# Paths
base_path = os.getcwd()
in_path = base_path+os.sep+"data"+os.sep
out_path = base_path+os.sep+"output"+os.sep

corpus_name = "nytimes"

serialized_corpus_file = in_path+"corpus."+corpus_name+".obj"
corpus_file = in_path+"docword."+corpus_name+".txt"
vocab_file = in_path+"vocab."+corpus_name+".txt"
lda_file = in_path+"lda."+corpus_name+".obj"
simulation_file = in_path+"simulated_news.csv"

# Saving
save_as_numpy = True
save_as_bin = True  # Save as binary file
