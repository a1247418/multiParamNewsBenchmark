import os

# Simulation
seed = 0  # Random seed
k = 30  # Treatment assignment parameter
C = 50  # Outcome generation parameter
# Treatment strength generation parameters
str_const = 2
str_mean = 0.3
str_std = 0.05
nr_documents = 5000  # Nr topics per sample
sets = ['train', 'test']  # Sets to generate
nr_simulations = {
    sets[0]: 1,
    sets[1]: 5
}
treatment_types = [0, 1]  # List of treatments to simulate. binary = 0, parametric = 1
nr_cf_samples = 5  # Number of additional counterfactual samples for parametric treatments

# Preprocessing
do_lda = True  # Set to false if there already exists a LDA file
nr_topics = 50  # Dimensionality of z space
nr_top_words_per_topic = 100  # The number of top words to keep in the dictionary for each topic
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

#Saving
save_as_numpy = True
save_as_bin = True # Save as binary file that can be read in e.g. with R
save_as_tfrecord = True