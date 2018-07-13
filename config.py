import os

# Simulation
default_k = 10
default_C = 50
default_nr_simulations = 1
default_parametric_treatment = True  # Whether to simulate only a binary treatment, or a parametric one as well.
generate_testset = True

# Preprocessing
do_lda = False  # Set to false if there already exists a LDA file
nr_topics = 50
nr_documents = 5000
nr_top_words_per_topic = 100
normalize_outputs = False #  Normalize x and z

# Paths
base_path = os.getcwd()
in_path = base_path+os.sep+"data"+os.sep
out_path = base_path+os.sep+"output"+os.sep

corpus_name = "nytimes"  #"kos" #

serialized_corpus_file = in_path+"corpus."+corpus_name+".obj"
corpus_file = in_path+"docword."+corpus_name+".txt"
vocab_file = in_path+"vocab."+corpus_name+".txt"
lda_file = in_path+"lda."+corpus_name+".obj"
simulation_file = in_path+"simulated_news.csv"
