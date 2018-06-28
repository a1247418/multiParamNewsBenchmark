import os

# Simulation
default_k = 10
default_C = 50
default_nr_simulations = 1

# Preprocessing
nr_topics = 50
nr_documents = 100 #  5000
nr_top_words_per_topic = 100

# Paths
base_path = os.getcwd()
in_path = base_path+os.sep+"data"+os.sep
out_path = base_path+os.sep+"output"+os.sep

corpus_name = "kos" #  "nytimes"

serialized_corpus_file = in_path+"corpus."+corpus_name+".obj"
corpus_file = in_path+"docword."+corpus_name+".txt"
vocab_file = in_path+"vocab."+corpus_name+".txt"
lda_file = in_path+"lda."+corpus_name+".obj"
simulation_file = in_path+"simulated_news.csv"
