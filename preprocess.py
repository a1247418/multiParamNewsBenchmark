import random
import pickle
import logging
import gensim

import config

import numpy as np


def get_average_doc(serialized_corpus, lda):
    """
    Calculates the mean document vector in topic space
    :param serialized_corpus: corpus serialized for random access
    :param lda: trained lda model
    :return: avg document dense vector in topic space
    """
    nr_docs = serialized_corpus.num_docs
    nr_topics = lda.num_topics

    avg_z = [0 for _ in range(nr_topics)]
    for doc in corpus:
        z_sparse = lda[doc]
        for entry in z_sparse:
            avg_z[entry[0]] += entry[1]

    avg_z = [i/nr_topics for i in avg_z]

    return avg_z


def get_reduced_word_representation(x, words):
    """
    Takes a sparse word vector and a set of words to keep,
    and returns the reduced vector with only certain dimensions kept.
    :param x: Sparse word vector
    :param words: Dimensions to keep
    :return: Sparse word vector with reduced dimensionality
    """
    w2w_red = {}
    for w, word in enumerate(words):
        w2w_red[word] = w

    x_red = []
    for i in x:
        if i[0] in w2w_red.keys():
            x_red.append(w2w_red[i[0]], i[1])

    return x_red


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    np.random.seed(1)

    # load corpus iterator
    corpus = gensim.corpora.UciCorpus(config.corpus_file, config.vocab_file)
    id2word = corpus.create_dictionary()

    do_lda = False
    if do_lda:
        # extract 50 LDA topics
        lda = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=config.nr_topics)

        pickle.dump(lda, open(config.in_path+"lda.obj", "wb"))
    else:
        lda = pickle.load(open(config.in_path+"lda.obj", "rb"))

    # Select the X highest ranking words in each topic, merging them to a set
    words = set()
    for topic_id in range(config.nr_topics):
        words_for_topic = lda.get_topic_terms(topicid=topic_id, topn=config.nr_top_words_per_topic)
        words |= set(words_for_topic)
    words = sorted(list(words))

    gensim.corpora.UciCorpus.serialize(config.serialized_corpus_file, corpus)

    # Calculate mean centroid centroid
    z0 = get_average_doc(corpus, lda)

    corpus_x = []  # Documents in word space, with reduced dimensionality
    corpus_z = []  # Documents in topic space
    count = 0
    for doc in corpus:
        count += 1
        if count % 1000 == 0:
            print("Processing document %d" % count)
        x = get_reduced_word_representation(doc, words)
        corpus_x.append(x)
        z = lda[doc]
        corpus_z.append(z)

    # Save prepared corpus
    to_save = {
        'x' : corpus_x,
        'z' : corpus_z,
        'z0' : z0,
        'dim_x' : len(words),
        'dim_z' : config.nr_topics
    }
    pickle.dump(to_save, open(config.lda_file, "wb"))
