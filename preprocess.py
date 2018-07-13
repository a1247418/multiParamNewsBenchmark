import random
import pickle
import logging
import gensim

import config

import numpy as np


def get_average_doc(corpus, dim):
    """
    Calculates the mean document vector in the corpus of dimensionality dim
    :param corpus: list of sparse documents
    :return: avg document dense vector
    """
    nr_docs = len(corpus)
    avg_z = [0 for _ in range(dim)]
    for doc in corpus:
        for topic in doc:
            avg_z[topic[0]] += topic[1]
    avg_z = [i/nr_docs for i in avg_z]

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
            x_red.append((w2w_red[i[0]], i[1]))

    return x_red


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    np.random.seed(1)

    # load corpus iterator
    corpus = gensim.corpora.UciCorpus(config.corpus_file, config.vocab_file)
    id2word = corpus.create_dictionary()

    nr_words_in_corpus = len(corpus)

    do_lda = config.do_lda
    if not do_lda:
        print("Loading LDA from file.")
        try:
            lda = pickle.load(open(config.in_path + "lda.obj", "rb"))
        except:
            print("Loading unsuccessful.")
            do_lda = True
    if do_lda:
        print("Calculating LDA.")
        # extract 50 LDA topics
        lda = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=config.nr_topics,
            alpha='auto',
            eval_every=10)

        pickle.dump(lda, open(config.in_path+"lda.obj", "wb"))

    # Select the X highest ranking words in each topic, merging them to a set
    words = set()
    for topic_id in range(config.nr_topics):
        words_for_topic = lda.get_topic_terms(topicid=topic_id, topn=config.nr_top_words_per_topic)
        words_for_topic = [term for term, val in words_for_topic]
        words |= set(words_for_topic)
    words = sorted(list(words))

    print("Selected %d words for document representation." %len(words))

    gensim.corpora.UciCorpus.serialize(config.serialized_corpus_file, corpus)

    corpus_x = []  # Documents in word space, with reduced dimensionality
    corpus_z = []  # Documents in topic space
    max_x = 0 # Normalization factor
    max_z = 0 # Normalization factor
    count = 0
    for doc in corpus:
        count += 1
        if count % 1000 == 0:
            print("Processing document %d" % count)

        x = get_reduced_word_representation(doc, words)
        if x:
            max_x = max(max_x, max([val for dim, val in x]))
            corpus_x.append(x)

            z = lda[doc]
            max_z = max(max_z, max([val for dim, val in z]))
            corpus_z.append(z)

    # Calculate mean centroid centroid
    z0 = get_average_doc(corpus_z, config.nr_topics)
    x0 = get_average_doc(corpus_x, len(words))

    # Normalize
    if config.normalize_outputs:
        for i in range(nr_words_in_corpus):
            for j in range(len(corpus_x[i])):
                corpus_x[i][j] = (corpus_x[i][j][0], corpus_x[i][j][1]/max_x)
            for j in range(len(corpus_z[i])):
                corpus_z[i][j] = (corpus_z[i][j][0], corpus_z[i][j][1]/max_z)
        z0 /= max_z

    # Save prepared corpus
    to_save = {
        'x': corpus_x,
        'z': corpus_z,
        'z0': z0,
        'x0': x0,
        'dim_x': len(words),
        'dim_z': config.nr_topics
    }
    pickle.dump(to_save, open(config.lda_file, "wb"))
