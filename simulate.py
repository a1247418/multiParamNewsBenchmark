import sys
import random
import pickle
import logging

import config

import numpy as np


def simulate_outcomes(C, t, z, z0, z1):
    eta = np.random.normal(0, 1)
    yF = C*(np.dot(z,z0) + t*np.dot(z, z1))+eta

    eta = np.random.normal(0, 1)
    yCF = C*(np.dot(z,z0) + (not t)*np.dot(z, z1))+eta

    return yF, yCF


def calc_treatment_probability(k, z, z0, z1):
    numerator = np.power(np.e, k*np.dot(z,z1))
    denominator = np.power(np.e, k*np.dot(z,z0)) + numerator

    return numerator/denominator


def sparse_to_dense(x, nr_dims):
    """
    :param x: Sparse vector to transform
    :param nr_dims: Number of dimensions of dense vector
    :return: Dense representation of sparse vec
    """
    x_dense = np.zeros(nr_dims)
    for i in x:
        x_dense[i[0]] = i[1]

    return x_dense


if __name__ == '__main__':
    nr_simulations = config.default_nr_simulations
    k = config.default_k
    C = config.default_C

    nr_arguments = len(sys.argv)
    if nr_arguments > 1:
        try:
            nr_simulations = int(sys.argv[1])
        except:
            print("nr_simulations must be integer")

        if nr_arguments > 2:
            try:
                k = int(sys.argv[2])
            except:
                print("k must be integer")

            if nr_arguments > 3:
                try:
                    C = int(sys.argv[3])
                except:
                    print("C must be integer")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    random.seed()

    corpus = pickle.load(open(config.lda_file, 'rb'))
    corpus_x = corpus['x']
    corpus_z = corpus['z']
    z0 = corpus['z0']

    dim_x = corpus['dim_x']
    dim_z = corpus['dim_z']

    nr_docs = len(corpus_x)

    for _ in range(nr_simulations):
        # Sample X documents
        doc_ids = sorted(random.sample(range(nr_docs), config.nr_documents))

        # Sample "treated" centroid
        z1_id = random.randint(0, nr_docs-1)
        z1 = sparse_to_dense(corpus_z[z1_id], dim_z)

        sample_x = []  # Documents in word space, with reduced dimensionality
        sample_z = []  # Documents in topic space
        sample_p = []  # Treatment probability
        sample_t = []  # Treatment assignment
        sample_yF = []  # Factual outcome
        sample_yCF = []  # Counterfactual outcome
        for d in doc_ids:
            x = sparse_to_dense(corpus_x[d], dim_x)
            sample_x.append(x)
            z = sparse_to_dense(corpus_z[d], dim_z)
            sample_z.append(z)
            p = calc_treatment_probability(k, z, z0, z1)
            sample_p.append(p)
            t = random.random() < p
            sample_t.append(t)
            yF, yCF = simulate_outcomes(C, t, z, z0, z1)
            sample_yF.append(yF)
            sample_yCF.append(yCF)

        to_save = {
            'x': sample_x,
            't': sample_t,
            'yf': sample_yF,
            'ycf': sample_yCF
        }

        # Save a simulation run
        np.save("simulation_outcome", to_save)
