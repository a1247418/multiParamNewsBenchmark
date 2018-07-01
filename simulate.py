import sys
import random
import pickle
import logging

import config

import numpy as np

SIM_BIN = "binary"
SIM_MUL = "multi"

def simulate_outcomes(C1, C2, t, z, z0, z1, z2):
    etaF = np.random.normal(0, 1)
    etaCF = np.random.normal(0, 1)

    mu = [0, 0, 0]
    mu[0] = C1*np.dot(z, z0)
    mu[1] = mu[0] + C1*(np.dot(z, z1))
    mu[2] = mu[0] + C1*(C2 * np.dot(z, z2))

    yF = etaF + (mu[1] if t==1 else mu[0])
    yCF = etaCF + (mu[0] if t==0 else mu[1])

    return yF, yCF, mu[0], mu[1]


def calc_treatment_probability(k, z, z0, z1):
    dot_zz0 = np.dot(z, z0)
    dot_zz1 = np.dot(z, z1)

    numerator = np.power(np.e, k*dot_zz1)
    denominator = np.power(np.e, k*dot_zz0) + numerator

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
    simulytion_type = config.default_simulation_type
    nr_simulations = config.default_nr_simulations
    k = config.default_k
    C1 = config.default_C1
    C2 = config.default_C2

    nr_arguments = len(sys.argv)
    if nr_arguments > 1:
        simulytion_type = sys.argv[1]
        if simulytion_type is not SIM_BIN and simulytion_type is not SIM_MUL:
            print("nr_simulations must be either '%s' or '%s'", (SIM_BIN, SIM_MUL))
        if nr_arguments > 2:
            try:
                nr_simulations = int(sys.argv[2])
            except:
                print("nr_simulations must be integer")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    random.seed()

    corpus = pickle.load(open(config.lda_file, 'rb'))
    corpus_x = corpus['x']
    corpus_z = corpus['z']
    z0 = corpus['z0']

    dim_x = corpus['dim_x']
    dim_z = corpus['dim_z']

    nr_docs = len(corpus_x)
    sample_size = config.nr_documents

    set_types = ["train"]
    if config.generate_testset:
        set_types.append("test")

    sample_x_all = np.zeros([sample_size, dim_x, nr_simulations])  # Documents in word space, with reduced dimensionality
    sample_t_all = np.zeros([sample_size, nr_simulations])  # Treatment assignment
    sample_mu0_all = np.zeros([sample_size, nr_simulations])  # True control outcome
    sample_mu1_all = np.zeros([sample_size, nr_simulations])  # True treatment outcome
    sample_yF_all = np.zeros([sample_size, nr_simulations])  # Factual outcome
    sample_yCF_all = np.zeros([sample_size, nr_simulations])  # Counterfactual outcome

    for set_type in set_types:
        for sim in range(nr_simulations):
            # Sample X documents
            doc_ids = sorted(random.sample(range(nr_docs), sample_size))

            # Sample "treated" centroid
            # Mobile decive
            z1_id = random.randint(0, nr_docs-1)
            z1 = sparse_to_dense(corpus_z[z1_id], dim_z)
            # Time of reading
            z2_id = random.randint(0, nr_docs-1)
            z2 = sparse_to_dense(corpus_z[z2_id], dim_z)

            sample_x = np.zeros([sample_size, dim_x])  # Documents in word space, with reduced dimensionality
            sample_z = np.zeros([sample_size, dim_z])  # Documents in topic space
            sample_p = np.zeros([sample_size])  # Treatment probability
            sample_t = np.zeros([sample_size])  # Treatment assignment
            sample_mu0 = np.zeros([sample_size])  # True control outcome
            sample_mu1 = np.zeros([sample_size])  # True treatment outcome
            sample_yF = np.zeros([sample_size])  # Factual outcome
            sample_yCF = np.zeros([sample_size])  # Counterfactual outcome
            sample_strength = np.zeros([sample_size])  # Time effect on y
            count = 0
            for d in doc_ids:
                x = sparse_to_dense(corpus_x[d], dim_x)
                sample_x[count] = x
                z = sparse_to_dense(corpus_z[d], dim_z)
                sample_z[count] = z
                p = calc_treatment_probability(k, z, z0, z1)
                sample_p[count] = p
                t = random.random() < p
                sample_t[count] = t
                if SIM_MUL:
                    sample_strength[count] = np.dot(z, z2)
                yF, yCF, mu0, mu1 = simulate_outcomes(C1, C2, t, z, z0, z1, z2)
                sample_yF[count] = yF
                sample_yCF[count] = yCF
                sample_mu0[count] = mu0
                sample_mu1[count] = mu1

                count += 1

            sample_x_all[:,:,sim] = sample_x
            sample_t_all[:,sim] = sample_t
            sample_mu0_all[:,sim] = sample_mu0
            sample_mu1_all[:,sim] = sample_mu1
            sample_yF_all[:,sim] = sample_yF
            sample_yCF_all[:,sim] = sample_yCF

            to_save = {
                'x': sample_x_all,
                't': sample_t_all,
                'yf': sample_yF_all,
                'ycf': sample_yCF_all,
                'mu0': sample_mu0_all,
                'mu1': sample_mu1_all
                #'ymul': np.array(1, dtype=np.int64),
                #'yadd' : np.array(0, dtype=np.int64),
                #'ate': np.array(4, dtype=np.int64),
            }
            if SIM_MUL:
                to_save['s'] = sample_strength

            # Save a simulation run
            np.save("simulation_outcome."+set_type, to_save)
