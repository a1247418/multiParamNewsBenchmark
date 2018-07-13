import sys
import random
import pickle
import logging

import config

import numpy as np


def simulate_outcomes(C, z, centroids, strengths):
    nr_treatments = len(centroids)
    mu = np.zeros(nr_treatments)
    y = np.zeros(nr_treatments)

    mu[0] = 0
    for i in range(nr_treatments):
        mu[i] = mu[0] + C * strengths[i] * np.dot(z, centroids[i])
        y[i] = mu[i] + np.random.normal(0, 1)

    return mu, y


def calc_treatment_probability(k, z, centroids):
    """ Returns the normalized weight of each treatment option. """
    nr_treatments = len(centroids)
    term = np.zeros([nr_treatments], dtype=np.float64)

    for i in range(nr_treatments):
        term[i] = np.power(np.e, k * np.dot(z, centroids[i]))
    term_sum = sum(term)

    p = np.zeros([nr_treatments])

    for i in range(nr_treatments):
        p[i] = term[i] / term_sum

    return p / sum(p)


def sample_treatment(probability_weights):
    """
    Returns an integer >=0 identifying the treatment, selected according to the probability weights.
    :param probability_weights: Vector of weights summing to 1.
    :return: Treatment ID
    """
    assert 0.99 < sum(probability_weights) < 1.01

    nr_options = len(probability_weights)

    rv = random.random()

    t_id = 0
    for i in range(nr_options-1):
        if rv >= sum(probability_weights[:i + 1]):
            t_id = i + 1

    return t_id


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


def circular_clamp(low, high, value):
    ''' Clamps the value between low and high, but in a circular way. '''
    assert high - low > 0

    ret = value
    while ret > high:
        ret = low + (ret % high)
    while ret < low:
        ret = high - (low - ret)
    return ret


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    random.seed()

    ''' Get the default parameters. '''
    parametric_treatment = config.default_parametric_treatment
    nr_simulations = config.default_nr_simulations
    kappa = config.default_k
    C = config.default_C

    # TODO: Rethink input params.
    ''' Read parameters from console arguments. '''
    nr_arguments = len(sys.argv)
    if nr_arguments > 1:
        parametric_treatment = sys.argv[1]
        if parametric_treatment != 0 and parametric_treatment != 1:
            print("parametric_treatment must be either 0 or 1")
            parametric_treatment = config.default_parametric_treatment
        if nr_arguments > 2:
            try:
                nr_simulations = int(sys.argv[2])
            except:
                print("nr_simulations must be integer")

    are_treatments_param = [False, False]  # False: binary, True: parametric
    if parametric_treatment:
        are_treatments_param.append(True)

    ''' Load data '''
    print("Loading corpus...")
    corpus = pickle.load(open(config.lda_file, 'rb'))
    corpus_x = corpus['x']
    corpus_z = corpus['z']
    z0 = corpus['z0']
    x0 = corpus['x0']

    dim_x = corpus['dim_x']
    dim_z = corpus['dim_z']

    nr_docs = len(corpus_x)
    sample_size = config.nr_documents

    set_types = ["train"]
    if config.generate_testset:
        set_types.append("test")

    ''' Simulate nr_simulations many samples of the data. '''
    nr_treatments = len(are_treatments_param)
    sample_x_all = np.zeros(
        [sample_size, dim_x, nr_simulations])  # Documents in word space, with reduced dimensionality
    sample_z_all = np.zeros(
        [sample_size, dim_z, nr_simulations])  # Documents in topic space, with reduced dimensionality
    sample_t_all = np.zeros([sample_size, nr_simulations])  # Treatment assignment
    sample_mu_all = np.zeros([sample_size, nr_treatments, nr_simulations])  # Outcome truth
    sample_y_all = np.zeros([sample_size, nr_treatments, nr_simulations])  # Noisy outcome
    sample_strength_all = np.zeros([sample_size, nr_treatments, nr_simulations])
    for set_type in set_types:
        for sim in range(nr_simulations):
            print("Simulation %d/%d of %s data" % (sim + 1, nr_simulations, set_type))
            # Sample X documents
            doc_ids = sorted(random.sample(range(nr_docs), sample_size))

            # Sample "treated" centroids
            treatment_centroids = [z0]
            treatment_centroids_x = [x0]
            for i in range(nr_treatments-1):  # -1 since z0 is given
                centroid_id = random.randint(0, nr_docs - 1)
                centroid = sparse_to_dense(corpus_z[centroid_id], dim_z)
                treatment_centroids.append(centroid)

                centroid = sparse_to_dense(corpus_x[centroid_id], dim_x)
                treatment_centroids_x.append(centroid)


            for i in range(nr_treatments):
                treatment_centroids[i] /= sum(treatment_centroids[i])

            sample_x = np.zeros([sample_size, dim_x])  # Documents in word space, with reduced dimensionality
            sample_z = np.zeros([sample_size, dim_z])  # Documents in topic space
            sample_t = np.zeros([sample_size])  # Treatment assignment
            sample_mu = np.zeros([sample_size, nr_treatments])  # True outcome
            sample_y = np.zeros([sample_size, nr_treatments])  # Noisy outcome
            sample_strength = np.zeros([sample_size, nr_treatments])  # Time effect on y
            count = 0
            for d in doc_ids:
                x = sparse_to_dense(corpus_x[d], dim_x)
                sample_x[count] = x
                z = sparse_to_dense(corpus_z[d], dim_z)
                z /= sum(z)
                sample_z[count] = z
                p = calc_treatment_probability(kappa, z, treatment_centroids)
                t = sample_treatment(p)
                sample_t[count] = t
                sample_strength[count] = np.ones([nr_treatments])
                for i in range(nr_treatments):
                    if are_treatments_param[i]:
                        sample_strength[count][i] = circular_clamp(0, 1, 8 * np.dot(z, treatment_centroids[
                            i]) + np.random.normal(0, 1))
                mu, y = simulate_outcomes(C, z, treatment_centroids, sample_strength[count])
                sample_y[count] = y
                sample_mu[count] = mu

                count += 1

            ''' Gather samples '''
            sample_z_all[:, :, sim] = sample_z
            sample_x_all[:, :, sim] = sample_x
            sample_t_all[:, sim] = sample_t
            sample_mu_all[:, :, sim] = sample_mu
            sample_y_all[:, :, sim] = sample_y
            sample_strength_all[:, :, sim] = sample_strength

            ''' Save data set '''
            to_save = {
                'centroids': treatment_centroids,  # For analysis purposes only
                'centroids_x': treatment_centroids_x,  # For analysis purposes only
                'z': sample_z_all,
                'x': sample_x_all,
                't': sample_t_all,
                'y': sample_y_all,
                'mu': sample_mu_all,
                's': sample_strength_all,
                'param': are_treatments_param  # Whether a treatment is parametric
            }

            # Print sample
            ''' print("Printing head of data: ")
            nr_lines = 10
            for k in to_save.keys():
                line = k + "\t"
                for i in range(min(nr_lines, len(to_save[k]))):
                    line += str(to_save[k][i])
                print(line)
            '''

            # Save a simulation run
            np.save("simulation_outcome." + set_type, to_save)

            # TODO: offer to save as csv
