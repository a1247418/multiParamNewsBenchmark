import random
import pickle
import logging
import config
import numpy as np


def simulate_outcomes(C, z, centroids, strengths):
    """
    Simulates the outcome for a single unit/treatment pair.
    :param C: Weighting constant
    :param z: A unit in topic space
    :param centroids: Treatment centroids in topic space
    :param strengths: Treatment strength
    :return: mu, y Where mu is the true treatment effect, and y is the noisy measurement
    """
    nr_treatments = len(centroids)
    mu = np.zeros(nr_treatments)
    y = np.zeros(nr_treatments)

    mu[0] = 0
    for i in range(nr_treatments):
        mu[i] = mu[0] + C * strengths[i] * np.dot(z, centroids[i])
        y[i] = mu[i] + np.random.normal(0, 1)

    return mu, y


def calc_treatment_probability(k, z, centroids):
    """
    Returns the normalized weight of each treatment option.
    :param k: Weighting constant kappa. Higher k means stronger influence of proximity to centroid.
    :param z: A unit in topic space
    :param centroids:
    :return: Vector of probabilities for each treatment. Sums to 1.
    """
    nr_treatments = len(centroids)
    term = np.zeros([nr_treatments], dtype=np.float64)

    for i in range(nr_treatments):
        term[i] = np.power(np.e, k * np.dot(z, centroids[i]))

    p = np.zeros([nr_treatments])
    for i in range(nr_treatments):
        p[i] = term[i] / sum(term)

    return p


def sample_treatment(probability_weights):
    """
    Returns a nonnegative integer identifying the treatment, selected according to the probability weights.
    :param probability_weights: Vector of weights summing to 1.
    :return: Treatment ID
    """
    assert 0.99 < sum(probability_weights) < 1.01

    nr_treatments = len(probability_weights)
    rv = random.random()

    t_id = 0
    for i in range(nr_treatments-1):
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


def clamp(low, high, value):
    ''' Clamps the value between low and high. '''
    assert high - low > 0

    ret = value
    if ret > high:
        ret = high
    if ret < low:
        ret = low
    return ret


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    random.seed(config.seed)
    np.random.seed(config.seed)

    ''' Get the default parameters. '''
    nr_simulations = config.nr_simulations
    kappa = config.k
    C = config.C
    treatment_types = config.treatment_types
    
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
    nr_treatments = len(treatment_types) + 1  # Here, the control group is also counted as treatment
    sample_x_all = np.zeros(
        [sample_size, dim_x, nr_simulations])  # Documents in word space, with reduced dimensionality
    sample_z_all = np.zeros(
        [sample_size, dim_z, nr_simulations])  # Documents in topic space, with reduced dimensionality
    sample_t_all = np.zeros([sample_size, nr_simulations])  # Treatment assignment
    sample_mu_all = np.zeros([sample_size, nr_treatments, nr_simulations])  # Outcome truth
    sample_y_all = np.zeros([sample_size, nr_treatments, nr_simulations])  # Noisy outcome
    sample_strength_all = np.zeros([sample_size, nr_treatments, nr_simulations])

    # Resimulate nr_simulations times with the same data, but newly chosen centroids/treatment assignments/outcomes
    for sim in range(nr_simulations):
        # Sample X documents
        doc_ids = sorted(random.sample(range(nr_docs), sample_size))

        # Sample centroids for each treatment
        treatment_centroids_z = np.array([z0])
        treatment_centroids_x = np.array([x0])
        for i in range(nr_treatments-1):  # -1 since z0 is given
            centroid_id = random.randint(0, nr_docs - 1)
            # Centroid in topic space
            centroid = sparse_to_dense(corpus_z[centroid_id], dim_z)
            treatment_centroids_z = np.vstack([treatment_centroids_z, centroid])
            # Centroid in word space
            centroid = sparse_to_dense(corpus_x[centroid_id], dim_x)
            treatment_centroids_x = np.vstack([treatment_centroids_x, centroid])

        # For training set and possibly test set:
        for set_type in set_types:
            print("Simulation %d/%d of %s data" % (sim + 1, nr_simulations, set_type))
            # For each document: get its data vector, treatment assignment, and outcome
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
                sample_z[count] = z
                p = calc_treatment_probability(kappa, z, treatment_centroids_z)
                t = sample_treatment(p)
                sample_t[count] = t
                # Calculate treatment strength for parametric treatments. For all others it's 1.
                sample_strength[count] = np.ones([nr_treatments])
                for i in range(1, nr_treatments):
                    if treatment_types[i-1]:
                        sample_strength[count][i] = clamp(0, 1, config.str_const * np.dot(z, treatment_centroids_z[
                            i]) + np.random.normal(config.str_mean, config.str_std))
                mu, y = simulate_outcomes(C, z, treatment_centroids_z, sample_strength[count])
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
            'centroids_z': treatment_centroids_z,  # For analysis purposes only
            'centroids_x': treatment_centroids_x,  # For analysis purposes only
            'z': sample_z_all,
            'x': sample_x_all,
            't': sample_t_all,
            'y': sample_y_all,
            'mu': sample_mu_all,
            's': sample_strength_all,
            'param': treatment_types  # Whether a treatment is parametric
        }

        # Save all simulation runs for the current data set
        if config.save_as_numpy:
            np.save("simulation_outcome." + set_type, to_save)
        if config.save_as_csv:
            def sparsify(vec):
                sparse_vec = []
                for i in range(len(vec)):
                    if vec[i] != 0:
                        sparse_vec.append((i, vec[i]))
            continue
            # TODO: save as sparse csv
        if config.save_as_tfrecord:
            continue
            # TODO: save as tf_record