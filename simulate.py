import csv
import random
import pickle
import logging
import config
import struct
import numpy as np
import pdb


def simulate_outcomes(C, z, centroids, strengths, for_treatment=None):
    """
    Simulates the outcome for a single unit/treatment pair.
    :param C: Weighting constant
    :param z: A unit in topic space
    :param centroids: Treatment centroids in topic space
    :param strengths: Treatment strength
    :param for_treatment: If none, the output is calculated for each treatment. If integer, all outputs are for
        the treatment with this id.
    :return: mu, y Where mu is the true treatment effect, and y is the noisy measurement
    """
    nr_treatments = len(strengths)
    mu = np.zeros(nr_treatments)
    y = np.zeros(nr_treatments)

    if for_treatment is None:
        mu[0] = 0
        for i in range(nr_treatments):
            mu[i] = mu[0] + C * strengths[i] * np.dot(z, centroids[i])
            y[i] = mu[i] + np.random.normal(0, 1)
    else:
        mu0 = C * np.dot(z, centroids[0])
        for i in range(nr_treatments):
            mu[i] = mu0 + C * strengths[i] * np.dot(z, centroids[for_treatment])
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
    for i in range(nr_treatments - 1):
        if rv >= sum(probability_weights[:i + 1]):
            t_id = i + 1

    return t_id


def sample_treatment_strength(z, centroid_z):
    """
    Returns a nonnegative number [0,1] biased by the dot product of the given vectors.
    :param z: Document vector in topic space
    :param centroid_z: Treatment centroid in topic space
    :return: Treatment strength
    """
    strength = clamp(0, 1, config.str_const * np.dot(z, centroid_z) +
                     np.random.normal(config.str_mean, config.str_std))
    return strength


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
    nr_cf_samples = config.nr_cf_samples
    nr_parametric_treatments = np.sum(treatment_types)

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

    ''' Simulate nr_simulations many samples of the data. '''
    nr_treatments = len(treatment_types) + 1  # Here, the control group is also counted as treatment
    # Sample documents & centroids - these stay fixed for all simulations and sets
    # Sample X documents
    doc_ids = sorted(random.sample(range(nr_docs), sample_size))

    # Sample centroids for each treatment
    treatment_centroids_z = np.array([z0])
    treatment_centroids_x = np.array([x0])
    for i in range(nr_treatments - 1):  # -1 since z0 is given
        centroid_id = random.randint(0, nr_docs - 1)
        # Centroid in topic space
        centroid = sparse_to_dense(corpus_z[centroid_id], dim_z)
        treatment_centroids_z = np.vstack([treatment_centroids_z, centroid])
        # Centroid in word space
        centroid = sparse_to_dense(corpus_x[centroid_id], dim_x)
        treatment_centroids_x = np.vstack([treatment_centroids_x, centroid])

    # For training set and possibly test set:
    for set_type in config.sets:
        nr_simulations = config.nr_simulations[set_type]
        sample_x_all = np.zeros(
            [sample_size, dim_x, nr_simulations])  # Documents in word space, with reduced dimensionality
        sample_z_all = np.zeros(
            [sample_size, dim_z, nr_simulations])  # Documents in topic space, with reduced dimensionality
        sample_t_all = np.zeros([sample_size, nr_simulations])  # Treatment assignment
        sample_mu_all = np.zeros([sample_size, nr_treatments, nr_simulations])  # Outcome truth
        sample_y_all = np.zeros([sample_size, nr_treatments, nr_simulations])  # Noisy outcome
        sample_strength_all = np.zeros([sample_size, nr_treatments, nr_simulations])
        sample_mu_param_all = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples, nr_simulations])
        sample_y_param_all = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples, nr_simulations])
        sample_strength_param_all = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples, nr_simulations])

        # Resimulate nr_simulations times with the same data, but newly chosen treatment assignments/outcomes
        for sim in range(nr_simulations):
            print("Simulation %d/%d of %s data" % (sim + 1, nr_simulations, set_type))
            # For each document: get its data vector, treatment assignment, and outcome
            sample_x = np.zeros([sample_size, dim_x])  # Documents in word space, with reduced dimensionality
            sample_z = np.zeros([sample_size, dim_z])  # Documents in topic space
            sample_t = np.zeros([sample_size])  # Treatment assignment
            sample_mu = np.zeros([sample_size, nr_treatments])  # True outcome
            sample_y = np.zeros([sample_size, nr_treatments])  # Noisy outcome
            sample_strength = np.zeros([sample_size, nr_treatments])  # Time effect on y

            # Additional samples for parametric treatments to cover the whole range of counterfactual options
            sample_mu_param = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples])  # True outcome
            sample_y_param = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples])  # Noisy outcome
            sample_strength_param = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples])  # Time effect on y

            for count, d in enumerate(doc_ids):
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
                    # If treatment is parametric
                    if treatment_types[i - 1]:
                        sample_strength[count, i] = sample_treatment_strength(z, treatment_centroids_z[i])
                mu, y = simulate_outcomes(C, z, treatment_centroids_z, sample_strength[count])
                sample_y[count] = y
                sample_mu[count] = mu

                # Additional parametric samples
                param_idx = 0
                for t_type in treatment_types:
                    if t_type == 1:
                        sample_strength_param[count, param_idx] = np.random.random(nr_cf_samples)
                        mu_pcf, y_pcf = simulate_outcomes(C, z, treatment_centroids_z,
                                                          sample_strength_param[count][param_idx], for_treatment=t_type)
                        sample_y_param[count, param_idx] = y_pcf
                        sample_mu_param[count, param_idx] = mu_pcf
                        param_idx += 1

            ''' Gather samples '''
            sample_z_all[:, :, sim] = sample_z
            sample_x_all[:, :, sim] = sample_x
            sample_t_all[:, sim] = sample_t
            sample_mu_all[:, :, sim] = sample_mu
            sample_y_all[:, :, sim] = sample_y
            sample_strength_all[:, :, sim] = sample_strength

            sample_mu_param_all[:, :, :, sim] = sample_mu_param
            sample_y_param_all[:, :, :, sim] = sample_y_param
            sample_strength_param_all[:, :, :, sim] = sample_strength_param

        ''' Save data set '''
        to_save = {
            'centroids_z': treatment_centroids_z,  # For analysis purposes only
            'centroids_x': treatment_centroids_x,  # For analysis purposes only
            'z': sample_z_all,  # For analysis purposes only
            'x': sample_x_all,
            't': sample_t_all,
            'y': sample_y_all,
            'mu': sample_mu_all,
            's': sample_strength_all,
            'y_pcf': sample_y_param_all,
            'mu_pcf': sample_mu_param_all,
            's_pcf': sample_strength_param_all,
            'treatment_types': treatment_types  # Whether a treatment is parametric
        }

        # Save all simulation runs for the current data set
        if config.save_as_numpy:
            print("Saving to numpy file...")
            np.save("simulation_outcome." + set_type, to_save)
        if config.save_as_bin:
            print("Saving to binary...")
            def sparsify(vec):
                sparse_vec = []
                for i in range(len(vec)):
                    if vec[i] != 0:
                        sparse_vec.append((i, vec[i]))


            def write2dmatrix(mat, mat_name, set_name):
                dim0 = mat.shape[0]
                dim1 = mat.shape[1]
                n_sim = mat.shape[2]

                for sim in range(n_sim):
                    with open('simulation_outcome.%s%d.%s' % (set_name, sim, mat_name), 'wb') as binfile:
                        header = struct.pack('2I', dim0, dim1)
                        binfile.write(header)
                        for i in range(dim1):
                            data = struct.pack('%id' % dim0, *mat[:, i, sim])
                            binfile.write(data)
                        binfile.close()

            x2export = np.stack([sample_strength_all[np.arange(sample_size), sample_t_all[:, i].astype(int), i] for i in
                                 range(nr_simulations)], 1)
            x2export = np.concatenate([sample_x_all,
                                       sample_t_all[:, np.newaxis, :],
                                       x2export[:, np.newaxis, :]], 1)
            write2dmatrix(x2export, "x", set_type)

            y2export = np.stack([sample_y_all[np.arange(sample_size), sample_t_all[:, i].astype(int), i] for i in
                                 range(nr_simulations)], 1)
            y2export = y2export[:, np.newaxis, :]
            write2dmatrix(y2export, "y", set_type)

            t_cf = []
            for treatment in treatment_types:
                if treatment == 1:
                    t_cf += [treatment for i in range(nr_cf_samples)]

            t_cf = np.tile(np.array(t_cf * sample_size)[:, np.newaxis, np.newaxis], [1, 1, nr_simulations])
            xpcf2export = np.concatenate([t_cf, np.reshape(sample_strength_param_all, [-1, 1, nr_simulations])], 1)
            write2dmatrix(xpcf2export, "xcf", set_type)

            ypcf2export = np.reshape(sample_y_param_all, [-1, 1, nr_simulations])
            write2dmatrix(ypcf2export, "ycf", set_type)

        if config.save_as_tfrecord:
            continue
            # TODO: save as tf_record
