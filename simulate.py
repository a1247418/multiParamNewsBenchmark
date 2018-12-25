import random
import pickle
import logging
import config

import numpy as np
import scipy.stats

import saver


def dot_norm(v, w):
    """
    Normalized two vectors and returns their dot product.
    """
    v_n = v/np.linalg.norm(v)
    w_n = w/np.linalg.norm(w)
    value = np.dot(v_n, w_n)
    return value


def simulate_outcomes(C, z, centroids, strengths, for_treatment=None):
    """
    Simulates the outcome for a single unit/treatment pair.
    :param C: Weighting constant
    :param z: A unit in topic space
    :param centroids: Treatment centroids in topic space
    :param strengths: Treatment strength
    :param for_treatment: If None, the output is calculated for each treatment. If integer, all outputs are for
        the treatment with this id.
    :return: mu, y Where mu is the true treatment effect, and y is its noisy measurement.
    """
    out_length = len(strengths)
    mu = np.zeros(out_length)
    y = np.zeros(out_length)
    std_dev = config.out_std

    if for_treatment is None:
        for treatment in range(out_length):
            dot_prod = np.sqrt(dot_norm(z, centroids[treatment]))
            mu[treatment] = C * (1-np.power(1 - 2*strengths[treatment], 2)) * dot_prod
            y[treatment] = mu[treatment] + np.random.normal(0, std_dev)
    else:
        dot_prod = np.sqrt(dot_norm(z, centroids[for_treatment]))
        for sample in range(out_length):
            mu[sample] = C * (1-np.power(1 - 2*strengths[sample], 2)) * dot_prod
            y[sample] = mu[sample] + np.random.normal(0, std_dev)

    return mu, y


def calc_treatment_probability(k, z, centroids):
    """
    Returns the normalized weight of each treatment option.
    :param k: Weighting constant kappa. Higher k means stronger influence of proximity to centroid.
    :param z: A unit in topic space
    :param centroids:
    :return: Vector of probabilities for each treatment. Sums to 1.
    """
    n_treatments = len(centroids)
    term = np.zeros([n_treatments], dtype=np.float64)

    for i in range(n_treatments):
        term[i] = np.power(np.e, k * dot_norm(z, centroids[i]))

    p = np.zeros([n_treatments])
    for i in range(n_treatments):
        p[i] = term[i] / sum(term)

    return p


def sample_treatment(probability_weights):
    """
    Returns a non-negative integer identifying the treatment, selected according to the probability weights.
    :param probability_weights: Vector of weights summing to 1.
    :return: Treatment ID
    """
    assert 0.99 < sum(probability_weights) < 1.01

    t_id = np.random.choice(range(len(probability_weights)), p=probability_weights)

    return t_id


def sample_treatment_strength(z, centroid_z):
    """
    Returns a nonnegative number [0,1] biased by the dot product of the given vectors.
    :param z: Document vector in topic space
    :param centroid_z: Treatment centroids in topic space
    :return: Treatment strength
    """
    mu = config.str_mean
    sig = config.str_std

    strength = np.sqrt(dot_norm(z, centroid_z))

    noise = scipy.stats.truncnorm.rvs((-strength - mu) / sig, (1 - strength - mu) / sig, loc=mu, scale=sig, size=1)

    noisy_strength = strength + noise

    return noisy_strength


def sparse_to_dense(x, n_dims):
    """
    Takes a sparse and returns a dense vector.
    :param x: Sparse vector to transform
    :param n_dims: Number of dimensions of dense vector
    :return: Dense representation of sparse vec
    """
    x_dense = np.zeros(n_dims)
    for i in x:
        x_dense[i[0]] = i[1]
    return x_dense


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    random.seed(config.seed)
    np.random.seed(config.seed)

    ''' Get the default parameters. '''
    n_simulations = config.n_simulations
    kappa = config.k
    C = config.C
    treatment_types = config.treatment_types
    n_cf_samples = config.n_cf_samples
    n_parametric_treatments = np.sum(treatment_types)

    ''' Load data '''
    print("Loading corpus...")
    corpus = pickle.load(open(config.lda_file, 'rb'))
    corpus_x = corpus['x']
    corpus_z = corpus['z']

    dim_x = corpus['dim_x']
    dim_z = corpus['dim_z']

    n_docs = len(corpus_x)
    sample_size = config.n_documents

    ''' Simulate n_simulations many samples of the data. '''
    n_treatments = len(treatment_types)  # Here, the control group is also counted as treatment
    assert n_treatments <= sample_size

    # For training set and possibly test set:
    for set_type in config.sets:
        n_simulations = config.n_simulations[set_type]

        # Resimulate n_simulations times with the same data, but newly chosen treatment assignments/outcomes
        for sim in range(n_simulations):
            print("Simulation %d/%d of %s data" % (sim + 1, n_simulations, set_type))

            # Sample X documents
            doc_ids = sorted(random.sample(range(n_docs), sample_size))

            # Sample centroids for each treatment
            treatment_centroids_z = []
            treatment_centroids_x = []
            all_centroid_ids = []
            for i in range(n_treatments):
                # Choose a random centroid
                centroid_id = None
                while centroid_id is None:
                    proposal_centroid_id = random.randint(0, n_docs - 1)
                    if proposal_centroid_id not in all_centroid_ids:
                        centroid_id = proposal_centroid_id
                        all_centroid_ids.append(proposal_centroid_id)
                # Centroid in topic space
                centroid_z = sparse_to_dense(corpus_z[centroid_id], dim_z)
                treatment_centroids_z.append(centroid_z)
                centroid_x = sparse_to_dense(corpus_x[centroid_id], dim_x)
                treatment_centroids_x.append(centroid_x)

            # For each document: get its data vector, treatment assignment, and outcome
            sample_x = np.zeros([sample_size, dim_x])  # Documents in word space; reduced dimensions
            sample_z = np.zeros([sample_size, dim_z])  # Documents in topic space; reduced dimensions
            sample_t = np.zeros([sample_size])  # Treatment assignment
            sample_mu = np.zeros([sample_size, n_treatments])  # True outcome
            sample_y = np.zeros([sample_size, n_treatments])  # Noisy outcome
            sample_strength = np.zeros([sample_size, n_treatments])  # Treatment strength
            # Additional samples for parametric treatments to cover the whole range of counterfactual options
            sample_mu_param = np.zeros([sample_size, n_parametric_treatments, n_cf_samples])
            sample_y_param = np.zeros([sample_size, n_parametric_treatments, n_cf_samples])
            sample_strength_param = np.zeros([sample_size, n_parametric_treatments, n_cf_samples])

            # Generate data for each sampled document
            for count, d in enumerate(doc_ids):
                x = sparse_to_dense(corpus_x[d], dim_x)
                sample_x[count] = x

                z = sparse_to_dense(corpus_z[d], dim_z)
                sample_z[count] = z

                p = calc_treatment_probability(kappa, z, treatment_centroids_z)
                t = sample_treatment(p)

                sample_t[count] = t

                # Calculate treatment strength for parametric treatments. For binary treatments it's 1.
                sample_strength[count] = np.ones([n_treatments])
                for i in range(n_treatments):
                    # If treatment is parametric, calculate strength
                    if treatment_types[i]:
                        sample_strength[count, i] = sample_treatment_strength(z, treatment_centroids_z[i])

                mu, y = simulate_outcomes(C, z, treatment_centroids_z, sample_strength[count])
                sample_y[count] = y
                sample_mu[count] = mu

                # Additional parametric samples
                param_idx = 0
                for t_type in treatment_types:
                    if t_type == 1:
                        sample_strength_param[count, param_idx] = np.random.random(n_cf_samples)
                        mu_pcf, y_pcf = simulate_outcomes(C, z, treatment_centroids_z,
                                                          sample_strength_param[count][param_idx], for_treatment=t_type)
                        sample_y_param[count, param_idx] = y_pcf
                        sample_mu_param[count, param_idx] = mu_pcf
                        param_idx += 1

            ''' Save data set '''
            to_save = {
                'centroids_z': treatment_centroids_z,  # For analysis purposes only
                'centroids_x': treatment_centroids_x,  # For analysis purposes only
                'z': sample_z,  # For analysis purposes only
                'x': sample_x,
                't': sample_t,
                'y': sample_y,
                'mu': sample_mu,
                's': sample_strength,
                'y_pcf': sample_y_param,
                'mu_pcf': sample_mu_param,
                's_pcf': sample_strength_param,
                'treatment_types': treatment_types  # Whether a treatment is parametric
            }

            file_name_modifyer = set_type + ''.join(str(e) for e in treatment_types) + "_" + str(sample_size) + "k" + \
                                 str(kappa) + "_" + str(sim)

            # Save as numpy file
            if config.save_as_numpy:
                saver.save_as_npy("simulation_outcome." + file_name_modifyer, to_save)


            # Save as binary file
            if config.save_as_bin:
                saver.save_as_binary(file_name_modifyer, to_save)
