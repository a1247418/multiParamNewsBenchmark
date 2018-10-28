import random
import pickle
import logging
import config
import struct

import numpy as np
import scipy.stats

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
    :return: mu, y Where mu is the true treatment effect, and y is its noisy measurement.
    """
    nr_treatments = len(strengths)
    mu = np.zeros(nr_treatments)
    y = np.zeros(nr_treatments)

    if for_treatment is None:
        mu[0] = C * np.dot(z, centroids[0][0])
        y[0] = mu[0] + np.random.normal(0, 0.2)

        for i in range(1, nr_treatments):
            dot_prod = np.sqrt(sum([np.dot(z, centroid) for centroid in centroids[i]]))
            mu[i] = mu[0] + C * (1-np.power(1 - 2*strengths[i], 2)) * dot_prod
            y[i] = mu[i] + np.random.normal(0, 0.2)
    else:
        mu0 = C * np.dot(z, centroids[0][0])
        for i in range(nr_treatments):
            dot_prod = np.sqrt(sum([np.dot(z, centroid) for centroid in centroids[for_treatment]]))

            mu[i] = mu0 + C * (1-np.power(1 - 2*strengths[i], 2)) * dot_prod
            y[i] = mu[i] + np.random.normal(0, 0.2)

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


def sample_treatment_strength(z, centroids_z):
    """
    Returns a nonnegative number [0,1] biased by the dot product of the given vectors.
    :param z: Document vector in topic space
    :param centroids_z: Treatment centroids in topic space
    :return: Treatment strength
    """

    mu = config.str_mean
    sig = config.str_std

    strength = np.sqrt( sum([np.dot(z, centroid) for centroid in centroids_z]) )

    noise = scipy.stats.truncnorm.rvs((-strength - mu) / sig, (1 - strength - mu) / sig, loc=mu, scale=sig, size=1)

    noisy_strength = strength + noise

    return noisy_strength


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

    # For training set and possibly test set:
    for set_type in config.sets:
        nr_simulations = config.nr_simulations[set_type]

        # Resimulate nr_simulations times with the same data, but newly chosen treatment assignments/outcomes
        for sim in range(nr_simulations):
            print("Simulation %d/%d of %s data" % (sim + 1, nr_simulations, set_type))

            # Sample X documents
            doc_ids = sorted(random.sample(range(nr_docs), sample_size))

            # Sample centroids for each treatment
            treatment_centroids_z = [np.array([z0])]
            treatment_centroids_x = [np.array([x0])]
            for i in range(nr_treatments - 1):  # -1 since z0 is given
                treatment_centroids_x.append(np.array([]))
                treatment_centroids_z.append(np.array([]))
                for j in range(config.nr_centroids):
                    centroid_id = random.randint(0, nr_docs - 1)
                    # Centroid in topic space
                    centroid = sparse_to_dense(corpus_z[centroid_id], dim_z)
                    if j == 0:
                        treatment_centroids_z[i+1] = np.append(treatment_centroids_z[i+1], centroid)
                        if config.nr_centroids == 1:
                            treatment_centroids_z[i + 1] = [treatment_centroids_z[i+1]]
                    else:
                        treatment_centroids_z[i+1] = np.vstack([treatment_centroids_z[i+1], centroid])
                    # Centroid in word space
                    centroid = sparse_to_dense(corpus_x[centroid_id], dim_x)
                    if j == 0:
                        treatment_centroids_x[i+1] = np.append(treatment_centroids_x[i+1], centroid)
                        if config.nr_centroids == 1:
                            treatment_centroids_x[i + 1] = [treatment_centroids_x[i+1]]
                    else:
                        treatment_centroids_x[i+1] = np.vstack([treatment_centroids_x[i+1], centroid])

                first_treatment_centroids_z = np.array([i[0] for i in treatment_centroids_z])

            # For each document: get its data vector, treatment assignment, and outcome
            sample_x = np.zeros([sample_size, dim_x])  # Documents in word space; reduced dimensions
            sample_z = np.zeros([sample_size, dim_z])  # Documents in topic space; reduced dimensions
            sample_t = np.zeros([sample_size])  # Treatment assignment
            sample_mu = np.zeros([sample_size, nr_treatments])  # True outcome
            sample_y = np.zeros([sample_size, nr_treatments])  # Noisy outcome
            sample_strength = np.zeros([sample_size, nr_treatments])  # Treatment strength

            # Additional samples for parametric treatments to cover the whole range of counterfactual options
            sample_mu_param = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples])
            sample_y_param = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples])
            sample_strength_param = np.zeros([sample_size, nr_parametric_treatments, nr_cf_samples])

            # Generate data for each sampled document
            for count, d in enumerate(doc_ids):
                x = sparse_to_dense(corpus_x[d], dim_x)
                sample_x[count] = x

                z = sparse_to_dense(corpus_z[d], dim_z)
                sample_z[count] = z

                p = calc_treatment_probability(kappa, z, first_treatment_centroids_z)
                t = sample_treatment(p)

                sample_t[count] = t

                # Calculate treatment strength for parametric treatments. For binary ones it's 1.
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
                print("Saving to numpy file...")
                np.save("simulation_outcome." + file_name_modifyer, to_save)

            # Save as binary file
            if config.save_as_bin:
                print("Saving to binary...")

                def write2dmatrix(mat, mat_name, file_name_modifyer):
                    '''Write a 2D matrix to a file.'''
                    dim0 = mat.shape[0]
                    dim1 = mat.shape[1]

                    with open('simulation_outcome.%s.%s' % (file_name_modifyer, mat_name), 'wb') as binfile:
                        header = struct.pack('2I', dim0, dim1)
                        binfile.write(header)
                        for i in range(dim1):
                            data = struct.pack('%id' % dim0, *mat[:, i])
                            binfile.write(data)
                        binfile.close()

                # Flatten matrices to 2D for export to binary file
                x2export = sample_strength[np.arange(sample_size), sample_t.astype(int)]
                x2export = np.concatenate([sample_x,
                                           sample_t[:, np.newaxis],
                                           x2export[:, np.newaxis]], 1)
                write2dmatrix(x2export, "x", file_name_modifyer)

                y2export = sample_y[np.arange(sample_size), sample_t.astype(int)]
                y2export = y2export[:, np.newaxis]
                write2dmatrix(y2export, "y", file_name_modifyer)

                t_cf = []
                for treatment in treatment_types:
                    if treatment == 1:
                        t_cf += [treatment for i in range(nr_cf_samples)]

                t_cf = np.array(t_cf * sample_size)[:, np.newaxis]
                xpcf2export = np.concatenate([t_cf, np.reshape(sample_strength_param, [-1, 1])], 1)
                write2dmatrix(xpcf2export, "xpcf", file_name_modifyer)

                ypcf2export = np.reshape(sample_y_param, [-1, 1])
                write2dmatrix(ypcf2export, "ypcf", file_name_modifyer)

                # As binary cf samples don't neatly fit into a matrix shape, save them as vector of:
                # sample-id, t, s, y for each binary cf sample
                bin_cf2export = np.array([])
                for i in range(sample_size):
                    for t_id, t_type in enumerate([0]+treatment_types):
                        if t_type == 0 and sample_t[i] != t_id:
                            bin_cf2export = np.append(bin_cf2export, float(i+1))
                            bin_cf2export = np.append(bin_cf2export, float(t_id))
                            bin_cf2export = np.append(bin_cf2export, sample_strength[i, t_id])
                            bin_cf2export = np.append(bin_cf2export, sample_y[i, t_id])

                bin_cf2export = np.reshape(bin_cf2export, [-1, 1])
                write2dmatrix(bin_cf2export, "bcf", file_name_modifyer)
