import struct

import numpy as np

import config


def save_as_npy(file_name, file):
    print("Saving to numpy file...")
    np.save(file_name, file)


def save_as_binary(file_name, file):
    print("Saving to binary file...")

    sample_x = file['z']
    sample_t = file['t']
    sample_y = file['y']
    sample_strength = file['s']
    sample_y_param = file['y_pcf']
    sample_strength_param = file['s_pcf']
    treatment_types = file['treatment_types']
    
    n_documents = config.n_documents
    n_cf_samples = config.n_cf_samples

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
    x2export = sample_strength[np.arange(n_documents), sample_t.astype(int)]
    x2export = np.concatenate([sample_x,
                               sample_t[:, np.newaxis],
                               x2export[:, np.newaxis]], 1)
    write2dmatrix(x2export, "x", file_name)

    y2export = sample_y[np.arange(n_documents), sample_t.astype(int)]
    y2export = y2export[:, np.newaxis]
    write2dmatrix(y2export, "y", file_name)

    t_cf = []
    for treatment in treatment_types:
        if treatment == 1:
            t_cf += [treatment for i in range(n_cf_samples)]

    t_cf = np.array(t_cf * n_documents)[:, np.newaxis]
    xpcf2export = np.concatenate([t_cf, np.reshape(sample_strength_param, [-1, 1])], 1)
    write2dmatrix(xpcf2export, "xpcf", file_name)

    ypcf2export = np.reshape(sample_y_param, [-1, 1])
    write2dmatrix(ypcf2export, "ypcf", file_name)

    # As binary cf samples don't neatly fit into a matrix shape, save them as vector of:
    # sample-id, t, s, y for each binary cf sample
    bin_cf2export = np.array([])
    for i in range(n_documents):
        for t_id, t_type in enumerate([0]+treatment_types):
            if t_type == 0 and sample_t[i] != t_id:
                bin_cf2export = np.append(bin_cf2export, float(i+1))
                bin_cf2export = np.append(bin_cf2export, float(t_id))
                bin_cf2export = np.append(bin_cf2export, sample_strength[i, t_id])
                bin_cf2export = np.append(bin_cf2export, sample_y[i, t_id])

    bin_cf2export = np.reshape(bin_cf2export, [-1, 1])
    write2dmatrix(bin_cf2export, "bcf", file_name)
