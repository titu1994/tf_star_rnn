import numpy as np


def one_hot_sequence(x, depth=None):
    if depth is None:
        depth = int(x.max() + 1)

    one_hot = np.zeros(x.shape + (depth,))
    # TODO: speed up this operation its super slow.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot[i, j, int(x[i, j])] = 1

    return one_hot


def one_hot_encoder(labels, n_classes=None):
    '''
    A function that converts an array of integers, representing the labels
    of data, into an array of one-hot encoded vectors.
    Inputs:
    Labels      - a numpy array of integers (nsamples)
    n_classes   - (int) the number of classes if labels does not contain all
    Returns:
    A numpy array of one-hot encoded vectors with size [nsamples, n_classes]
    '''
    if n_classes is None:
        n_classes = int(labels.max() + 1)

    one_hot = np.zeros((labels.shape[0], n_classes))
    one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return one_hot


def pad_seqs(seqs):
    """Create the matrices from the datasets.
    This pad each sequence to the same length: the length of the
    longest sequence.
    Output:
    x -- a numpy array with shape (batch_size, max_time_steps, num_features)
    lengths -- an array of the sequence lengths
    """
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = seqs[0].shape[1]

    x = np.zeros((n_samples, maxlen, inputDimSize))

    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx], :] = s

    return x, np.array(lengths)


def random_split(n, test_frac=0.1):
    all_idx = np.arange(n)
    test_idx = all_idx[np.random.choice(
        n, int(np.ceil(test_frac * n)), replace=False)]
    train_idx = np.setdiff1d(all_idx, test_idx)
    assert (np.all(np.sort(np.hstack([train_idx, test_idx])) == all_idx))
    return train_idx, test_idx


def randomly_split_data(y, test_frac=0.5, valid_frac=0.0):
    '''
    Split the data into 3 sets:
    Returns a tuple with:
        train-idx       - A list of the train indices.
        valid-idx       - A list of the validation indices.
        test-idx        - A list of the testing indices.
    '''
    if len(y.shape) == 1:
        print('Warning: are you sure Y contains all the classes?')
        y = one_hot_encoder(y)

    split = None
    smallest_class = min(y.sum(axis=0))

    while split is None:
        not_test_idx, test_idx = random_split(
            y.shape[0], test_frac=test_frac + valid_frac)

        cond1 = np.all(y[not_test_idx, :].sum(axis=0) >=
                       0.8 * (1 - test_frac - valid_frac) * smallest_class)
        cond2 = np.all(y[test_idx, :].sum(axis=0) >=
                       0.8 * (test_frac + valid_frac) * smallest_class)

        if cond1 and cond2:
            if valid_frac != 0:
                while split is None:
                    final_test_idx, valid_idx = random_split(
                        y[test_idx].shape[0],
                        test_frac=valid_frac / (test_frac + valid_frac))

                    cond1 = np.all(
                        y[test_idx, :][final_test_idx, :].sum(axis=0) >=
                        0.6 * test_frac * smallest_class)
                    cond2 = np.all(
                        y[test_idx, :][valid_idx, :].sum(axis=0) >=
                        0.6 * valid_frac * smallest_class)
                    if cond1 and cond2:
                        split = (np.sort(not_test_idx),
                                 np.sort(test_idx[valid_idx]),
                                 np.sort(test_idx[final_test_idx]))
                        print('Split completed.\n')
                        break
                    else:
                        print('Valid labels unevenly split, resplitting...\n')
                        print(y[test_idx, :][final_test_idx, :].sum(axis=0))
                        print(0.6 * test_frac * smallest_class)
                        print(y[test_idx, :][valid_idx, :].sum(axis=0))
                        print(0.6 * valid_frac * smallest_class)
            else:
                split = (np.sort(not_test_idx), None, np.sort(test_idx))
                print('Split completed.\n')
        else:
            print('Test labels unevenly split, resplitting...\n')
            print(y[not_test_idx, :].sum(axis=0))
            print(0.7 * (1 - test_frac) * smallest_class)
            print(y[test_idx, :].sum(axis=0))
            print(0.7 * test_frac * smallest_class)

    if valid_frac != 0:
        print("Train:Validation:Testing - %d:%d:%d" % (len(split[0]),
                                                       len(split[1]),
                                                       len(split[2])))
    else:
        print("Train:Testing - %d:%d" % (len(split[0]), len(split[2])))

    return split


def load_data(dataset_name, seq_len=200):
    '''
    Returns:
    x - a n_samples long list containing arrays of shape (sequence_length,
                                                          n_features)
    y - an array of the labels with shape (n_samples, n_classes)
    '''
    print("Loading " + dataset_name + " dataset ...")

    if dataset_name == 'test':
        n_data_points = 5000
        sequence_length = 100
        n_features = 1
        x = list(np.random.rand(n_data_points, sequence_length, n_features))
        n_classes = 4
        y = np.random.randint(low=0, high=n_classes, size=n_data_points)

    if dataset_name == 'add':
        x, y = get_add(n_data=150000, seq_len=seq_len)

    if dataset_name == 'copy':
        return get_copy(n_data=150000, seq_len=seq_len)

    train_idx, valid_idx, test_idx = randomly_split_data(
        y, test_frac=0.2, valid_frac=0.1)

    x_train = [x[i] for i in train_idx]
    y_train = y[train_idx]
    x_valid = [x[i] for i in valid_idx]
    y_valid = y[valid_idx]
    x_test = [x[i] for i in test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_add(n_data, seq_len):
    x = np.zeros((n_data, seq_len, 2))
    x[:, :, 0] = np.random.uniform(low=0., high=1., size=(n_data, seq_len))
    inds = np.random.randint(seq_len / 2, size=(n_data, 2))
    inds[:, 1] += seq_len // 2
    for i in range(n_data):
        x[i, inds[i, 0], 1] = 1.0
        x[i, inds[i, 1], 1] = 1.0

    y = (x[:, :, 0] * x[:, :, 1]).sum(axis=1)
    y = np.reshape(y, (n_data, 1))
    return x, y


def get_copy(n_data, seq_len):
    x = np.zeros((n_data, seq_len + 1 + 2 * 10))
    info = np.random.randint(1, high=9, size=(n_data, 10))

    x[:, :10] = info
    x[:, seq_len + 10] = 9 * np.ones(n_data)

    y = np.zeros_like(x)
    y[:, -10:] = info

    x = one_hot_sequence(x)
    y = one_hot_sequence(y)

    n_train, n_valid, n_test = [100000, 10000, 40000]
    x_train = list(x[:n_train])
    y_train = y[:n_train]
    x_valid = list(x[n_train:n_train + n_valid])
    y_valid = y[n_train:n_train + n_valid]
    x_test = list(x[-n_test:])
    y_test = y[-n_test:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test
