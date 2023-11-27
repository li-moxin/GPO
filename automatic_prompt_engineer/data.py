import random, copy
import numpy as np
import logging


def subsample_data(data, subsample_size, config):
    """
    Subsample data.
    """
    logging.info("Sampling data {}".format(subsample_size))

    if type(data) == list:
        inputs, outputs = [], []
        message = data[-1]
        assert message in ['shuffle_all', '']
        for i, split in enumerate(data[:-1]):
            input, output = subsample_data(split, subsample_size, config)
            inputs.extend(input)
            outputs.extend(output)
        if 'shuffle_all' in message:
            logging.info('shuffle all')

            ret = list(zip(inputs, outputs))
            random.shuffle(ret)
            inputs, outputs = zip(*ret)

        return inputs, outputs
                
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    if len(inputs) == subsample_size:
        logging.info("No need to sample")

        return data
    indices = config['seed'][:subsample_size]
    logging.info('sampling {} data with index {}'.format(subsample_size, indices[:subsample_size]))

    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs

def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
