"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        if self.shuffle:
            iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            iterator = iter(range(len(self.dataset)))

        curr_batch = []
        for idx in iterator:
            curr_batch.append(self.dataset[idx])

            if len(curr_batch) == self.batch_size:
                yield self._build_batch_dict(curr_batch)
                curr_batch = []

        if curr_batch and not self.drop_last:
            yield self._build_batch_dict(curr_batch)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def _build_batch_dict(self, batch):
        batch_dict = {}
        for label_value_pair in batch:
            for key, value in label_value_pair.items():
                if key not in batch_dict:
                    batch_dict[key] = []
                batch_dict[key].append(value)

        batch_dict = {label: np.array(values) for label, values in batch_dict.items()}
        return batch_dict

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################

        r = len(self.dataset) % self.batch_size
        length = len(self.dataset) // self.batch_size

        if self.drop_last:
            return length

        length = length + 1 if r else length
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
