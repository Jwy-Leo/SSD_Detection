import numpy as np
class _sampler(object):
    def __init__(self):
        pass
    def __call__(self, reference_set):
        pass
class random_sampler(_sampler):
    def __init__(self):
        super(random_sampler,self).__init__()
    def __call__(self, reference_set, sample_num):
        assert type(reference_set) is list
        sample_index = np.random.choice(reference_set, sample_num)
        return sample_index
class uncertainty_sampler(_sampler):
    def __init__(self):
        super(uncertainty_sampelr,self).__init__()
    def __call__(self, reference_set):
        # assert type(reference_set) is list
        # assert type(reference_set[0]) is tuple
        assert type(reference_set) is np.ndarray
        index_max = reference_set[:,0].argsort()
        data = reference_set[index_max,1]
        return data 
