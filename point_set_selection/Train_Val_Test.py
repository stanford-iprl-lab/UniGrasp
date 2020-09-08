import numpy as np
np.random.seed(42)

class Train_Val_Test(object):
  def __init__(self, total_lists, splitting=[0.7, 0.1, 0.2],exclude_list=None):
    self._num_examples = len(total_lists)
    tmp_permutations = np.random.permutation(self._num_examples)
    self.num_train = int(splitting[0]*self._num_examples)
    self.num_val = int(splitting[1]*self._num_examples)
    self.num_test = self._num_examples - self.num_train - self.num_val
    self._train_pre = total_lists[tmp_permutations[0:self.num_train]]
    self._val_pre   = total_lists[tmp_permutations[self.num_train: self.num_train+self.num_val]]
    self._test_pre  = total_lists[tmp_permutations[self.num_train+self.num_val:]]
    if exclude_list is not None:
       self._train = [l for l in self._train_pre if l not in exclude_list] 
       self._val = [l for l in self._val_pre if l not in exclude_list]
       self._test = [l for l in self._test_pre if l not in exclude_list]
    else:
       self._train = self._train_pre
       self._val = self._val_pre
       self._test = self._test_pre
