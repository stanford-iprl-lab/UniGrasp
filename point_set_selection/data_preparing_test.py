import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,os.pardir))

from Train_Val_Test import *
import numpy as np
np.random.seed(42)
data_top_dir = os.path.join('../data/objects')
total_ids = []

for line_id in os.listdir(data_top_dir):
  sub_dir = os.path.join(data_top_dir,line_id)
  print(sub_dir)
  f3 = [line for line in os.listdir(sub_dir) if line.endswith('grasp3_new.npz')]
  f4 = [line for line in os.listdir(sub_dir) if line.endswith('grasp4_new.npz')]
  f5 = [line for line in os.listdir(sub_dir) if line.endswith('grasp5_new.npz')]
  print("len1",len(f3),len(f4),len(f5))
  total_ids.append(line_id)

train_val_test_list = Train_Val_Test(np.array(total_ids),splitting=[1.0,0.0,0.0])

print("train num %d , val num %d , test num %d" % (len(train_val_test_list._train),len(train_val_test_list._val),len(train_val_test_list._test)))
