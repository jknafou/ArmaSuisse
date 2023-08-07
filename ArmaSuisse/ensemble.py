import argparse
from data import *

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_DIR', help='Directory where the json files are', type=str, default='/home/jknafou/Projects/ArmaSuisse/Dataset/')
parser.add_argument('--MODEL_DIR', help='Directory where the model will be saved', type=str, default='/home/jknafou/Projects/ArmaSuisse/Models/')
parser.add_argument('--K_FOLD', help='using the k-th fold split as the test set, the k+1 as the dev set and k+2 to k+9 as training set', type=int, default=0)
parser.add_argument('--N_FOLD', help='number of fold in our cross fold splitting', type=int, default=10)

args = parser.parse_args()

data = Data(args.DATA_DIR, None, None, args.K_FOLD, args.N_FOLD)
data.ensemble(args.MODEL_DIR)