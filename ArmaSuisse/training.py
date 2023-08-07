import argparse, math
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--DATA_DIR', help='Directory where the json files are', type=str, default='/home/jknafou/Projects/ArmaSuisse/Dataset/')
parser.add_argument('--MODEL_DIR', help='Directory where the model will be saved', type=str, default='/home/jknafou/Projects/ArmaSuisse/Models/')
parser.add_argument('--MODEL', help='model name (huggingface) or path (local)', type=str, default='distilbert-base-uncased')
parser.add_argument('--LEARNING_RATE', help='learning rate for the model', type=float, default=2e-5)
parser.add_argument('--MAX_EPOCH', help='number of epochs the model will be trained on', type=int, default=5)
parser.add_argument('--WARMUP_PROPORTION', help='proportion of the steps the model learning rate will be going up to get to the actual learning rate', type=float, default=0.1)
parser.add_argument('--BATCH_SIZE', help='training batch size', type=int, default=24)
parser.add_argument('--GRAD_ACC', help='gradient accumulation when batch size is too high for GPU memory (1 equals no gradient accumulation)', type=int, default=1)
parser.add_argument('--K_FOLD', help='using the k-th fold split as the test set, the k+1 as the dev set and k+2 to k+9 as training set', type=int, default=0)
parser.add_argument('--N_FOLD', help='number of fold in our cross fold splitting', type=int, default=10)
parser.add_argument('--DEVICE_ID', help='gpu id for computation allocation', type=int, default=0)

args = parser.parse_args()

# if 'large' in args.MODEL:
#     args.BATCH_SIZE = math.ceil(args.BATCH_SIZE/5)
#     args.GRAD_ACC = 5

results2dataset(args.DATA_DIR)
cross_fold_splitting(args.DATA_DIR, args.N_FOLD)

model = Models(args.MODEL, args.LEARNING_RATE, args.MAX_EPOCH, args.WARMUP_PROPORTION, args.BATCH_SIZE, args.GRAD_ACC, args.MODEL_DIR, args.K_FOLD, args.DEVICE_ID)
data = Data(args.DATA_DIR, model.tokenizer, int(model.batch_size/model.n_gpu) if model.n_gpu != 0 else model.batch_size, args.K_FOLD, args.N_FOLD)
model.training(data)
model.testing(data)