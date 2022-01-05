import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('-c','--conf',dest='conf')
args = parser.parse_args()

with open('utils/conf.json','r') as f:
    conf = json.load(f)

train_datasets, eval_datasets = datasets.get_dataset("./data/0HP-7/",conf)

server = Server(conf, eval_datasets)
acc,loss = server.model_eval()
print("acc: %f, loss: %f\n" % ( acc, loss))