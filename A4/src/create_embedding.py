from collections import Counter
import numpy as np
import torch

veclen = 0
trainpath = '../../../A4_data/train_data.txt'

def one_hot_encode(ind, obj):
	vec = [0.0]*veclen #np.zeros((veclen,), dtype=np.float32)
	if obj not in ['_PAD','_UNK']:
		vec[ind] = 1.0
	return vec

def get_embedding():
	global veclen
	content = []
	with open(trainpath) as f:
		for line in f.readlines():
			content.append(line.strip().split(" "))

	words = Counter()
	for s in content:
		words.update(token for token in s)

	words = sorted(words, key=lambda x: int(x), reverse=False)
	veclen = len(words) # We don't include _PAD and _UNK as their embedding is all zeros
	words =  words + ['_PAD','_UNK']
	
	word2emb = {o:one_hot_encode(i,o) for i,o in enumerate(words)}

	return word2emb