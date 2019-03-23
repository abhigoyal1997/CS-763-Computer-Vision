from collections import Counter
import numpy as np
import torch
from create_embedding import get_embedding

maxlen = 0

def pad_sequence(s):
	global maxlen
	padded = ['_PAD']*maxlen
	if len(s) > maxlen:
		padded[:] = s[:maxlen]
	else:
		padded[:len(s)] = s
	return padded

def get_data(filepath):
	global maxlen
	content = []
	with open(filepath) as f:
		for line in f.readlines():
			content.append(line.strip().split(" "))

	maxlen = max([len(s) for s in content])
	word2emb = get_embedding()
	
	padded_content = [pad_sequence(s) for s in content]
	final_content = [[word2emb[elem] for elem in s] for s in padded_content]
	final_data = torch.Tensor(final_content)

	return final_data