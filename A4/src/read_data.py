from collections import Counter
import numpy as np
import torch
import json

maxlen = 0
embedpath = '../../A4_data/embedding.txt'

def pad_sequence(s):
	global maxlen
	padded = ['_PAD']*maxlen
	if len(s) > maxlen:
		padded[:] = s[:maxlen]
	else:
		padded[:len(s)] = s
	return padded


def embed(elem, word2emb):
	try:
		return word2emb[elem]
	except:
		return word2emb['_UNK']

def get_data(filepath, limit=None):
	global maxlen
	content = []
	with open(filepath) as f:
		for line in f.readlines():
			if limit is not None and len(content) >= limit:
				break
			content.append(line.strip().split(" "))

	maxlen = max([len(s) for s in content])
	with open(embedpath,'r') as file:
		word2emb = json.loads(file.read())
	
	padded_content = [pad_sequence(s) for s in content]
	final_content = [[embed(elem, word2emb) for elem in s] for s in padded_content]
	final_data = torch.Tensor(final_content)

	lengths = [len(s) for s in content]

	return final_data, lengths