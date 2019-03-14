from collections import Counter
import numpy as np
import pandas as pd
import torch

maxlen = 0

def pad_sequence(s):
	global maxlen
	padded = np.zeros((maxlen,), dtype=np.int64)
	if len(s) > maxlen:
		padded[:] = s[:maxlen]
	else:
		padded[:len(s)] = s
	return padded

def one_hot_encode(s):
	# TODO
	return s

def return_data(filepath, num_sequences=None):
	global maxlen
	with open(filepath) as f:
		seq_df = pd.DataFrame(f)
		if num_sequences is not None:
			seq_df = seq_df[:num_sequences]

	seq_df = seq_df[0].apply(lambda x: x.strip().split(" "))
	words = Counter()
	for s in seq_df:
		words.update(token for token in s)

	words = sorted(words, key=lambda x: int(x), reverse=False)
	words = ['_PAD','_UNK'] + words

	word2idx = {o:i for i,o in enumerate(words)}
	idx2word = {i:o for i,o in enumerate(words)}

	maxlen = seq_df.apply(len).values.max()

	padded_seq_df = seq_df.apply(pad_sequence)

	final_data = padded_seq_df.apply(one_hot_encode)
	final_data = torch.Tensor(final_data)


	return final_data
