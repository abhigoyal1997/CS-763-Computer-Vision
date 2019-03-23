import numpy as np
import torch

def get_labels(filepath):
	content = []
	with open(filepath) as f:
		for line in f.readlines():
			content.append( float(line.strip()) )
	
	final_labels = torch.Tensor(content)
	
	return final_labels
