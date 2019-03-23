import numpy as np
import torch

def get_labels(filepath, limit=None):
	content = []
	with open(filepath) as f:
		for line in f.readlines():
			if limit is not None and len(content) >= limit:
				break
			content.append( float(line.strip()) )
	
	final_labels = torch.Tensor(content)

	return final_labels