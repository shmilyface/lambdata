"""lambdata - a collection of data science helper functions"""

import pandas as pd
import numpy as np

ONES = pd.DataFrame(np.ones(10))

def increment(x):
	return x + 1