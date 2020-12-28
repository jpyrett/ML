# multi class entropy / information gain:
# entropy = -(m / m + n+ o) log2 (m / m + n + o)   +  ....
#         = -p1Log2(m/total) -p2Log2(n/total) - p3Log2(o/total)    

import numpy as np
print(  (-1 * (8/13) * np.log2(8/13)) +  (-1 * (3/13) * np.log2(3/13)) + ( -1 * (2/13) * np.log2(2/13)))