
import math
p = [0.24199236279, 0.50125273233, 0.25675490488] 
baseline_entropy = -sum(pi * math.log(pi) for pi in p)
print(baseline_entropy)