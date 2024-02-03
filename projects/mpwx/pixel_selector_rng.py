import sys
from numpy import random

pixels = []

nmbActivePixel = int(sys.argv[1])
output_file = sys.argv[2]

while len(pixels) < nmbActivePixel:
	p = (random.randint(64), random.randint(64))
	if not p in pixels:
		pixels.append(p)
		
		
f = open(output_file, 'w')
		
for i in range(64):
	for j in range(64):
		currPix = (i, j)
		if not currPix in pixels:
			f.write(f'{i} {j} 1 0 0 0 -1\n')
				
