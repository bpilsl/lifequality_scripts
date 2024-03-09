import argparse

import numpy as np
from numpy import random


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a configuration file with specified active pixels.")
    parser.add_argument("nmbActivePixel", type=int, help="Number of active pixels")
    parser.add_argument("output_file", type=str, help="Output file for the configuration")
    parser.add_argument("-f", "--filled_file", type=str, help="Optional filled config file")

    return parser.parse_args()


def parse_matrix_file(file):
    config = np.empty((64, 64), dtype=object)
    with open(file) as f:
        for line in f:
            splitted = line.split(' ')
            row = int(splitted[0])
            col = int(splitted[1])

            d = {'row': splitted[0], 'col': splitted[1], 'mask': splitted[2], 'etc': splitted[3:]}
            config[row, col] = d

    return config


def main():
    args = parse_args()

    activePixels = set()

    # If a filled config file is provided, read the filled pixels from it
    config = None

    if args.filled_file:
        config = parse_matrix_file(args.filled_file)

    # Generate random pixels until reaching the specified number of active pixels
    while len(activePixels) < args.nmbActivePixel:
        p = (random.randint(64), random.randint(64))
        if p not in activePixels:
            if config is not None:
                if config[p[0], p[1]]['mask'] != '1':
                    activePixels.add(p)
            else:
                activePixels.add(p)


    # Write the final config file
    print(activePixels)
    with open(args.output_file, 'w') as f:
        # Iterate over all pixels in the 64x64 grid
        for i in range(64):
            for j in range(64):
                currPix = (i, j)
                etc = ''
                for s in config[i, j]["etc"]:
                    etc += s.strip() + ' '

                # Check if the current pixel is in the list of pixels to be preserved
                if currPix in activePixels:
                    if config is not None:
                        line = f'{config[i, j]["row"]} {config[i, j]["col"]} {config[i, j]["mask"]} {etc}\n'

                    else:
                        line = f'{config[i, j]["row"]} {config[i, j]["col"]} 0 {etc}\n'

                else:
                    if config is None:
                        line = f'{i} {j} 1 0 0 0 -1\n'
                    else:
                        line = f'{i} {j} 1 {etc}\n'
                f.write(line)


if __name__ == "__main__":
    main()
