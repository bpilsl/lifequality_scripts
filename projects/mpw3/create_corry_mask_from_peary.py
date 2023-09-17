import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process pixel configuration file and generate masked pixel/row output file.")
parser.add_argument("input_file", help="Input configuration file")
parser.add_argument("output_file", help="Output file for masked pixels/rows")
args = parser.parse_args()

input_filename = args.input_file
output_filename = args.output_file
masked_rows = set()
masked_pixels = []

with open(input_filename, 'r') as input_file:
    current_row = None
    for line in input_file:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        row, col, mask, en_inj, hb_en, en_sfout, TDAC = map(int, parts[:7])
        if current_row is None or current_row != row:
            if len(masked_pixels) == 64:
                masked_rows.add(current_row)
            masked_pixels = []
            current_row = row
        if mask == 1:
            masked_pixels.append((row, col))

# Write the output file
with open(output_filename, 'w') as output_file:
    for row, col in masked_pixels:
        output_file.write(f"p {row:02d} {col:02d}\n")
    for row in masked_rows:
        output_file.write(f"r {row}\n")

print(f"Processed {len(masked_pixels)} masked pixels and {len(masked_rows)} masked rows, and saved to '{output_filename}'.")

