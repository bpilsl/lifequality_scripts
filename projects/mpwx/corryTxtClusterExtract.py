import re
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def parse_events(file_content):
    events = file_content.split("=== ")
    return [event for event in events if event.strip()]

def find_large_clusters_and_pixels(events):
    large_cluster_events = []
    pixel_data = []
    cluster_positions = []
    event_numbers = []
    for event in events:
        lines = event.splitlines()
        event_number = None        
        match = re.search(r'(\d+) ===', lines[0])         
        if match:
            event_number = match.group(1)                
        in_rd50 = False
        pixels = np.full((64, 64), -1)  # Initialize a 64x64 array to store pixel values
        for line in lines:
            if "--- RD50_MPWx_base_0 ---" in line:
                in_rd50 = True
            elif "---" in line and in_rd50:
                # Exit if another detector section starts
                in_rd50 = False
            if in_rd50 and "Cluster" in line:
                parts = line.split(',')
                cluster_pos = (float(parts[5]), float(parts[6]))
                cluster_size = int(parts[9].strip())
                if cluster_size >= 3:
                    large_cluster_events.append(f"=== {event_number} ===\n{event}")
                    pixel_data.append(pixels)                    
                    cluster_positions.append(cluster_pos)
                    event_numbers.append(event_number)  # Store the event number
                    break
            if in_rd50 and "Pixel" in line:
                parts = line.split(',')
                x = int(re.search(r'\d+', parts[0]).group(0))
                y = int(parts[1].strip())
                value = float(parts[3].strip())  # Extract the hit value from parts[3]
                if 0 <= x < 64 and 0 <= y < 64:  # Ensure the pixel coordinates are within bounds                   
                    pixels[x, y] = value  # Aggregate the values for each pixel
    return large_cluster_events, pixel_data, event_numbers, cluster_positions.copy()

def main(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    events = parse_events(file_content)
    large_cluster_events, pixel_data, event_numbers, cluster_positions = find_large_clusters_and_pixels(events)
    
    for event in large_cluster_events:        
        #print(event)
        pass

    # Plot the pixels using sns.heatmap
    for i, (pixels, event_number, cluster_pos) in enumerate(zip(pixel_data, event_numbers, cluster_positions)):
        if pixels.any():  # Check if there are any non-zero pixel values
            fig = plt.figure(figsize=(20, 13))
            ax = sns.heatmap(pixels, cmap='viridis', cbar=True, annot=True)
            ax.scatter([cluster_pos[1] + .5], [cluster_pos[0] + .5], marker='x', c='black', s=70) # add .5 to center with the heatmap cells, otherwise we align with the edges, just looks wrong..
            plt.xlim(cluster_pos[1] + 10, cluster_pos[1] - 10)
            plt.ylim(cluster_pos[0] + 10, cluster_pos[0] - 10)
            plt.title(f'Event {event_number}')
            print('plotting ', event_number)            
            plt.xlabel('X')
            plt.ylabel('Y')
            # plt.show()
            plt.savefig(f'{sys.argv[2]}/hitmapEvent{event_number}')
            plt.close(fig)

if __name__ == "__main__":
    file_path = sys.argv[1]  # Replace with your actual file path
    main(file_path)

