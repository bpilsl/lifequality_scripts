import pyvisa
import numpy as np
import csv
import time
import struct

# Set up VISA resource manager
rm = pyvisa.ResourceManager()

# IP address for the oscilloscope
IP_ADDRESS = "192.168.130.78"  # Replace with your oscilloscope's IP
VISA_ADDRESS = f"TCPIP::{IP_ADDRESS}::INSTR"

# Set desired record length (adjust based on oscilloscope capability)
RECORD_LENGTH = 2500  # Example: 2500 points

def fetch_waveform_data(oscilloscope, channel=1):
    try:
        # Set up the oscilloscope for waveform data acquisition
        oscilloscope.write(f"DATa:SOUrce CH{channel}")    # Select Channel 1
        oscilloscope.write("DATa:ENCdg ASCii")   # Set data encoding to ASCII
        oscilloscope.write("DATa:WIDth 1")       # Set data width to 1 byte per sample (if supported)
        
        # Set the record length (number of data points to capture)
        oscilloscope.write(f"HORizontal:RECOrdlength {RECORD_LENGTH}")

        # Specify the data range for transfer (points 1 to RECORD_LENGTH)
        oscilloscope.write("DATa:STARt 1")
        oscilloscope.write(f"DATa:STOP {RECORD_LENGTH}")

        # Query waveform scaling information
        x_increment = float(oscilloscope.query("WFMPre:XINcr?"))
        y_increment = float(oscilloscope.query("WFMPre:YMUlt?"))
        y_offset = float(oscilloscope.query("WFMPre:YOFf?"))
        y_zero = float(oscilloscope.query("WFMPre:YZEro?"))

        # Request binary waveform data from Channel 1
        raw_data = oscilloscope.query("CURVe?")
        raw_data = np.array(raw_data.split(','), dtype=float)

        # Convert binary data to voltages
        waveform_data = (raw_data - y_offset) * y_increment + y_zero

        # Generate the time axis
        time_data = np.arange(0, len(waveform_data) * x_increment, x_increment)

        return time_data, waveform_data

    except Exception as e:
        print(f"An error occurred while fetching waveform: {e}")
        return None, None

def save_to_csv(filename, time_data, ch1, ch2):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Ch1 (V)", "Ch2 (V)"])
            for t, v1, v2 in zip(time_data, ch1, ch2):
                writer.writerow([t, v1, v2])
        print(f"Waveform data saved to {filename}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

def main():
    try:
        # Open connection to the oscilloscope
        oscilloscope = rm.open_resource(VISA_ADDRESS)
        oscilloscope.timeout = 10000  # Set timeout to 10 seconds

        print("Monitoring for triggers... Press Ctrl+C to stop.")

        trigger_count = 0
        while True:
            # Check the oscilloscope trigger state
            trigger_state = oscilloscope.query("TRIGger:STATE?").strip()
            
            if trigger_state == "TRIGGER":
                # Increment trigger count for unique file naming
                trigger_count += 1
                filename = f"waveform_data_trigger_{trigger_count}.csv"

                # Fetch and save waveform data on trigger
                time_data, ch1 = fetch_waveform_data(oscilloscope, 1)
                time_data, ch2 = fetch_waveform_data(oscilloscope, 2)
                if time_data is not None and ch1 is not None:
                    save_to_csv(filename, time_data, ch1, ch2)

                # Wait for the next trigger
                print(f"Trigger {trigger_count} processed, waiting for the next trigger...")

    except KeyboardInterrupt:
        print("\nStopping monitoring due to user interruption.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the connection to the oscilloscope
        oscilloscope.close()

if __name__ == "__main__":
    main()
