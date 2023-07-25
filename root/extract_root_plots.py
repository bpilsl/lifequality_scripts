import glob
import os.path
import sys
import ROOT

# thanks @ ChatGPT for the comments ;)

# Get the input path from the command-line arguments
pathIn = sys.argv[1]

# Find all files ending with '.root' in the specified input path
files = glob.glob(pathIn + '*.root')
print('found files: ', files)

# Get the output path from the command-line arguments
pathOut = sys.argv[2]

# Key representing the object to extract from ROOT files
key = 'AnalysisEfficiency/RD50_MPW3_base_0/efficiencyVsTime'

# Output file extension (default: 'png')
fileExtOutput = 'png'
# Uncomment the lines below to allow a different file extension from command-line arguments
# if sys.argv[3]:
#     fileExtOutput = sys.argv[3]

# Create a TCanvas for plotting
c1 = ROOT.TCanvas("c1", "", 10, 10, 1100, 700)
c1.SetRightMargin(0.2)

# Loop over each ROOT file found
for f in files:
    # Create the output file name based on the input file name and key
    name = os.path.basename(f) + '_' + key.replace('/', '_') + '.'
    print('printing to ', name)

    # Open the current ROOT file
    r = ROOT.TFile(f)

    # Get the object specified by the key from the ROOT file
    obj = r.Get(key)

    # Clone the object to avoid potential side effects on the original object
    if obj:
        h1 = obj.Clone()
    else:
        print((f'WARNING, file {f} does not contain key {key}'))

    # Set the directory of the cloned object to the global ROOT directory
    h1.SetDirectory(ROOT.gROOT)

    # Clear the canvas before drawing the next plot
    c1.Clear()
    c1.cd()
    c1.SetName(name)
    c1.SetTitle(name)
    h1.SetTitle(name)

    # Draw the histogram on the canvas
    h1.Draw("")

    # Update the canvas and save it to the output path with the specified file extension
    ROOT.gPad.Modified()
    ROOT.gPad.Update()
    c1.Print(pathOut + name + fileExtOutput, fileExtOutput)
