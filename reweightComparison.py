import os, uproot
import mplhep as hep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools import logger, text
import awkward as ak

def extract_data(samplePath, samples, variables, weights, tree_name="nominal"):
    """
    Extracts and process data from selected root files

    samples:       list of root files to be used
    variables:     list of variables to extract (same as name in root file)
    yt_weights:    list of  weights to be used for reweighting [must include SM weight]
    tree_name:     tree used for reading information [default:    nominal]
    """

    logger.info('Reading in data from root files')

    filenames = []
    nominalWeight = 'Default'

    for sample in samples: # List of files
        filenames += [f'{samplePath}{sample}.root']

    for treename in ["nominal"]: # List of files with tree
        filenames_for_tree = {f: treename for f in filenames}

    # Open File
    try:    file = uproot.open(filenames_for_tree)
    except: logger.warning('Sample does not exist. Check sample names:')

    # Select branches of interest
    branches = variables + weights #+ ['finalWeight']
    data = file.arrays(branches)
    # print(data)

    # Calculate reweighting by dividing new weight by SM predicted one
    for w in weights:
        print('\n')
        print(w)
        print(data[w])
        data[w] = data[w] #/data[nominalWeight]
        print(data[w])

    logger.info('Returning necessary information')
    return data

def plot_histogram(data, variable, yt_weights, x_label, xrange, bins=25,nominalWeight="Default"):
    """
    Comparison plot of reweighted histograms with ratio plot to the nominal value.

    data:           Uproot object with necessary branches
    variable:       variable to be plotted
    yt_weights:     list of weights to be used for reweighting
    xrange:         Tuple containing the x-axis range for each hist
    bins:           Number of bins per histogram
    """

    logger.info(f"Creating {variable} comparison plot")

    points, bins = np.histogram(data[variable], bins=bins, weights=data[nominalWeight]#*data['finalWeight']
                                , range=xrange)
    center = (bins[:-1] + bins[1:]) / 2
    print(nominalWeight)
    print(points)
    ratios_dict = {}

    for weight in yt_weights:
        print('\n'+weight)
        mid_points, bins = np.histogram(data[variable], bins=bins, weights=data[weight]#*data['finalWeight']
                                        , range=xrange)
        ratios_dict[weight] = [mid_points[i]/points[i] for i in range(len(points))]
        print(mid_points)

    print('\n')
    print(ratios_dict)
    hep.style.use(hep.style.ATLAS)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 10))
    hep.atlas.label(ax = ax1, loc=1, data=False)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    max_bin_height = 0

    for i in range(len(yt_weights)):
        # label = '$Y_{t}$ = ' + f'{yt_weights[i].split("_")[1]}'
        label = f'{yt_weights[i]}'
        # print(label)
        n, bins, patches = ax1.hist(data[variable], bins=bins, weights=data[yt_weights[i]]#*data['finalWeight']
                                    , range=xrange, histtype="step", label=label, color=colors[i])
        # if yt_weights[i] != nominalWeight:
        ax2.plot(center, ratios_dict[yt_weights[i]], color=colors[i], label=label, marker='+', linestyle='')
        if max(n) > max_bin_height:
            max_bin_height = max(n)

    ax1.set_xlim(xrange)
    ax1.set_ylim(0, max_bin_height*1.2)
    ax1.set_position([0.15, 0.4, 0.7, 0.5])
    ax1.set_xticks([])
    ax1.set_ylabel("Entries")

    ax2.plot(np.linspace(0, 2000, 10), np.linspace(1, 1, 10), linestyle='--', color='black')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("$(C_i = X)/(C_i = 0)$")
    ax2.set_position([0.15, 0.2, 0.7, 0.15])
    ax2.set_xlim(xrange)

    handles, labels = ax1.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

    ax1.legend(handles=new_handles, labels=labels, loc=1)

    logger.info(f'Saving to . . .  figures/Yt_ratio_{variable}_inj.png')#
    plt.savefig(f'figures/Yt_ratio_{variable}_inj.png')

def Info_Text(samplePath,variables,x_labels,weights):
    """
    Function takes in the settings used for the script and outputs an
    aesthetically appealing terminal output :)
    """
    print('\n{:<10s}\n'.format(text.green +
          text.HEADER + 'SETTINGS    ' + text.end))

    if not os.path.isdir(samplePath):
        print('{:<10s}{:>4s}\n'.format(
            text.fail + '[Fail] Invalid Input Directory:     ' + text.end, samplePath))
        exit()

    print('{:<10s}{:>4s}'.format(text.green + text.bold +
          'Ntuple path:    ' + text.end, samplePath))
    print('\n{:<29s}{:>4s}'.format(text.green + text.bold +
          'Variables:      ' + text.end, '      '.join(variables)))
    print('{:<29s}{:>4s}'.format(text.green + text.bold +
          'xLabels:      ' + text.end, '      '.join(x_labels)))
    print('{:<29s}{:>4s}'.format(text.green + text.bold +
          'Weights:      ' + text.end, '      '.join(weights)))

    print('\n')



if __name__ == "__main__":

    logger.info("Script for comparing effects of reweighting")
    samplePath = '/eos/user/n/nsangwen/ttbar_dilep_yukawa/fitTrees/all/CMS_regions_mc16a/emu/'
    samples = ['EFT']
    variables = ['m_llbb', 'm_ll']
    x_labels =  ["$m_{llbb}$ [GeV]","$m_{ll}$ [GeV]"]
    ranges=[(100,750), (10,750)]
    # yt_weights = ['Default','ctGRe_m0p8', 'ctGRe_p0p2', 'ctGRe_m0p5']
    yt_weights = ['Default', "cQd1_m0p3", 
                  "cQd1_p0p2", "cQd1_p0p6"
                  ]

    # yt_weights = ['Yt_08','Yt_09', 'Yt_1', 'Yt_11', 'Yt_12']

    Info_Text(samplePath,variables,x_labels,yt_weights)

    data = extract_data(samplePath, samples, variables, yt_weights)


    for i in range(len(variables)):
        plot_histogram(data, variables[i], yt_weights, x_labels[i], ranges[i], bins=25)
