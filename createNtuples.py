import os
import time
import multiprocessing
import sys
from datetime import timedelta
import numpy as np
from argparse import ArgumentParser
import ROOT
from ROOT import TFile, TTree

import config_Yukawa as config
import tools as tools
from tools import text, logger

import cProfile


def MainFunc(sample_entries):

    run_tree = sample_entries[0]
    start = sample_entries[1]
    end = sample_entries[2]
    sample = sample_entries[3]
    sub_sample = sample_entries[4]
    file = sample_entries[5]
    type = sample_entries[6]
    year = sample_entries[7]
    runningWeightSum = sample_entries[8]
    runningEventSum = sample_entries[9]

    chunkDict = {}
    chunkDict = tools.createFitVarsDict(config)

    path = config.pathStem
    filename = path + "/" + \
        config.yearpath[year] + "/" + sub_sample + "/" + file
    file = TFile(filename)

    tree = file.Get(run_tree)
    weightsTree = file.Get(config.weightsTreeName)
    finalWeight = 1.0

    if (("data" not in sub_sample) and (config.buildHashTables == True) and (config.truth_matched.count(sub_sample) > 0)):
        truthTree = file.Get(config.truthTreeName)
        # print("Building hash table ...")
        if (truthTree.GetEntries() >= 0):
            truthTree.BuildIndex("runNumber", "eventNumber")
            print("Truth hash table successfully built, with " +
                  str(truthTree.GetEntries()) + " entries")
        else:
            print("WARNING: Truth hash table unsuccesfull for file: %s" % filename)

    for iEvent in range(start, int(end)):
        tree.GetEntry(iEvent)
        if (iEvent == start):
            print('\n{:<29s}{:>4s}'.format(text.blue + text.bold + 'Campaign, Sample, DSID, Tree:      ' +
                  text.end, '      '.join([config.yearpath[year], sample, sub_sample, run_tree])))

        if (iEvent % config.reportEvery == 0):
            print("     nEvents processed = " + str(iEvent) + "/" + str(tree.GetEntries()
                                                                        ) + " (" + str(float(iEvent*100)/float(tree.GetEntries())) + "%)")
        tree.GetEntry(iEvent)

        if type == "mc":
            normWeight = (
                config.lumi[year]*config.xSections[sub_sample]*tree.weight_mc)/(runningWeightSum)
            calibrationWeight = tree.weight_pileup*tree.weight_leptonSF*tree.weight_jvt * \
                tree.weight_bTagSF_DL1r_77  # tree.weight_globalLeptonTriggerSF*
            finalWeight = normWeight*calibrationWeight
        else:
            normWeight = 1
            finalWeight = 1.0

        # print("\n============")
        # print(sub_sample)
        # if type=='mc':
        #     print(f'lumi,xsec,mc &wSum:  {config.lumi[year]},{config.xSections[sub_sample]},{tree.weight_mc},{runningWeightSum}')

        Jets, Filler_jets = tools.CreateJets(tree)
        recon_leptons = tools.CreateReconLeps(tree, sub_sample)

        totC = tools.ChargeSum(recon_leptons)
        Jets = [j for j in Jets if j.pt > config.jetptCut]
        bjets = [j for j in Jets if j.Bjet == True]
        allLepPts = [l.pt for l in recon_leptons]
        allLepPts = sorted(allLepPts, reverse=True)
        lepID = [abs(l.ID) for l in recon_leptons]

        # =========================
        #   ___ _   _ _____ ___  #
        #  / __| | | |_   _/ __| #
        # | (__| |_| | | | \__ \ #
        #  \___|\___/  |_| |___/ #
        ##########################

        channel = "all"

        if (totC != 0):
            continue

        if (len(recon_leptons) != config.nLeps):
            continue

        # if (lepID[0] == lepID[1]):
        #     continue

        if (allLepPts[0] < config.ptl1cut) or (allLepPts[1] < config.ptl2cut):
            continue

        if (len(bjets) < config.minBjets) or (len(bjets) > config.maxBjets):
            continue

        if (len(Jets) < config.minJets) or (len(Jets) > config.maxJets):
            continue

        if (lepID[0] == lepID[1]):
            if tree.met_met/1000 < 30:
                continue

        # ==============================================================
        #   ___   _   _    ___ _   _ _      _ _____ ___ ___  _  _ ___ #
        #  / __| /_\ | |  / __| | | | |    /_\_   _|_ _/ _ \| \| / __|#
        # | (__ / _ \| |_| (__| |_| | |__ / _ \| |  | | (_) | .` \__ \#
        #  \___/_/ \_\____\___|\___/|____/_/ \_\_| |___\___/|_|\_|___/#
        ###############################################################

        pt_ordered_leps = sorted(
            recon_leptons, key=lambda x: x.pt, reverse=True)
        m_ll = (pt_ordered_leps[0].vec+pt_ordered_leps[1].vec).mass()/1000

        if (lepID[0] == lepID[1]):
            if (81. < m_ll < 101.) or (m_ll < 50):
                continue

        # print(lepID)

        ht = tools.hT(Jets)
        pt_ordered_leps = sorted(
            recon_leptons, key=lambda x: x.pt, reverse=True)

        m_ll = (pt_ordered_leps[0].vec+pt_ordered_leps[1].vec).mass()/1000
        m_llbb = (pt_ordered_leps[0].vec+pt_ordered_leps[1].vec +
                  bjets[0].vec+bjets[1].vec).mass()/1000

        # print(m_ll,m_llbb)

        ###################################################################
        ###### CHECKING MOTHER, ONLY HAPPENS FOR TRUTH MATCHED EVENTS #####
        ###################################################################
        nonTruthMatched = 0

        # print('\nEvent %i'%iEvent)
        if (config.truth_matched.count(sub_sample) > 0) and (config.buildHashTables == True) and (truthTree.GetEntryWithIndex(tree.runNumber, tree.eventNumber) > 0):
            truth_leptons = tools.CreateTruthLeps(truthTree)
            tvec, tbvec, parton1, parton2 = tools.ExtractTruthTops(truthTree)
            parton1, parton2 = parton1[0], parton2[0]
            m_tt = (tvec+tbvec).mass()/1000

            if abs(parton1) != abs(parton2):
                parton = 21
            else:
                parton = parton1
            Yt_weights = [tools.yt_reweight(
                tvec, tbvec, parton, yt) for yt in range(4)]
        else:
            Yt_weights = [1, 1, 1, 1]
            m_tt = 1

        electrons = [lep for lep in recon_leptons if abs(lep.ID) == 11]
        muons = [lep for lep in recon_leptons if abs(lep.ID) == 13]
        # print(len(electrons),len(muons))
        # ====================================================#
        # _  _ ___ ___ _____ ___   ___ ___    _   __  __ ___ #
        # | || |_ _/ __|_   _/ _ \ / __| _ \  /_\ |  \/  / __|#
        # | __ || |\__ \ | || (_) | (_ |   / / _ \| |\/| \__ \#
        # |_||_|___|___/ |_| \___/ \___|_|_\/_/ \_\_|  |_|___/#
        ######################################################

        # cat = 0

        cat = tools.checkCategory(len(electrons), len(
            muons), 0, len(Jets), len(bjets), config.categories)

        if (cat != -1):

            chunkDict[channel][config.categories[cat]][sample][run_tree]["normWeight"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["normWeight"], normWeight)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["finalWeight"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["finalWeight"], finalWeight)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["nJets"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["nJets"], len(Jets))
            chunkDict[channel][config.categories[cat]][sample][run_tree]["nBTags"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["nBTags"], len(bjets))
            chunkDict[channel][config.categories[cat]][sample][run_tree]["MET"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["MET"], tree.met_met)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["HT"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["HT"], ht)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["m_ll"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["m_ll"], m_ll)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["m_llbb"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["m_llbb"], m_llbb)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["m_tt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["m_tt"], m_tt)
            # chunkDict[channel][config.categories[cat]][sample][run_tree]["nZcands"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["nZcands"],nZcands)

            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_pt"], pt_ordered_leps[0].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_eta"], pt_ordered_leps[0].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_phi"], pt_ordered_leps[0].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep0_E"], pt_ordered_leps[0].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_pt"], pt_ordered_leps[1].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_eta"], pt_ordered_leps[1].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_phi"], pt_ordered_leps[1].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["pt_lep1_E"], pt_ordered_leps[1].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["delta_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["delta_phi"], tools.deltaPhi(pt_ordered_leps[1].phi - pt_ordered_leps[0].phi))
            chunkDict[channel][config.categories[cat]][sample][run_tree]["delta_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["delta_eta"], abs(pt_ordered_leps[1].eta - pt_ordered_leps[0].eta))

            chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_pt"], bjets[0].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_eta"], bjets[0].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_phi"], bjets[0].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_E"], bjets[0].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_DL1r"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_DL1r"], bjets[0].DL1r)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_tagweightbin"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b0_tagweightbin"], bjets[0].tagweightbin)

            chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_pt"], bjets[1].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_eta"], bjets[1].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_phi"], bjets[0].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_E"], bjets[1].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_DL1r"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_DL1r"], bjets[1].DL1r)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_tagweightbin"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["b1_tagweightbin"], bjets[1].tagweightbin)

            chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_0"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_0"], Yt_weights[0])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_1"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_1"], Yt_weights[1])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_2"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_2"], Yt_weights[2])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_3"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Yt_3"], Yt_weights[3])
            
            # weightsTree.GetEntry(iEvent)
            # for i in range(len(tree.mc_generator_weights)):
                # chunkDict[channel][config.categories[cat]][sample][run_tree]["EFT_weights"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["EFT_weights"], tree.mc_generator_weights[i])
            # chunkDict[channel][config.categories[cat]][sample][run_tree]["EFT_weights_names"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["EFT_weights_names"], weightsTree.names_mc_generator_weights)
            # chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_m0p8"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_m0p8"], var)
            # chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_p0p2"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_p0p2"], var)
                
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Default"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["Default"], tree.mc_generator_weights[0])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_m0p8"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_m0p8"], tree.mc_generator_weights[1])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_p0p2"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_p0p2"], tree.mc_generator_weights[223])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_m0p5"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["ctGRe_m0p5"], tree.mc_generator_weights[112])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["cQd1_m0p3"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["cQd1_m0p3"], tree.mc_generator_weights[2])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["cQd1_p0p2"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["cQd1_p0p2"], tree.mc_generator_weights[13])
            chunkDict[channel][config.categories[cat]][sample][run_tree]["cQd1_p0p6"] = np.append(chunkDict[channel][config.categories[cat]][sample][run_tree]["cQd1_p0p6"], tree.mc_generator_weights[24])

            if 'data' not in sample:
                for var in config.Weights:
                    chunkDict[channel][config.categories[cat]][sample][run_tree][var] = np.append(
                        chunkDict[channel][config.categories[cat]][sample][run_tree][var], getattr(tree, var))

            else:
                for var in config.Weights:
                    chunkDict[channel][config.categories[cat]][sample][run_tree][var] = np.append(
                        chunkDict[channel][config.categories[cat]][sample][run_tree][var], 1.0)
    file.Close()
    return chunkDict


if __name__ == "__main__":
    Dict = []
    channel = "all"

    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--samples",
        nargs="*",
        help="Supply list of samples. Use naming convention found in ttw_config.Main_samples",
    )
    parser.add_argument(
        "-y",
        "--years",
        nargs="*",
        help="specify which year you want to run",
    )
    args = parser.parse_args()

    if args.samples:
        for sample in args.samples:
            # Check if the sample is known. Only checks the 2016 labels for samples
            if sample not in config.Samples_in:
                logger.error(
                    "Provided unknown sample: " + sample + "\nValid samples are " +
                    str(list(config.Samples_in.keys())),
                )
                sys.exit()
        config.Samples_out = args.samples

    if args.years:
        years = []
        for year in args.years:
            if year not in ["2016", "2017", "2018"]:
                logger.error("Invalid year: " + year)
                continue
            years.append(year)
        if years == []:
            logger.warning("No valid years provided, using default.")
        else:
            config.epochs = years

    tools.Info_Text()

    Dict = []

    start_time = time.time()
    print("")

    for year in config.epochs:
        print(text.green + text.bold + '\nProcessing ' +
              year + '...\n' + text.end)

        Samples = config.Samples_out
        sample_entries, progress = tools.split_entries(
            Samples, config.chunk_size, year)
        # sys.exit()

        if (not hasattr(config, "use_multiprocessing")) or config.use_multiprocessing:
            print(text.HEADER + '\nEntering multiprocessing ' + text.end)
            pool = multiprocessing.Pool(config.Num_Threads)
            sampleDicts = pool.map(MainFunc, sample_entries)
            pool.close()
        else:
            sampleDicts = list(map(MainFunc, sample_entries))
        Dict.append(sampleDicts)

    t_time_r = round(time.time() - start_time, 10)
    print('\n{:<29s}{:>4s}'.format(text.green + text.bold +
          'Time taken (reading):    ' + text.end, str(timedelta(seconds=t_time_r))))

    fitvarsDict = {}
    fitvarsDict = tools.createFitVarsDict(config)

    print(text.green + text.bold + '\nCreating root files ' + '...\n' + text.end)
    start_time = time.time()

    for sample in Samples:
        for t in config.trees:
            for sampleDict in Dict:
                for dict in sampleDict:
                    for cat in dict[channel]:
                        for var, value in dict[channel][cat][sample][t].items():
                            if var == 'mc_generator_weights':
                                for weight_vector in value:
                                    # print(weight_vector)
                                    fitvarsDict[channel][cat][sample][t][var].append(
                                        weight_vector)
                            else:
                                fitvarsDict[channel][cat][sample][t][var] = np.append(
                                    fitvarsDict[channel][cat][sample][t][var], value)

        for category in config.categories:
            tools.createFitTree(
                sample, channel, category, fitvarsDict[channel][category][sample], config.outputPath)

    t_time = round(time.time() - start_time, 10)
    print('{:<29s}{:>4s}'.format(text.green + text.bold +
          'Time taken (writing):    ' + text.end, str(t_time)))
    print('{:<29s}{:>4s}'.format(text.green + text.bold +
          'Time taken (reading):    ' + text.end, str(t_time_r)))
    print('{:<29s}{:>4s}'.format(text.green + text.bold +
          'Total time taken:    ' + text.end, str(t_time_r + t_time)))
