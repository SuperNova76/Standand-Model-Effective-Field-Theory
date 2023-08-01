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

            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_pt"], bjets[0].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_eta"], bjets[0].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_phi"], bjets[0].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet0_E"], bjets[0].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_pt"], bjets[1].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_eta"], bjets[1].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_phi"], bjets[1].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["bjet1_E"], bjets[1].E)

            chunkDict[channel][config.categories[cat]][sample][run_tree]["njets"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["njets"], len(Jets))
            chunkDict[channel][config.categories[cat]][sample][run_tree]["nBjets"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["nBjets"], len(bjets))
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Ht"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Ht"], ht)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Mbb"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Mbb"], m_llbb)

            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_pt"], pt_ordered_leps[0].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_eta"], pt_ordered_leps[0].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_phi"], pt_ordered_leps[0].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_E"], pt_ordered_leps[0].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_ID"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_ID"], pt_ordered_leps[0].ID)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_ptvarcone30"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep0_ptvarcone30"], pt_ordered_leps[0].ptvarcone30)

            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_pt"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_pt"], pt_ordered_leps[1].pt)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_eta"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_eta"], pt_ordered_leps[1].eta)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_phi"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_phi"], pt_ordered_leps[1].phi)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_E"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_E"], pt_ordered_leps[1].E)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_ID"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_ID"], pt_ordered_leps[1].ID)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_ptvarcone30"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["Lep1_ptvarcone30"], pt_ordered_leps[1].ptvarcone30)

            # Preparing tree
            chunkDict[channel][config.categories[cat]][sample][run_tree]["T"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["T"], 0)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["T"].dtype = "float64"

            # Preparing the weightTree to fill the tree as well
            weightsTree.GetEntry(iEvent)
            chunkDict[channel][config.categories[cat]][sample][run_tree]["weightsTree"] = np.append(
                chunkDict[channel][config.categories[cat]][sample][run_tree]["weightsTree"], weightsTree.weight_mc)
        else:
            continue

    print("nEvents processed = " + str(iEvent+1) +
          "/" + str(tree.GetEntries()) + " (100%)")

    return chunkDict


if __name__ == '__main__':

    logger.level = logger.info
    parser = ArgumentParser(
        description="Top Physics analysis main program")
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=False, help="increase output verbosity")

    start_time = time.time()

    # Setting up argument parser
    parser = ArgumentParser(
        description="Run Top Physics analysis")

    parser.add_argument("-c", "--cprofiler", action="store_true",
                        default=False, help="run cProfiler")

    parser.add_argument("-s", "--systematic", action="store_true",
                        default=False, help="run with systematic variation")
    parser.add_argument("-b", "--book", action="store_true",
                        default=False, help="run the bookkeeping code")
    parser.add_argument("-m", "--mc", action="store_true",
                        default=False, help="run the montecarlo code")
    parser.add_argument("-d", "--data", action="store_true",
                        default=False, help="run the data code")
    parser.add_argument("-p", "--preselect", action="store_true",
                        default=False, help="run the preselect code")
    parser.add_argument("-j", "--jetbinning", action="store_true",
                        default=False, help="run the jetbinning code")
    parser.add_argument("-t", "--template", action="store_true",
                        default=False, help="run the template code")
    parser.add_argument("-n", "--nice", action="store_true",
                        default=False, help="run the nice plots code")

    parser.add_argument("-y", "--year", type=int, default=2018,
                        help="year to be analysed. (default: %(default)s)")
    parser.add_argument("-l", "--list", default="",
                        help="list of samples to be analysed, comma-separated (default: %(default)s)")
    parser.add_argument("-u", "--file", default="",
                        help="list of samples to be analysed, comma-separated (default: %(default)s)")
    parser.add_argument("-o", "--output", default="",
                        help="output file (default: %(default)s)")

    parser.add_argument("-r", "--runScript", default="runJetTool",
                        help="the executable to be ran (default: %(default)s)")
    parser.add_argument("-e", "--entryMethod", default="runJetTool",
                        help="the entry point to be ran (default: %(default)s)")

    parser.add_argument("--recursive", default=False,
                        action="store_true", help="use recursive")

    args = parser.parse_args()
    print(args)
    year = args.year

    if args.recursive:
        cmd = "find /atlas/stage_out/user/fredrikdsk/ | grep root | grep Top"
        files = os.popen(cmd).readlines()
        list_of_files = ""
        for file in files:
            list_of_files += file.replace("\n", "") + ","
        print(list_of_files[:-1])

    if args.list == "":
        sys.exit("Please provide a list of samples to analyze with -l option")

    # Set the code for each type of analysis
    code = "main"

    samples = [sample.strip() for sample in args.list.split(",")]
    if args.file != "":
        files = [file.strip() for file in args.file.split(",")]
    else:
        files = ["" for sample in samples]

    if args.nice:
        code = "nice"

    if args.systematic:
        code = "systematic"

    if args.book:
        code = "book"

    if args.mc:
        code = "mc"

    if args.data:
        code = "data"

    if args.preselect:
        code = "preselect"

    if args.jetbinning:
        code = "jetbinning"

    if args.template:
        code = "template"

    codeDir = config.codeDir
    mainFile = codeDir + "/" + code + ".py"

    # If this is the main program, we want to compile it
    if args.entryMethod == "main":
        compile_command = "python " + mainFile + " -c"

        if args.cprofiler:
            compile_command += " -p"

        # os.system(compile_command)
        # print(compile_command)

    mainFile = codeDir + "/" + code + ".py"
    print("Main file: ", mainFile)

    entries = []

    for s, f in zip(samples, files):
        sample = s
        sub_sample = ""
        file = f
        run_tree = "Nominal"

        if config.type == "mc":
            year_sample = config.yearpath[year] + "/" + sample
            if year_sample not in config.mc_samples:
                continue

        if config.type == "data":
            year_sample = config.yearpath[year] + "/" + sample
            if year_sample not in config.data_samples:
                continue

        print("Sample, DSID, Sub_sample, File, Type: " +
              sample + ", " + sub_sample + ", " + file + ", " + config.type)

        # Open root file
        path = config.pathStem
        filename = path + "/" + \
            config.yearpath[year] + "/" + sub_sample + "/" + file
        print("Opening " + filename + "...")

        file = TFile(filename)
        tree = file.Get(run_tree)
        nEvents = tree.GetEntries()
        file.Close()

        if nEvents == 0:
            print("Skipping " + filename + " - no events.")
            continue

        nCpu = multiprocessing.cpu_count()
        nCpu -= 2
        if nCpu > 8:
            nCpu = 8

        if nCpu < 1:
            nCpu = 1

        if nEvents < nCpu:
            nCpu = nEvents

        print("Running with ", nCpu, " processes.")

        jobs = []
        nEventsPerCore = int(nEvents/nCpu)
        runningWeightSum = 0
        runningEventSum = 0
        for i in range(nCpu):
            jobs.append([])

        for iEvent in range(nEvents):
            if (iEvent % config.reportEvery == 0):
                print("Processed ", iEvent, " out of ", nEvents)
            tree.GetEntry(iEvent)

            if config.type == "mc":
                weight = (
                    config.lumi[year]*config.xSections[sub_sample]*tree.weight_mc)/(runningWeightSum+1)
                runningWeightSum += tree.weight_mc
                runningEventSum += 1
                weightsTree.GetEntry(iEvent)
                finalWeight = weight*weightsTree.weight_mc
            else:
                weight = 1
                finalWeight = 1.0

            jobs[iEvent % nCpu].append((run_tree, iEvent, iEvent+1, sample, sub_sample, file, config.type,
                                       year, runningWeightSum, runningEventSum))
            # for i in range(nCpu):
            #     print('Running ',len(jobs[i]),' events in process ',i,' of ',nCpu)

        pool = multiprocessing.Pool(processes=nCpu)
        result = pool.map(MainFunc, jobs)
        pool.close()
        pool.join()

        finalDict = result[0]
        for i in range(1, len(result)):
            for channel in finalDict:
                for category in finalDict[channel]:
                    for sample in finalDict[channel][category]:
                        for run_tree in finalDict[channel][category][sample]:
                            for key in finalDict[channel][category][sample][run_tree]:
                                finalDict[channel][category][sample][run_tree][key] = np.append(
                                    finalDict[channel][category][sample][run_tree][key], result[i][channel][category][sample][run_tree][key])

        # write out the final output file
        if args.output != "":
            outputFileName = args.output
        else:
            outputFileName = config.outputDir + "/" + code + "_" + \
                year_sample + ".root"

        outputFilePath = os.path.dirname(outputFileName)
        os.system("mkdir -p " + outputFilePath)

        outFile = TFile(outputFileName, "RECREATE")
        outFile.cd()

        for channel in finalDict:
            for category in finalDict[channel]:
                for sample in finalDict[channel][category]:
                    for run_tree in finalDict[channel][category][sample]:
                        for key in finalDict[channel][category][sample][run_tree]:
                            branchName = channel + "_" + category + "_" + \
                                sample + "_" + run_tree + "_" + key
                            finalDict[channel][category][sample][run_tree][key].dtype = "float64"
                            outTree = TTree(branchName, branchName)
                            outTree.Branch(branchName, finalDict[channel][category][sample][run_tree]
                                           [key], branchName+"/D")

        outFile.Write()
        outFile.Close()

    print("Total time: ", timedelta(seconds=(time.time()-start_time)))
