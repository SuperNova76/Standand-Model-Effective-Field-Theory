version = 'phase0_eft'

pathStem = '/eos/user/n/nsangwen/' + version
outputPath = "../fitTrees/all/CMS_regions_mc16a/"

###############
#  TREE INFO  #
###############

treeName = ["nominal"]
weightsTreeName = "sumWeights"
truthTreeName = "truth"
particleTreeName = 'particleLevel'
LooseTreeName = ["nominal_Loose"]


##############
#  SETTINGS  #
##############

# job progress (the number of processed Events will be reported in multiples of reportEvery)
reportEvery = 10000
gridEff = 1.0
buildHashTables = False          # used for matching reco to truth
Nfolds = 5
bdt_train = False
bdt_eval = False  # This option is currently giving memory problems

RunLoose = False             # Switch between running over tight and loose for Data and MC
RunSysts = False
frac = 1           # controls what percent of the samples you run over
use_multiprocessing = True
Num_Threads = 8              # Number of threads used for the multiprocessing
chunk_size = 20000             # 100000 vv; 30000 rest
Calculate_asym = False          # Use only on ttW with BDT

#############
#  SAMPLES  #
#############
epochs = [
    "2016",
    # "2017",
    # "2018"
]


nom_tree = treeName
trees = treeName


signal = ['410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad',
          '410472.PhPy8EG_A14_ttbar_hdamp258p75_dil',
          '506210.MGPy8_ttbar_SMEFTsim_reweighted_nonallhad'
]
truth_matched = ['410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad',
                 '410472.PhPy8EG_A14_ttbar_hdamp258p75_dil']


channels = ['all']

Main_samples = [

    "data15",
    "data16",
    # "data17",
    # "data18",
    "tt_PhPy8_dilep",
    # "tt_PhPy8_nonhad",
    "Wt",
    "EFT",
]

Samples_out = Main_samples

lumi = {"2015":     3219.56,
        "2016":     32988 + 3219.56,
        "2017":     44307.4,
        "2018":     58450.1,
        }

Samples_in = {


    # TTBAR
    "tt_PhPy8_dilep":     ['410472.PhPy8EG_A14_ttbar_hdamp258p75_dil'],
    "tt_PhPy8_nonhad":    ['410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad'],

    'Wt':           ['410648.PowhegPythia8EvtGen_A14_Wt_DR_dilepton_top',
                     '410649.PowhegPythia8EvtGen_A14_Wt_DR_dilepton_antitop'],
    # '410646.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_top',
    # '410647.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_antitop'

    "EFT":  ['506210.MGPy8_ttbar_SMEFTsim_reweighted_nonallhad'],

    # DATA
    "data15":         ["data15_13TeV"],
    "data16":         ["data16_13TeV"],
    "data17":         ["data17_13TeV"],
    "data18":         ["data18_13TeV"],

}

yearpath = {"2016":     "mc16a",
            "2017":     "mc16d",
            "2018":     "mc16e",
            }


##########
#  CUTS  #
##########

nLeps = 2
minBjets = 2
maxBjets = 10
minJets = 0
maxJets = 100
ptl1cut = 25000
ptl2cut = 25000
ptl3cut = 25000
jetptCut = 20000

#############
#  REGIONS  #
#############

categories = [
              'ee',
              'emu',
              'mumu',
]

SR = categories[0]

###############
#  VARIABLES  #
###############

Weights = [
    'weight_mc', 'weight_pileup', 'weight_leptonSF', 'weight_bTagSF_DL1r_77', 'weight_jvt',
]


Observables = [
    "normWeight", "finalWeight", "weight_mc",

    "nJets", "nBTags", "MET", "HT", "m_ll", "m_llbb", "m_tt",

    "pt_lep0_pt", "pt_lep0_eta", "pt_lep0_phi", 'pt_lep0_E',
    "pt_lep1_pt", "pt_lep1_eta", "pt_lep1_phi", "pt_lep1_E", "delta_phi", "delta_eta",

    "b0_pt", "b0_eta", "b0_phi", 'b0_E', 'b0_DL1r', 'b0_tagweightbin',
    "b1_pt", "b1_eta", "b1_phi", 'b1_E', 'b1_DL1r', 'b1_tagweightbin',

    "Yt_0", "Yt_1", "Yt_2", "Yt_3",

]


EFT = [
    "ctGRe_m0p8", "ctGRe_p0p2", "Default", "ctGRe_m0p5", #"EFT_weights", "EFT_weights_names",
    "cQd1_m0p3", "cQd1_p0p2", "cQd1_p0p6",
]

fitVars = Observables + Weights + EFT

###############
#  X-SECTIONS #
###############

xSections = {

    'test_ttW': 1.0,
    'v8_test': 1.0,

    # TTW
    '410155.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttW':    0.54830*1.10,  # 410155
    '412123.MGPy8EG_A14_NNPDF23LO_EWttWsm': 0.057737*1.0,  # 412123

    'ttW_Sherpa228':    0.58922*1.0,  # 700000

    '700168.Sh_2210_ttW':    0.597*1,  # 700168
    '700205.Sh_2210_ttW_EWK':    0.042111*1.0,  # 700205

    '600793.PhPy8EG_A14NNPDF23_ttWm_EW':   0.016468*1.0,  # 600793
    '600794.PhPy8EG_A14NNPDF23_ttWp_EW':    0.032747*1.0,  # 600794
    '600795.PhPy8EG_A14NNPDF23_ttWm_QCD':   0.1864*1.0,  # 600795
    '600796.PhPy8EG_A14NNPDF23_ttWp_QCD':   0.37202*1.0,  # 600796

    # TTZ
    '410218.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee':    0.036888*1.12,  # 410218
    '410219.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttmumu':    0.036895*1.12,  # 410219
    '410220.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_tttautau':    0.036599*1.12,  # 410220
    '410276.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttee_mll_1_5':    0.0184*1.0,  # 410276
    '410277.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_ttmumu_mll_1_5':    0.0184*1.0,  # 410277
    '410278.aMcAtNloPythia8EvtGen_MEN30NLO_A14N23LO_tttautau_mll_1_5':    0.00197*1.0,  # 410278

    '504330.aMCPy8EG_NNPDF30NLO_A14N23LO_ttee': 0.036853*1.12,  # 504330
    '504334.aMCPy8EG_NNPDF30NLO_A14N23LO_ttmumu':   0.036854*1.12,  # 504334
    '504342.aMCPy8EG_NNPDF30NLO_A14N23LO_tttautau': 0.036676*1.12,  # 504342

    '700309.Sh_2211_ttll': 0.1451*1.31,  # 700309

    '413023.Sherpa_221_ttll_multileg_NLO':  0.13058*0.94731,  # 413023

    # TTBAR
    '410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad':    396.87*1.1398,  # 410470
    '410472.PhPy8EG_A14_ttbar_hdamp258p75_dil': 76.95*1.1398,  # 410472

    '700122.Sh_2210_ttbar_SingleLeptonP_maxHTavrgTopPT_SSC': 151.01*1.2075,  # 700122
    '700123.Sh_2210_ttbar_SingleLeptonM_maxHTavrgTopPT_SSC': 151.05*1.2075,  # 700123
    '700124.Sh_2210_ttbar_dilepton_maxHTavrgTopPT_SSC': 72.592*1.2075,  # 700124

    'tt_PhHw7_HIGGS':    77.00*1.1391,  # 410558
    'tt_PhHw704_SingleLep': 320.112*1.1392,  # 410557
    '411233.PowhegHerwig7EvtGen_tt_hdamp258p75_713_SingleLep': 320.1853*1.1392,  # 411233
    '411234.PowhegHerwig7EvtGen_tt_hdamp258p75_713_dil': 77.01622*1.1391,  # 411234
    '600666.PhH7EG_H7UE_tt_hdamp258p75_721_singlelep': 320.1*1.1392,  # 600666
    '600667.PhH7EG_H7UE_tt_hdamp258p75_721_dil': 76.93*1.139,  # 600667


    # TTH
    'ttH_dilep':    0.051245*1.1,  # 346443
    'ttH_semilep':    0.204960*1.1,  # 346444
    'ttH_allhad':    0.205100*1.1,  # 346445

    '346343.PhPy8EG_A14NNPDF23_NNPDF30ME_ttH125_allhad': 0.23082*1.0,  # 346343
    '346344.PhPy8EG_A14NNPDF23_NNPDF30ME_ttH125_semilep': 0.22276*1.0,  # 346344
    '346345.PhPy8EG_A14NNPDF23_NNPDF30ME_ttH125_dilep': 0.05343*1.0,  # 346345

    # VV
    '364250.Sherpa_222_NNPDF30NNLO_llll':    1.2523*1.0,  # 364250
    '364253.Sherpa_222_NNPDF30NNLO_lllv':    4.5832*1.0,  # 364253
    '364283.Sherpa_222_NNPDF30NNLO_lllljj_EW6':    0.010471*1.0,  # 364283
    '364284.Sherpa_222_NNPDF30NNLO_lllvjj_EW6':    0.046367*1.0,  # 364284
    '364288.Sherpa_222_NNPDF30NNLO_llll_lowMllPtComplement':    1.4318*1.0,  # 364288
    '364289.Sherpa_222_NNPDF30NNLO_lllv_lowMllPtComplement':    2.9152*1.0,  # 364289
    '345705.Sherpa_222_NNPDF30NNLO_ggllll_0M4l130':    0.0099486*1.0,  # 345705
    '345706.Sherpa_222_NNPDF30NNLO_ggllll_130M4l':    0.010091*1.0,  # 345706

    # TZ
    '412063.aMcAtNloPythia8EvtGen_tllq_NNPDF30_nf4_A14':    0.03327624*1.0,  # 412063

    # OTHER
    '412118.aMcAtNloPythia8EvtGen_tWZ_Ztoll_DR1':    0.016082*1.0,  # 412118
    '412119.aMcAtNloPythia8EvtGen_tWZ_Ztoll_DR2':    0.014697*1.0,  # 412119

    '410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW':    0.0080975*1.2231,  # 410081

    '342284.Pythia8EvtGen_A14NNPDF23LO_WH125_inc':    1.1021*1.2522,  # 342284
    '342285.Pythia8EvtGen_A14NNPDF23LO_ZH125_inc':    0.60072*1.4476,  # 342285
    '346524.PowhegPythia8EvtGen_NNPDF3_AZNLO_ggZH125J_MINLO_ZinclWWlvlv':   0.05744*1.0,  # 346524

    '346310.PowhegPythia8EvtGen_NNPDF30_AZNLO_ZH125J_Zincl_H_incl_MINLO':   0.76102*1.0,  # 346310
    '346311.PowhegPythia8EvtGen_NNPDF30_AZNLO_WpH125J_Wincl_H_incl_MINLO':  0.86164*1.0,  # 346311
    '346312.PowhegPythia8EvtGen_NNPDF30_AZNLO_WmH125J_Wincl_H_incl_MINLO':  0.53979*1.0,  # 346312

    '410648.PowhegPythia8EvtGen_A14_Wt_DR_dilepton_top':   3.9968*0.945,  # 410648
    '410649.PowhegPythia8EvtGen_A14_Wt_DR_dilepton_antitop':   3.9940*0.946,  # 410649
    '410646.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_top':   37.936*0.945,  # 410646
    '410647.PowhegPythia8EvtGen_A14_Wt_DR_inclusive_antitop':   37.906*0.946,  # 410647

    '364242.Sherpa_222_NNPDF30NNLO_WWW_3l3v_EW6':    0.0071931*1.0,  # 364242
    '364243.Sherpa_222_NNPDF30NNLO_WWZ_4l2v_EW6':    0.0017956*1.0,  # 364243
    '364244.Sherpa_222_NNPDF30NNLO_WWZ_2l4v_EW6':    0.0035429*1.0,  # 364244
    '364245.Sherpa_222_NNPDF30NNLO_WZZ_5l1v_EW6':    0.00018812*1.0,  # 364245
    '364246.Sherpa_222_NNPDF30NNLO_WZZ_3l3v_EW6':    0.000747635474*1.0,  # 364246
    '364247.Sherpa_222_NNPDF30NNLO_ZZZ_6l0v_EW6':    1.4458e-05*1.0,  # 364247
    '364248.Sherpa_222_NNPDF30NNLO_ZZZ_4l2v_EW6':    8.63731512e-05*1.0,  # 364248
    '364249.Sherpa_222_NNPDF30NNLO_ZZZ_2l4v_EW6':    0.00017197896*1.0,  # 364249

    '304014.MadGraphPythia8EvtGen_A14NNPDF23_3top_SM':    0.0016398*1.0,  # 304014
    'tttt':    0.010624*1.1267,  # 412043
    '700075.Sh_2210_tttt_muQHT2':   0.0118*1.01046,  # 700075
    '700355.Sh_2211_tttt_muQHT2':   0.011873*1.0,  # 700355

    # TTGAMMA
    '410389.MadGraphPythia8EvtGen_A14NNPDF23_ttgamma_nonallhadronic':    4.6242*1.16,  # 410389

    # ZJETS
    '700320.Sh_2211_Zee_maxHTpTV2_BFilter': 55.5413852*1.0,  # 700320
    '700321.Sh_2211_Zee_maxHTpTV2_CFilterBVeto': 286.392209*1.0,  # 700321
    '700322.Sh_2211_Zee_maxHTpTV2_CVetoBVeto': 1879.375291*1.0,  # 700322
    '700323.Sh_2211_Zmumu_maxHTpTV2_BFilter': 54.1530727*1.0,  # 700323
    '700324.Sh_2211_Zmumu_maxHTpTV2_CFilterBVeto': 287.347368*1.0,  # 700324
    '700325.Sh_2211_Zmumu_maxHTpTV2_CVetoBVeto': 1879.997255*1.0,  # 700325
    '700326.Sh_2211_Ztautau_LL_maxHTpTV2_BFilter': 6.68473707*1.0,  # 700326
    '700327.Sh_2211_Ztautau_LL_maxHTpTV2_CFilterBVeto': 34.5071089*1.0,  # 700327
    '700328.Sh_2211_Ztautau_LL_maxHTpTV2_CVetoBVeto': 234.1268655*1.0,  # 700328
    '700329.Sh_2211_Ztautau_LH_maxHTpTV2_BFilter': 24.803136*1.0,  # 700329
    '700330.Sh_2211_Ztautau_LH_maxHTpTV2_CFilterBVeto': 127.00462*1.0,  # 700330
    '700331.Sh_2211_Ztautau_LH_maxHTpTV2_CVetoBVeto': 861.442904*1.0,  # 700331
    '700332.Sh_2211_Ztautau_HH_maxHTpTV2_BFilter': 23.11928376*1.0,  # 700332
    '700333.Sh_2211_Ztautau_HH_maxHTpTV2_CFilterBVeto': 117.1817198*1.0,  # 700333
    '700334.Sh_2211_Ztautau_HH_maxHTpTV2_CVetoBVeto': 792.6878012*1.0,  # 700334

    '364198.Sherpa_221_NN30NNLO_Zmm_Mll10_40_MAXHTPTV0_70_BVeto':    2330.19*0.9751,  # 364198
    '364199.Sherpa_221_NN30NNLO_Zmm_Mll10_40_MAXHTPTV0_70_BFilter':    82.25676*0.9751,  # 364199
    '364200.Sherpa_221_NN30NNLO_Zmm_Mll10_40_MAXHTPTV70_280_BVeto':    44.87913*0.9751,  # 364200
    '364201.Sherpa_221_NN30NNLO_Zmm_Mll10_40_MAXHTPTV70_280_BFilter':    5.114990*0.9751,  # 364201
    '364202.Sherpa_221_NN30NNLO_Zmm_Mll10_40_MAXHTPTV280_E_CMS_BVeto':    2.759784*0.9751,  # 364202
    '364203.Sherpa_221_NN30NNLO_Zmm_Mll10_40_MAXHTPTV280_E_CMS_BFilter':    0.4721560*0.9751,  # 364203
    '364204.Sherpa_221_NN30NNLO_Zee_Mll10_40_MAXHTPTV0_70_BVeto':    2331.223*0.9751,  # 364204
    '364205.Sherpa_221_NN30NNLO_Zee_Mll10_40_MAXHTPTV0_70_BFilter':    81.35769*0.9751,  # 364205
    '364206.Sherpa_221_NN30NNLO_Zee_Mll10_40_MAXHTPTV70_280_BVeto':    44.97143*0.9751,  # 364206
    '364207.Sherpa_221_NN30NNLO_Zee_Mll10_40_MAXHTPTV70_280_BFilter':    5.481415*0.9751,  # 364207
    '364208.Sherpa_221_NN30NNLO_Zee_Mll10_40_MAXHTPTV280_E_CMS_BVeto':    2.777411*0.9751,  # 364208
    '364209.Sherpa_221_NN30NNLO_Zee_Mll10_40_MAXHTPTV280_E_CMS_BFilter':    0.4730864*0.9751,  # 364209
    '364210.Sherpa_221_NN30NNLO_Ztt_Mll10_40_MAXHTPTV0_70_BVeto':    2333.926*0.9751,  # 364210
    '364211.Sherpa_221_NN30NNLO_Ztt_Mll10_40_MAXHTPTV0_70_BFilter':    81.10263*0.9751,  # 364211
    '364212.Sherpa_221_NN30NNLO_Ztt_Mll10_40_MAXHTPTV70_280_BVeto':    44.83686*0.9751,  # 364212
    '364213.Sherpa_221_NN30NNLO_Ztt_Mll10_40_MAXHTPTV70_280_BFilter':    5.540944*0.9751,  # 364213
    '364214.Sherpa_221_NN30NNLO_Ztt_Mll10_40_MAXHTPTV280_E_CMS_BVeto':    2.793550*0.9751,  # 364214
    '364215.Sherpa_221_NN30NNLO_Ztt_Mll10_40_MAXHTPTV280_E_CMS_BFilter':    0.4697209*0.9751,  # 364215

    '361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee':    1901.2000*1.026,  # 361106
    '361107.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu':    1901.2000*1.026,  # 361107
    '361108.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Ztautau':    1901.2000*1.026,  # 361108
    '361665.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYee_10M60':   1764.13960*1.0,  # 361665
    '361667.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYmumu_10M60': 1812.27718*1.0,  # 361667
    '361669.PowhegPythia8EvtGen_AZNLOCTEQ6L1_DYtautau_10M60':   100.28565*1.0,  # 361669

    '346413.PhPy8EG_AZNLOCTEQ6L1_ZmumuWithInternalConversionFilter':   0.368313*1.0,  # 346413

    # VGAMMA
    '700398.Sh_2211_mumugamma':  105.04*0.939914,  # 700398
    '700399.Sh_2211_eegamma':  105.08*0.939914,  # 700399
    '700400.Sh_2211_tautaugamma':  107.72*0.939914,  # 700400

    '700011.Sh_228_eegamma_pty7_EnhMaxpTVpTy': 98.705*1.0,  # 700011
    '700012.Sh_228_mmgamma_pty7_EnhMaxpTVpTy': 98.609*1.0,  # 700012
    '700013.Sh_228_ttgamma_pty7_EnhMaxpTVpTy': 98.715*1.0,  # 700013
    '700014.Sh_228_vvgamma_pty7_EnhMaxpTVpTy': 57.023*1.0,  # 700014
    '700015.Sh_228_evgamma_pty7_EnhMaxpTVpTy': 356.31*1.0,  # 700015
    '700016.Sh_228_mvgamma_pty7_EnhMaxpTVpTy': 358.25*1.0,  # 700016
    '700017.Sh_228_tvgamma_pty7_EnhMaxpTVpTy': 356.36*1.0,  # 700017

    '366140.Sh_224_NN30NNLO_eegamma_LO_pty_7_15':    46.293*1.0,  # 366140
    '366141.Sh_224_NN30NNLO_eegamma_LO_pty_15_35':    29.281*1.0,  # 366141
    '366142.Sh_224_NN30NNLO_eegamma_LO_pty_35_70':    5.1577*1.0,  # 366142
    '366143.Sh_224_NN30NNLO_eegamma_LO_pty_70_140':    0.40363*1.0,  # 366143
    '366144.Sh_224_NN30NNLO_eegamma_LO_pty_140_E_CMS':    0.05299*1.0,  # 366144
    '366145.Sh_224_NN30NNLO_mumugamma_LO_pty_7_15':    46.276*1.0,  # 366145
    '366146.Sh_224_NN30NNLO_mumugamma_LO_pty_15_35':    29.28*1.0,  # 366146
    '366147.Sh_224_NN30NNLO_mumugamma_LO_pty_35_70':    5.1566*1.0,  # 366147
    '366148.Sh_224_NN30NNLO_mumugamma_LO_pty_70_140':    0.40322*1.0,  # 366148
    '366149.Sh_224_NN30NNLO_mumugamma_LO_pty_140_E_CMS':    0.05289*1.0,  # 366149
    '366150.Sh_224_NN30NNLO_tautaugamma_LO_pty_7_15':    46.253*1.0,  # 366150
    '366151.Sh_224_NN30NNLO_tautaugamma_LO_pty_15_35':    29.282*1.0,  # 366151
    '366152.Sh_224_NN30NNLO_tautaugamma_LO_pty_35_70':    5.1534*1.0,  # 366152
    '366153.Sh_224_NN30NNLO_tautaugamma_LO_pty_70_140':    0.40358*1.0,  # 366153
    '366154.Sh_224_NN30NNLO_tautaugamma_LO_pty_140_E_CMS':    0.052981*1.0,  # 366154

    '506210.MGPy8_ttbar_SMEFTsim_reweighted_nonallhad':    221.46 * 1.0
}
