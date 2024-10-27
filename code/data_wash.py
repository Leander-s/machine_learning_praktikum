import pandas as pd
import numpy as np
from rdkit import Chem
import os
import rdkit
import copy
import warnings
warnings.filterwarnings('ignore')

# the y columns for different data sets
tasks_dic = {'bace': ['Class'], 'bbbp': ['p_np'], 'esol': ['measured log solubility in mols per litre'],
             'freesolv': ['expt'],
             'hiv': ['HIV_active'], 'lipophilicity': ['exp'],
             'clintox': ['FDA_APPROVED', 'CT_TOX'],
             'sider': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders',
                       'Product issues', 'Eye disorders', 'Investigations',
                       'Musculoskeletal and connective tissue disorders',
                       'Gastrointestinal disorders', 'Social circumstances',
                       'Immune system disorders', 'Reproductive system and breast disorders',
                       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                       'General disorders and administration site conditions',
                       'Endocrine disorders', 'Surgical and medical procedures',
                       'Vascular disorders', 'Blood and lymphatic system disorders',
                       'Skin and subcutaneous tissue disorders',
                       'Congenital, familial and genetic disorders',
                       'Infections and infestations',
                       'Respiratory, thoracic and mediastinal disorders',
                       'Psychiatric disorders', 'Renal and urinary disorders',
                       'Pregnancy, puerperium and perinatal conditions',
                       'Ear and labyrinth disorders', 'Cardiac disorders',
                       'Nervous system disorders',
                       'Injury, poisoning and procedural complications'],
             'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                       'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
             'muv': [
                 "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                 "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
             ],
             'toxcast': [
                 'ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive', 'APR_HepG2_CellCycleArrest_24h_dn',
                 'APR_HepG2_CellCycleArrest_24h_up', 'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                 'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn', 'APR_HepG2_MicrotubuleCSK_24h_up',
                 'APR_HepG2_MicrotubuleCSK_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn',
                 'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn', 'APR_HepG2_MitoMass_72h_up',
                 'APR_HepG2_MitoMembPot_1h_dn', 'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                 'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up', 'APR_HepG2_NuclearSize_24h_dn',
                 'APR_HepG2_NuclearSize_72h_dn', 'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
                 'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up', 'APR_HepG2_StressKinase_24h_up',
                 'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up',
                 'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up', 'APR_Hepat_CellLoss_24hr_dn',
                 'APR_Hepat_CellLoss_48hr_dn', 'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up',
                 'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up', 'APR_Hepat_MitoFxnI_1hr_dn',
                 'APR_Hepat_MitoFxnI_24hr_dn', 'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn',
                 'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up', 'APR_Hepat_Steatosis_48hr_up',
                 'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up', 'ATG_AP_2_CIS_dn', 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn',
                 'ATG_AR_TRANS_up', 'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up',
                 'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_dn',
                 'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up', 'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up',
                 'ATG_DR5_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up', 'ATG_EGR_CIS_up',
                 'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn', 'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up',
                 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up',
                 'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up', 'ATG_FoxO_CIS_dn', 'ATG_FoxO_CIS_up',
                 'ATG_GAL4_TRANS_dn', 'ATG_GATA_CIS_dn', 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up',
                 'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up', 'ATG_HIF1a_CIS_dn',
                 'ATG_HIF1a_CIS_up', 'ATG_HNF4a_TRANS_dn', 'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up',
                 'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up', 'ATG_ISRE_CIS_dn',
                 'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn', 'ATG_LXRa_TRANS_up', 'ATG_LXRb_TRANS_dn', 'ATG_LXRb_TRANS_up',
                 'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn', 'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up',
                 'ATG_M_32_CIS_dn', 'ATG_M_32_CIS_up', 'ATG_M_32_TRANS_dn', 'ATG_M_32_TRANS_up', 'ATG_M_61_TRANS_up',
                 'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up', 'ATG_Myc_CIS_dn', 'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn',
                 'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn', 'ATG_NF_kB_CIS_up', 'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up',
                 'ATG_NRF2_ARE_CIS_dn', 'ATG_NRF2_ARE_CIS_up', 'ATG_NURR1_TRANS_dn', 'ATG_NURR1_TRANS_up',
                 'ATG_Oct_MLP_CIS_dn', 'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up',
                 'ATG_PPARa_TRANS_dn', 'ATG_PPARa_TRANS_up', 'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up',
                 'ATG_PPRE_CIS_dn', 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up', 'ATG_PXR_TRANS_dn',
                 'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RARa_TRANS_dn', 'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn',
                 'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn', 'ATG_RARg_TRANS_up', 'ATG_RORE_CIS_dn', 'ATG_RORE_CIS_up',
                 'ATG_RORb_TRANS_dn', 'ATG_RORg_TRANS_dn', 'ATG_RORg_TRANS_up', 'ATG_RXRa_TRANS_dn',
                 'ATG_RXRa_TRANS_up', 'ATG_RXRb_TRANS_dn', 'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up',
                 'ATG_STAT3_CIS_dn', 'ATG_STAT3_CIS_up', 'ATG_Sox_CIS_dn', 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn',
                 'ATG_Sp1_CIS_up', 'ATG_TAL_CIS_dn', 'ATG_TAL_CIS_up', 'ATG_TA_CIS_dn', 'ATG_TA_CIS_up',
                 'ATG_TCF_b_cat_CIS_dn', 'ATG_TCF_b_cat_CIS_up', 'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up',
                 'ATG_THRa1_TRANS_dn', 'ATG_THRa1_TRANS_up', 'ATG_VDRE_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_VDR_TRANS_dn',
                 'ATG_VDR_TRANS_up', 'ATG_XTT_Cytotoxicity_up', 'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn',
                 'ATG_p53_CIS_up', 'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down',
                 'BSK_3C_MCP1_down', 'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
                 'BSK_3C_Thrombomodulin_down', 'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down',
                 'BSK_3C_TissueFactor_up', 'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down',
                 'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down', 'BSK_4H_Pselectin_up',
                 'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up',
                 'BSK_BE3C_HLADR_down', 'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                 'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down', 'BSK_BE3C_SRB_down',
                 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down', 'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up',
                 'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down', 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up',
                 'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_LDLR_up', 'BSK_CASM3C_MCP1_down',
                 'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down',
                 'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_Proliferation_up', 'BSK_CASM3C_SAA_down',
                 'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_down',
                 'BSK_CASM3C_Thrombomodulin_up', 'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                 'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up', 'BSK_KF3CT_ICAM1_down',
                 'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down', 'BSK_KF3CT_IP10_up', 'BSK_KF3CT_MCP1_down',
                 'BSK_KF3CT_MCP1_up', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down', 'BSK_KF3CT_TGFb1_down',
                 'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down', 'BSK_LPS_Eselectin_down',
                 'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL1a_up', 'BSK_LPS_IL8_down', 'BSK_LPS_IL8_up',
                 'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down', 'BSK_LPS_PGE2_up', 'BSK_LPS_SRB_down',
                 'BSK_LPS_TNFa_down', 'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down', 'BSK_LPS_TissueFactor_up',
                 'BSK_LPS_VCAM1_down', 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down',
                 'BSK_SAg_Eselectin_down', 'BSK_SAg_Eselectin_up', 'BSK_SAg_IL8_down', 'BSK_SAg_IL8_up',
                 'BSK_SAg_MCP1_down', 'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
                 'BSK_SAg_PBMCCytotoxicity_up', 'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down',
                 'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_EGFR_down', 'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down',
                 'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down', 'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
                 'BSK_hDFCGF_MMP1_up', 'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down', 'BSK_hDFCGF_SRB_down',
                 'BSK_hDFCGF_TIMP1_down', 'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn', 'CEETOX_H295R_ANDR_dn',
                 'CEETOX_H295R_CORTISOL_dn', 'CEETOX_H295R_DOC_dn', 'CEETOX_H295R_DOC_up', 'CEETOX_H295R_ESTRADIOL_dn',
                 'CEETOX_H295R_ESTRADIOL_up', 'CEETOX_H295R_ESTRONE_dn', 'CEETOX_H295R_ESTRONE_up',
                 'CEETOX_H295R_OHPREG_up', 'CEETOX_H295R_OHPROG_dn', 'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up',
                 'CEETOX_H295R_TESTO_dn', 'CLD_ABCB1_48hr', 'CLD_ABCG2_48hr', 'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr',
                 'CLD_CYP1A1_6hr', 'CLD_CYP1A2_24hr', 'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr',
                 'CLD_CYP2B6_48hr', 'CLD_CYP2B6_6hr', 'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr',
                 'CLD_GSTA2_48hr', 'CLD_SULT2A_24hr', 'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr', 'CLD_UGT1A1_48hr',
                 'NCCT_HEK293T_CellTiterGLO', 'NCCT_QuantiLum_inhib_2_dn', 'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn',
                 'NCCT_TPO_GUA_dn', 'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1',
                 'NVS_ADME_hCYP1A2', 'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6', 'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9',
                 'NVS_ADME_hCYP2D6', 'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12', 'NVS_ENZ_hAChE',
                 'NVS_ENZ_hAMPKa1', 'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE', 'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D',
                 'NVS_ENZ_hDUSP3', 'NVS_ENZ_hES', 'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b',
                 'NVS_ENZ_hMMP1', 'NVS_ENZ_hMMP13', 'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7', 'NVS_ENZ_hMMP9',
                 'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1', 'NVS_ENZ_hPDE5', 'NVS_ENZ_hPI3Ka', 'NVS_ENZ_hPTEN',
                 'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12', 'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9', 'NVS_ENZ_hPTPRC',
                 'NVS_ENZ_hSIRT1', 'NVS_ENZ_hSIRT2', 'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2', 'NVS_ENZ_oCOX1',
                 'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS', 'NVS_ENZ_rMAOAC', 'NVS_ENZ_rMAOAP',
                 'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C', 'NVS_GPCR_bAdoR_NonSelective',
                 'NVS_GPCR_bDR_NonSelective', 'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2', 'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4',
                 'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK', 'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A',
                 'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7', 'NVS_GPCR_hAT1', 'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a',
                 'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C', 'NVS_GPCR_hAdrb1', 'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3',
                 'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s', 'NVS_GPCR_hDRD4.4', 'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1',
                 'NVS_GPCR_hM1', 'NVS_GPCR_hM2', 'NVS_GPCR_hM3', 'NVS_GPCR_hM4', 'NVS_GPCR_hNK2', 'NVS_GPCR_hOpiate_D1',
                 'NVS_GPCR_hOpiate_mu', 'NVS_GPCR_hTXA2', 'NVS_GPCR_p5HT2C', 'NVS_GPCR_r5HT1_NonSelective',
                 'NVS_GPCR_r5HT_NonSelective', 'NVS_GPCR_rAdra1B', 'NVS_GPCR_rAdra1_NonSelective',
                 'NVS_GPCR_rAdra2_NonSelective', 'NVS_GPCR_rAdrb_NonSelective', 'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3',
                 'NVS_GPCR_rOpiate_NonSelective', 'NVS_GPCR_rOpiate_NonSelectiveNa', 'NVS_GPCR_rSST', 'NVS_GPCR_rTRH',
                 'NVS_GPCR_rV1', 'NVS_GPCR_rabPAF', 'NVS_GPCR_rmAdra2B', 'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL',
                 'NVS_IC_rCaDHPRCh_L', 'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1', 'NVS_LGIC_h5HT3',
                 'NVS_LGIC_hNNR_NBungSens', 'NVS_LGIC_rGABAR_NonSelective', 'NVS_LGIC_rNNR_BungSens', 'NVS_MP_hPBR',
                 'NVS_MP_rPBR', 'NVS_NR_bER', 'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR', 'NVS_NR_hCAR_Antagonist',
                 'NVS_NR_hER', 'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist', 'NVS_NR_hGR', 'NVS_NR_hPPARa',
                 'NVS_NR_hPPARg', 'NVS_NR_hPR', 'NVS_NR_hPXR', 'NVS_NR_hRAR_Antagonist', 'NVS_NR_hRARa_Agonist',
                 'NVS_NR_hTRa_Antagonist', 'NVS_NR_mERa', 'NVS_NR_rAR', 'NVS_NR_rMR', 'NVS_OR_gSIGMA_NonSelective',
                 'NVS_TR_gDAT', 'NVS_TR_hAdoT', 'NVS_TR_hDAT', 'NVS_TR_hNET', 'NVS_TR_hSERT', 'NVS_TR_rNET',
                 'NVS_TR_rSERT', 'NVS_TR_rVMAT2', 'OT_AR_ARELUC_AG_1440', 'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960',
                 'OT_ER_ERaERa_0480', 'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440',
                 'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_ERa_EREGFP_0480',
                 'OT_FXR_FXRSRC1_0480', 'OT_FXR_FXRSRC1_1440', 'OT_NURR1_NURR1RXRa_0480', 'OT_NURR1_NURR1RXRa_1440',
                 'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2', 'TOX21_ARE_BLA_agonist_ratio',
                 'TOX21_ARE_BLA_agonist_viability', 'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2',
                 'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1', 'TOX21_AR_BLA_Antagonist_ch2',
                 'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_BLA_Antagonist_viability', 'TOX21_AR_LUC_MDAKB2_Agonist',
                 'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist',
                 'TOX21_Aromatase_Inhibition', 'TOX21_AutoFluor_HEK293_Cell_blue', 'TOX21_AutoFluor_HEK293_Media_blue',
                 'TOX21_AutoFluor_HEPG2_Cell_blue', 'TOX21_AutoFluor_HEPG2_Cell_green',
                 'TOX21_AutoFluor_HEPG2_Media_blue', 'TOX21_AutoFluor_HEPG2_Media_green', 'TOX21_ELG1_LUC_Agonist',
                 'TOX21_ERa_BLA_Agonist_ch1', 'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio',
                 'TOX21_ERa_BLA_Antagonist_ch1', 'TOX21_ERa_BLA_Antagonist_ch2', 'TOX21_ERa_BLA_Antagonist_ratio',
                 'TOX21_ERa_BLA_Antagonist_viability', 'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist',
                 'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2', 'TOX21_ESRE_BLA_ratio', 'TOX21_ESRE_BLA_viability',
                 'TOX21_FXR_BLA_Antagonist_ch1', 'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2',
                 'TOX21_FXR_BLA_agonist_ratio', 'TOX21_FXR_BLA_antagonist_ratio', 'TOX21_FXR_BLA_antagonist_viability',
                 'TOX21_GR_BLA_Agonist_ch1', 'TOX21_GR_BLA_Agonist_ch2', 'TOX21_GR_BLA_Agonist_ratio',
                 'TOX21_GR_BLA_Antagonist_ch2', 'TOX21_GR_BLA_Antagonist_ratio', 'TOX21_GR_BLA_Antagonist_viability',
                 'TOX21_HSE_BLA_agonist_ch1', 'TOX21_HSE_BLA_agonist_ch2', 'TOX21_HSE_BLA_agonist_ratio',
                 'TOX21_HSE_BLA_agonist_viability', 'TOX21_MMP_ratio_down', 'TOX21_MMP_ratio_up', 'TOX21_MMP_viability',
                 'TOX21_NFkB_BLA_agonist_ch1', 'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio',
                 'TOX21_NFkB_BLA_agonist_viability', 'TOX21_PPARd_BLA_Agonist_viability',
                 'TOX21_PPARd_BLA_Antagonist_ch1', 'TOX21_PPARd_BLA_agonist_ch1', 'TOX21_PPARd_BLA_agonist_ch2',
                 'TOX21_PPARd_BLA_agonist_ratio', 'TOX21_PPARd_BLA_antagonist_ratio',
                 'TOX21_PPARd_BLA_antagonist_viability', 'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2',
                 'TOX21_PPARg_BLA_Agonist_ratio', 'TOX21_PPARg_BLA_Antagonist_ch1', 'TOX21_PPARg_BLA_antagonist_ratio',
                 'TOX21_PPARg_BLA_antagonist_viability', 'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist',
                 'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1', 'TOX21_VDR_BLA_agonist_ch2',
                 'TOX21_VDR_BLA_agonist_ratio', 'TOX21_VDR_BLA_antagonist_ratio', 'TOX21_VDR_BLA_antagonist_viability',
                 'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2', 'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p1_viability',
                 'TOX21_p53_BLA_p2_ch1', 'TOX21_p53_BLA_p2_ch2', 'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                 'TOX21_p53_BLA_p3_ch1', 'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p3_viability',
                 'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2', 'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p4_viability',
                 'TOX21_p53_BLA_p5_ch1', 'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio', 'TOX21_p53_BLA_p5_viability',
                 'Tanguay_ZF_120hpf_AXIS_up', 'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up',
                 'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up', 'Tanguay_ZF_120hpf_EYE_up',
                 'Tanguay_ZF_120hpf_JAW_up', 'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up',
                 'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up', 'Tanguay_ZF_120hpf_PIG_up',
                 'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up',
                 'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up', 'Tanguay_ZF_120hpf_YSE_up'
             ]}
# f = open('tasks_dic.txt', 'w')
# f.write(str(tasks_dic))
# f.close()

# store the washing info
info_data = pd.DataFrame(
    {'data set': [], 'origin_num': [], 'uncovered': [], 'inorganics': [], 'duplicates': [], 'new_num': []})
data_files = ['./bace/bace.csv', './bbbp/bbbp.csv', './clintox/clintox.csv', './esol/esol.csv',
              './freesolv/freesolv.csv', './hiv/hiv.csv',
              './lipophilicity/lipophilicity.csv', './muv/muv.csv', './sider/sider.csv', './tox21/tox21.csv',
              './toxcast/toxcast.csv']
# data_files = ['./bace/bace.csv', './tox21/tox21.csv']

# data_files = ['./clintox/clintox.csv']
targets = tasks_dic
for file_path in data_files:
    data_label = file_path.split('/')[1]
    print(
        '**************************************start washing for {}*****************************\n'.format(data_label))
    data = pd.read_csv(file_path)
    info = [data_label, len(data)]
    # dealing with salts, counterions, solvents, mixtures print('********Note: Heavy atom counting is generally an
    # effective way to remove extraneous solvents, salts and miscellaneous adducts, but will produce the wrong answer
    # sometimes if the desired component is not the largest.*******')
    print('********dealing with salts, counterions, solvents, mixtures*******')

    mdb_path = file_path.replace('.csv', '.mdb')
    sdf_path = file_path.replace('.csv', '.sdf')
    sdf_washed_path = file_path.replace('.csv', '_washed.sdf')
    mdb_washed_path = file_path.replace('.csv', '_washed.mdb')
    washed_file_path = file_path.replace('.csv', '_washed.csv')

    names_ls = data.columns.values.tolist()
    names_ls_cp = copy.deepcopy(names_ls)
    types_ls = copy.deepcopy(names_ls)

    types_ls[names_ls.index('smiles')] = 'molecule'
    names_ls_cp.remove('smiles')

    for col in names_ls_cp:
        if str(data[col].dtypes) == 'int64':
            types_ls[types_ls.index(col)] = 'int'
        elif str(data[col].dtypes) == 'float64':
            types_ls[types_ls.index(col)] = 'float'
        else:
            types_ls[types_ls.index(col)] = 'char'

    os.system('module load moe/2015.1001')

    # generate mdb(database) from csv file
    cmd1 = "moebatch  -exec \"mdb = db_Open ['%s', 'create']; db_ImportASCII [ascii_file:'%s', db_file:'%s', delimiter:',', quotes:1, titles:1, names:%s, types:%s]\"" % (
        mdb_path, file_path, mdb_path, names_ls, types_ls)
    os.system(cmd1)

    # export mdb(database) as sdf file
    cmd2 = "moebatch  -exec \"db_ExportSD ['%s', '%s', [],[]]\"" % (mdb_path, sdf_path)
    os.system(cmd2)

    # execute data washing
    # cmd3 = "E:\moe2018\moe2018\bin\sdwash %s -o %s  -quiet  -salts -component -neutralize"%(sdf_path,sdf_washed_path)
    cmd3 = "sdwash %s -o %s  -quiet  -salts -component -neutralize" % (sdf_path, sdf_washed_path)
    os.system(cmd3)

    # import the washed sdf file as mdb file
    cmd4 = "moebatch -exec \"mdb = db_Open ['%s', 'create']; db_ImportSD [mdb_file='%s', sd_file='%s', mol_field='mol', sd_fields=[],mdb_fields=[],mdb_field_types=[],[file_field:0, scan_file:1]];\"" % (
        mdb_washed_path, mdb_washed_path, sdf_washed_path)
    os.system(cmd4)

    # export the washed mdb file as the washed csv file
    cmd5 = "moebatch -exec \"[fields, field_types] = db_Fields '%s'; entries = db_Entries '%s'; db_ExportASCII [mdb_file='%s', ascii_file='%s',field_names=fields,entry_keys=entries,[delimiter:',', quotes:1, titles:1]]\"" % (
        mdb_washed_path, mdb_washed_path, mdb_washed_path, washed_file_path)
    os.system(cmd5)

    data_washed = pd.read_csv(washed_file_path)
    data_washed.rename(columns={'mol': 'smiles'}, inplace=True)
    data_washed = data_washed[data.columns]
    #     data_washed.to_csv('./bace/bace_washed.csv',index=False)
    data = copy.deepcopy(data_washed)

    # molecules can not be processed by RDKit or MOE
    print('********dealing with molecules can not be processed by RDKit or MOE*******')
    smiles_ls = data.smiles.values
    remained_smiles = []
    uncover_smiles = []
    uncover_indx = []
    uncover_moe = []
    for indx, smiles in enumerate(smiles_ls):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                remained_smiles.append(smiles)
            else:
                uncover_smiles.append(smiles)
                uncover_indx.append(indx)
        except:
            print('the smiles can not be recognized by MOE!')
            uncover_indx.append(indx)
            uncover_moe.append(indx)
    info.append(len(uncover_indx))
    print(
        'molecules in {} can not be processed by RDkit(version:2019.09.1) have:\n{}'.format(data_label, uncover_smiles))
    print('molecules in {} can not be processed by MOE(version:2015.1001) have:\n{}'.format(data_label, uncover_moe))
    print('the corresponding indexes for them are:\n{}'.format(uncover_indx))
    data.drop(index=uncover_indx, inplace=True)
    # data.reset_index(inplace=True, drop=True)

    # remove inorganic compounds
    print('********dealing with inorganic compounds*******')
    smiles_ls = data.smiles.values
    remained_smiles = []
    inorganics = []
    inorganics_indx = []
    for indx, smiles in enumerate(smiles_ls):
        mol = Chem.MolFromSmiles(smiles)
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                break
            else:
                count += 1
        if count == mol.GetNumAtoms():
            inorganics.append(smiles)
            inorganics_indx.append(data.index[indx])
        else:
            remained_smiles.append(smiles)
    info.append(len(inorganics))
    print('inorganic compounds in {} have:\n{}'.format(data_label, inorganics))
    print('the corresponding indexes for them are:\n{}'.format(inorganics_indx))
    data.drop(index=inorganics_indx, inplace=True)
    # data.reset_index(inplace=True, drop=True)
    # file_path_new = file_path.replace('.csv', '1.csv')
    # data.to_csv(file_path_new, index=False)  # update the csv file for the nest washing

    # in case of smiles nan
    # data.drop(index=data[data.smiles.isna()].index, inplace=True)
    # dealing with the duplicates according the canonical smiles
    print('********dealing with duplicates*******')
    smiles_ls = data.smiles.values
    cano_smiles = []
    for smiles in smiles_ls:
        # generate canonical smiles using RDKit
        cano_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    data['cano_smiles'] = cano_smiles
    duplicates = data.cano_smiles.value_counts()[data.cano_smiles.value_counts() >= 2]
    dup_confilct = []
    dup_confilct_indx = []
    print('duplicates in {} have:\n {}'.format(data_label, duplicates))
    # remove duplicates with inconsistent labels or retain any of them with consistent labels
    for dup_smiles in duplicates.index:
        new_df = pd.DataFrame(data[data.cano_smiles == dup_smiles][targets[data_label]].values)
        new_df.fillna(14687941, inplace=True)
        dup_flag = len(np.unique(new_df.values, axis=0))
        # duplicates with consistent labels (keep any one)
        if dup_flag == 1:
            for indx in data[data.cano_smiles == dup_smiles].index[1:]:
                data.drop(index=indx, inplace=True)
        # duplicates with inconsistent labels (remove all)
        else:
            dup_confilct.append(dup_smiles)
            dup_confilct_indx.extend(data[data.cano_smiles == dup_smiles].index)
            data.drop(index=data[data.cano_smiles == dup_smiles].index, inplace=True)
    print('duplicates with inconsistent labels in {} have:\n{}'.format(data_label, dup_confilct))
    print('the corresponding indexes for them are:\n{}'.format(dup_confilct_indx))
    info.append(len(dup_confilct))
    # data_new = copy.deepcopy(data)

    info.append(len(data))
    #print(info)
    data.to_csv('./washed/' + data_label + '_new.csv')
    print('**************************************washing over for {}*****************************\n'.format(data_label))
    info_data = info_data.append(
        pd.Series(info, index=['data set', 'origin_num', 'uncovered', 'inorganics', 'duplicates', 'new_num']),
        ignore_index=True)
info_data.to_csv('info_data.csv', index=False)


