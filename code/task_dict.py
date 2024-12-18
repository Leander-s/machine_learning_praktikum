tasks_dic = {'freesolv': ['activity'], 'esol': ['activity'], 'lipop': ['activity'], 'bace': ['activity'],
             'bbbp': ['activity'], 'hiv': ['activity'],
             'clintox': ['FDA_APPROVED', 'CT_TOX'],
             'sider': ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7', 'SIDER8', 'SIDER9',
                       'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13', 'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17',
                       'SIDER18', 'SIDER19', 'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25',
                       'SIDER26', 'SIDER27'],
             'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                       'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
             'muv': [
                 "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                 "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
             ],
             'toxcast': ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
                         'APR_HepG2_CellCycleArrest_24h_dn',
                         'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                         'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up',
                         'APR_HepG2_MitoMass_24h_dn', 'APR_HepG2_MitoMass_72h_dn',
                         'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                         'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up',
                         'APR_HepG2_OxidativeStress_24h_up',
                         'APR_HepG2_OxidativeStress_72h_up',
                         'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up',
                         'APR_HepG2_p53Act_72h_up', 'ATG_AP_1_CIS_up', 'ATG_Ahr_CIS_up',
                         'ATG_BRE_CIS_up', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_up',
                         'ATG_DR4_LXR_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_EGR_CIS_up',
                         'ATG_ERE_CIS_up', 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn',
                         'ATG_HIF1a_CIS_up', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn',
                         'ATG_ISRE_CIS_dn', 'ATG_MRE_CIS_up', 'ATG_NRF2_ARE_CIS_up',
                         'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_up', 'ATG_PPARg_TRANS_up',
                         'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up',
                         'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RORE_CIS_up',
                         'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_up', 'ATG_Sp1_CIS_up',
                         'ATG_TCF_b_cat_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_Xbp1_CIS_up',
                         'ATG_p53_CIS_dn', 'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down',
                         'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down', 'BSK_3C_MCP1_down',
                         'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
                         'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down',
                         'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down',
                         'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down',
                         'BSK_4H_Pselectin_down', 'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down',
                         'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_BE3C_HLADR_down',
                         'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                         'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down',
                         'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down',
                         'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down',
                         'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL8_down',
                         'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_MCP1_down',
                         'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MIG_down',
                         'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_SAA_down',
                         'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_up',
                         'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                         'BSK_CASM3C_uPAR_down', 'BSK_KF3CT_ICAM1_down',
                         'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down',
                         'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down',
                         'BSK_KF3CT_TGFb1_down', 'BSK_KF3CT_TIMP2_down',
                         'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down',
                         'BSK_LPS_Eselectin_down', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL8_down',
                         'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down',
                         'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down',
                         'BSK_LPS_TissueFactor_down', 'BSK_LPS_VCAM1_down',
                         'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down',
                         'BSK_SAg_Eselectin_down', 'BSK_SAg_IL8_down', 'BSK_SAg_MCP1_down',
                         'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
                         'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down',
                         'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_IL8_down',
                         'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down',
                         'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
                         'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down',
                         'BSK_hDFCGF_SRB_down', 'BSK_hDFCGF_TIMP1_down',
                         'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
                         'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn',
                         'CEETOX_H295R_ESTRONE_dn', 'CEETOX_H295R_ESTRONE_up',
                         'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_NR_bER', 'NVS_NR_hER',
                         'NVS_NR_hPPARg', 'NVS_NR_hPXR', 'NVS_NR_mERa', 'OT_AR_ARSRC1_0960',
                         'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480',
                         'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_FXR_FXRSRC1_0480',
                         'OT_NURR1_NURR1RXRa_0480', 'TOX21_ARE_BLA_agonist_ratio',
                         'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_LUC_MDAKB2_Antagonist',
                         'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist',
                         'TOX21_Aromatase_Inhibition', 'TOX21_ERa_BLA_Antagonist_ratio',
                         'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_FXR_BLA_antagonist_ratio',
                         'TOX21_MMP_ratio_down', 'TOX21_TR_LUC_GH3_Antagonist',
                         'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p2_ch2',
                         'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                         'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p4_ratio',
                         'TOX21_p53_BLA_p5_ratio', 'Tanguay_ZF_120hpf_AXIS_up',
                         'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_JAW_up',
                         'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_PE_up',
                         'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_YSE_up']}
