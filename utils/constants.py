
SEQ = 'seq'
STRUCT = 'struct'
SEQ_STRUCT = 'SST'
SEQ_STRUCT_2 = 'SST2'
__ClinVar__ = "ClinVar"
__BalancedClinVar__ = 'BalancedClinVar'
__DMS__ = "DMS"
__Genebass__ = "Genebass"
__u__ = "u_dataset"
__up__ = "UniProt"
__hg19__ = "hg19"
__hg38__ = "hg38"


ESM3_struct_pad_token_id = 4099
ESM3_struct_bos_token_id = 4098
ESM3_struct_eos_token_id = 4097
ESM3_struct_mask_token_id = 4096

ESM3_seq_pad_token_id = 1
ESM3_seq_bos_token_id = 0
ESM3_seq_eos_token_id = 2

SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}
VQVAE_DIRECTION_LOSS_BINS = 16
VQVAE_PAE_BINS = 64
VQVAE_MAX_PAE_BIN = 31.0
VQVAE_PLDDT_BINS = 50

STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS["MASK"]
STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS["BOS"]
STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS["EOS"]
STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS["PAD"]
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS["CHAINBREAK"]
STRUCTURE_UNDEFINED_TOKEN = 955

sequence_vocabs = {'I': 12, 'C': 23, '.': 29, '<pad>': 1, 'T': 11, 'F': 18, '|': 31, 'X': 24, 'U': 26, 'V': 7, 'E': 9, 'L': 4, '<eos>': 2, '<cls>': 0, 'W': 22, 'S': 8, 'N': 17, 'Y': 19, 'A': 5, 'D': 13, 'P': 14, '<unk>': 3, 'Q': 16, 'B': 25, '<mask>': 32, 'R': 10, 'K': 15, 'O': 28, '-': 30, 'G': 6, 'Z': 27, 'H': 21, 'M': 20}


AA_name2letter_dct = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# VESMin
esm_AAorder = ['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']
# AM
AM_AAorder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# ESM 
hg_tok_vocabs = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
hg_index2AA_dct = {4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C'}
hg_index_of_AA = lambda x: hg_tok_vocabs[x] - 4


esm_name_dct = {
     'esm1b': 'esm1b_t33_650M_UR50S',
     'esm1v_1': 'esm1v_t33_650M_UR90S_1',
     'esm1v_2': 'esm1v_t33_650M_UR90S_2',
     'esm1v_3': 'esm1v_t33_650M_UR90S_3',
     'esm1v_4': 'esm1v_t33_650M_UR90S_4',
     'esm1v_5': 'esm1v_t33_650M_UR90S_5',
     'esm2_650m': 'esm2_t33_650M_UR50D',
     'esm2_3b': 'esm2_t36_3B_UR50D',
     'esm2_150m': 'esm2_t30_150M_UR50D',
     'esm2_8m': 'esm2_t6_8M_UR50D',
     'esm2_35m': 'esm2_t12_35M_UR50D',
     'esm2_15b': 'esm2_t48_15B_UR50D'
}


llrs_stats_dcts = {
    "ESM11_UPh_min": {
        "mu": -10.223638271060164,
        "sig": 4,
        "esm1b": -8.321401208029547,
        "esm1v_1": -6.766126915719046,
        "esm1v_2": -7.354341409990318,
        "esm1v_3": -6.699579332221035,
        "esm1v_4": -6.944233840717475,
        "esm1v_5": -7.418966920112755, 
        "esm2_650m": -7.016498910674287,
        "esm2_3b": -8.411810653307501, 
        "esm2_150m": -5.5433880545648595,
        "esm2_8m": -4.272698638001586, 
        "esm2_35m": -4.778072858460834, 
    },
    "r2_UPh_avg": {
        "mu": -7.795108005099982,
        "sig": 1,
        "esm2_650m": -6.9773881540135685,
        "esm1b": -8.226354870084437,
        "esm2_3b": -8.389871923389759,
        "esm1v_5": -7.586817067214266,
        "esm2_8m": -4.92015507224189,
        "esm2_35m": -5.4958512054031825,
    },
    "r3_UPnh_avg": {
        "mu": -8.095926294511342,
        "sig": 1,
        "esm2_650m": -7.455500240645004,
        "esm1b": -8.028647766948222,
        "esm2_3b": -8.949950593695117,
        "esm1v_5": -7.9496065741643225,
    },
}
