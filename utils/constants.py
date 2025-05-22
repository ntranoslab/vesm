
CLINVAR = 'ClinVar'
DMS = 'DMS'
SEQ = 'seq'
STRUCT = 'struct'
SEQ_STRUCT = 'SST'
SEQ_STRUCT_2 = 'SST2'

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
esm_AAorder = ['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']
AM_AAorder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
hg_tok_vocabs = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
hg_index2AA_dct = {4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C'}

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