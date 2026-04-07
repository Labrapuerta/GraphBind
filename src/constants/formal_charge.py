'''Formal charge — a simple integer: Arg/Lys = +1, Asp/Glu = -1, His = 0 or +1 depending on pH assumption (usually treated as neutral = 0 at pH 7), all others = 0. This is a hardcoded lookup.'''

# Formal charge at pH 7
formal_charge = {
    "ARG":  1, "LYS":  1,
    "ASP": -1, "GLU": -1,
    "HIS":  0,  # neutral at pH 7 (protonated = +1, use PropKa for accuracy)
    "ALA":  0, "VAL":  0, "LEU":  0, "ILE":  0, "PRO":  0,
    "PHE":  0, "TRP":  0, "MET":  0, "GLY":  0, "SER":  0,
    "THR":  0, "CYS":  0, "TYR":  0, "ASN":  0, "GLN":  0,
}



