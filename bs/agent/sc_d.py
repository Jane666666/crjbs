import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from agent.scoring.get_score import get_isoelectric, get_gravy,isoelectric_getpoint
from agent.scoring.template import FVTemplate
from typing import List
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Seq import Seq

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoringFunctions:
    def __init__(self, scoring_func_names=None, weights=None, template=None):
        self.scoring_func_names = ['isoelectric'] if scoring_func_names is None else scoring_func_names
        self.weights = np.array([1] * len(self.scoring_func_names) if weights is None else weights)
        self.all_funcs = {'isoelectric': get_isoelectric, 'gravy': get_gravy}
        self.template = template
    def scores(self, aa_seqs: List, step: int, score_type='sum'):
        scores = []
        for fn_name in self.scoring_func_names:
            score=self.all_funcs[fn_name](template=self.template)(aa_seqs)
            scores.append(score)
        scores = np.float32(scores).T
class getpoint():
    def __init__(self,seq) :
        getpoint.seq = seq
    def isoelectric_getpoint(self):
        prot_analysis = ProteinAnalysis(str(self.seq))
        return 4+prot_analysis.isoelectric_point()
    def gravy_get(self):
        prot_analysis = ProteinAnalysis(str(self.seq))
        return prot_analysis.gravy()
def unit_tests():
    herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')
    sf = ScoringFunctions(template=herceptin, scoring_func_names=['isoelectric','gravy'])
    f = open("result/agent/agent_d/10k_samples.txt","r")
    listOfLines = f.read().splitlines()

    fp = open("output.txt","w")
    # Iterate over the lines
    for line in  listOfLines:
        prot_analysis = getpoint(line)
        point1 = prot_analysis.isoelectric_getpoint()
        point2 = prot_analysis.gravy_get()
        point = 2/3*point1+1/3*point2
        print(point,file=fp)
	
    fp.close()


if __name__ == "__main__":
    unit_tests()