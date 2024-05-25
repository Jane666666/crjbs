import os
import sys

sys.path.append("/geniusland/home/chenruijia/charlesxu90-ab-gen-0dec1ac/agent")
sys.path.append("/geniusland/home/chenruijia/charlesxu90-ab-gen-0dec1ac/agent/scoring")

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Seq import Seq
from agent.scoring.template import FVTemplate
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class get_isoelectric:
    def __init__(self, template: FVTemplate):
        self.template = template
    def __call__(self, aa_seqs: list):
        scores = np.zeros(len(aa_seqs))
        valid_seqs, valid_idxes = self.get_valid_seqs(aa_seqs)
        if len(valid_seqs) > 0:
            scores[np.array(valid_idxes)] = self.pred_prob(valid_seqs)
        return scores
    
class ge_gravy:
    def __init__(self, template: FVTemplate):
        self.template = template
    def __call__(self, aa_seqs: list):
        scores = np.zeros(len(aa_seqs))
        valid_seqs, valid_idxes = self.get_valid_seqs(aa_seqs)
        if len(valid_seqs) > 0:
            scores[np.array(valid_idxes)] = self.pred_prob(valid_seqs)
        return scores

def isoelectric_getpoint(self):
        prot_analysis = ProteinAnalysis(str(self))
        return prot_analysis.isoelectric_point()
def get_gravy(self):
        prot_analysis = ProteinAnalysis(str(self))
        return prot_analysis.gravy()