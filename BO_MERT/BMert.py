import argparse
import logging
import numpy as np
from itertools import izip
import cdec, cdec.score

N = 5 # 500

class BMert:

    def __init__(self,config,weights_file,sources,references):

        self.decoder = cdec.Decoder(config)
        self.decoder.read_weights(weights_file)
        self.w0 = self.decoder.weights.tosparse().copy()
        self.sources = sources
        self.references = references

        self.feats_set=[]
        self.cands_set=[]

        self.N = N # N of the NBestList
        self.ref_size = len(references)
        self.BLEURecord = {}

    def GenerateNBestList(self):
        for source in self.sources:
            hg = self.decoder.translate(source)
            cands = hg.unique_kbest(N)
            feats = hg.unique_kbest_features(N)

            cand_list = []
            for cand in cands:
                cand_list.append(cand)

            self.cands_set.append(cand_list)
            self.feats_set.append(feats)

    def BestCandsList(self,new_weights):
        # idx_ref
        best_list =[]
        for idx_ref in range(0,self.ref_size):
            score_list = []
            #print idx_ref
            for feat in self.feats_set[idx_ref]:
                #print feat.dot(self.w0)
                score_list.append(feat.dot(self.w0))
            best_list.append(np.array(score_list).argmax())
        return best_list

    def Score_BLEU(self,best_list):
        #sum_score = 0
        cand_list = []
        for idx_ref in xrange(0,self.ref_size):
            #print self.references[idx_ref]
            #print self.cands_set[idx_ref][best_list[idx_ref]]
            if (idx_ref, best_list[idx_ref]) in self.BLEURecord:
                sum_score += self.BLEURecord[(idx_ref, best_list[idx_ref])]
            else:
                cand = str(self.cands_set[best_list[idx_ref]])
                #score = evaluate_single_BLEU(self.references[idx_ref],cand)
                #self.BLEURecord[(idx_ref, best_list[idx_ref])] = score
                #sum_score += score
                #print score
                cand_list.append(cand)
        
        return evaluate_BLEU(self.references,cand_list)
        #return sum_score


def evaluate_BLEU(ref,hyp):
    return sum(cdec.score.BLEU(r).evaluate(h) for h, r in izip(hyp, ref)).score

def evaluate_single_BLEU(ref,hyp):
    return cdec.score.BLEU(ref).evaluate(hyp).score


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='cdec config', required=True)
    parser.add_argument('-w', '--weights', help='initial weights', required=True)
    parser.add_argument('-r', '--reference', help='reference file', required=True)
    parser.add_argument('-s', '--source', help='source file', required=True)
    args = parser.parse_args()

    with open(args.config) as fp:
            config = fp.read()
    with open(args.reference) as fp:
        references = fp.readlines()
    with open(args.source) as fp:
        sources = fp.readlines()

    assert len(references) == len(sources)
    weights_file= args.weights
    myBMert = BMert(config,weights_file,sources,references) 
    # 1. Generate NBestList
    myBMert.GenerateNBestList()

    # 2. Generate RefBLEUDict
    BestCandsList = myBMert.BestCandsList(myBMert.w0) 

    print myBMert.Score_BLEU(BestCandsList)


if __name__ == '__main__':
    main()
    #python BMert.py -w weights.0 -c cdec.ini -r refs.input -s source.input 
