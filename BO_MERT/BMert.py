import argparse
import logging
from itertools import izip
import cdec, cdec.score

N = 5 # 500



def evaluate(hyp, ref):
    """ Compute BLEU score for a set of hypotheses+references """
    return sum(cdec.score.BLEU(r).evaluate(h) for h, r in izip(hyp, ref)).score

def GenerateNBestList(decoder,sources,references):

    w = decoder.weights.tosparse()
    w0 = w.copy()

    hgs = []
    cands_set = []
    feats_set = []
    for source in sources:
        hg = decoder.translate(source)
        cands = hg.unique_kbest(N)
        feats = hg.unique_kbest_features(N)

        hgs.append(hg)
        cands_set.append(cands)
        feats_set.append(feats)

        
        for cand, feat in izip(cands,feats):
            print cand
            for f in feat:
                print f

    '''
    candidate_sets = [cdec.score.BLEU(refs).candidate_set() for refs in references]
    hgs = []
    for src, candidates in izip(sources, candidate_sets):
        hg = decoder.translate(src)
        hgs.append(hg)
        candidates.add_kbest(hg, N)
    score = evaluate((hg.viterbi() for hg in hgs), references)
    print score
    '''


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

    decoder = cdec.Decoder(config)
    decoder.read_weights(args.weights)
    with open(args.reference) as fp:
        references = fp.readlines()
    with open(args.source) as fp:
        sources = fp.readlines()
    assert len(references) == len(sources)
    GenerateNBestList(decoder,sources,references)


if __name__ == '__main__':
    main()
    #python BMert.py -w weights.0 -c cdec.ini -r refs.input -s source.input 
