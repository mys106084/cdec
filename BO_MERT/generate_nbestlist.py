import sys
import os
import subprocess
import logging, argparse
import gzip, itertools
try:
  import cdec.score
except ImportError:
  sys.stderr.write('Could not import pycdec, see cdec/python/README.md for details\n')
  sys.exit(1)


path_cdec = '/home/scratch/ymiao/workspace/cdec'
path_root = '/home/scratch/ymiao/workspace/cdec/BO_MERT'
path_decoder = path_cdec + '/decoder/cdec'
path_scorer=path_cdec + '/mteval/fast_score'

devFile = path_root + '/dev.sgm'
iniFile = path_root + '/cdec.ini'
weightsFile = path_root + '/weights'
# predefined
sourceFile = path_root + '/source.input'
refsFile = path_root + '/refs.input'
runFile = path_root + '/run'
decoderLogFile = path_root + '/decoder.sentserver.log'

hgsDir = path_root + '/hgs'
#mkdir hgs
if not os.path.exists(hgsDir):
	os.makedirs(hgsDir)


#From mira.py
def split_devset(devFile, sourceFile,refsFile):
	parallel = open(devFile)
	source = open(sourceFile,'w')
	refs = open(refsFile, 'w')
	references = []
	for line in parallel:
		s,r = line.strip().split(' ||| ',1)
		source.write(s+'\n')
		refs.write(r+'\n')
		references.append(r)
	source.close()
	refs.close()

#From mira.py	
def fast_score(hyps, refs, metric):
  scorer = cdec.score.Scorer(metric)
  logging.info('loaded {0} references for scoring with {1}'.format(
                len(refs), metric))
  if metric=='BLEU':
    logging.warning('BLEU is ambiguous, assuming IBM_BLEU\n')
    metric = 'IBM_BLEU'
  elif metric=='COMBI':
    logging.warning('COMBI metric is no longer supported, switching to '
                    'COMB:TER=-0.5;BLEU=0.5')
    metric = 'COMB:TER=-0.5;BLEU=0.5'
  stats = sum(scorer(r).evaluate(h) for h,r in itertools.izip(hyps,refs))
  logging.info('Score={} ({})'.format(stats.score, stats.detail))
  return stats.score

def point_score():
	onebestlist = []
	onebestlist_f = open(runFile)
	for line in onebestlist_f.read().strip().split('\n'):
		onebestlist.append(line.split(' ||| ')[1]) 
	onebestlist_f.close()
	return fast_score(onebestlist,references,'IBM_BLEU')

def main():
	'''
	parser= argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d', '--devset', required=True,
		help='dev set input file in parallel. '
		'format: src ||| ref1 ||| ref2')
	args = parser.parse_args()
	'''
	num_bestlist = 10
	max_iterations = 10

	# 0. Process the input dev data
	split_devset(devFile,sourceFile,refsFile)

	refs_f = open(refsFile)
	references = [line.split(' ||| ') for line in refs_f.read().strip().split('\n')]
	refs_f.close()

	#Loop
	for it in xrange(0,max_iterations):

		weights_f = weightsFile+'.'+str(it)
		decoder_f = decoderLogFile+'.'+str(it)
		run_f = runFile+'.'+str(it)
		# 1. Generate n-best-list
		cmd_decoder = path_decoder + ' -k '+ str(num_bestlist) + ' -r ' + ' -c ' + iniFile + ' -w ' + weights_f + ' -O ' + hgsDir + ' < ' + sourceFile + ' 2>'  + decoder_f + ' 1> ' + run_f
		subprocess.call(cmd_decoder,shell=True)



		# inner loop
		while 1:



			#print references
			#print onebestlist
			#cmd_score = path_scorer + ' -i '+ onebestlist +  ' -r ' + refs_f + ' > score'
			#cmd_score = path_scorer + ' -i '+ runFile +  ' -r ' + refsFile + ' > score'
			#subprocess.call(cmd_score,shell=True)
			


			break

		#../decoder/cdec -c ../python/cdec.ini -w weights.0 -show_weights -O ./hgs < source.input 



if __name__ == '__main__':
	main()



