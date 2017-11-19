import argparse
import re
import numpy as np
import nltk
from nltk import tokenize
import pickle
from tqdm import tqdm
from itertools import chain

INDEX_TO_WORD = 'ix_to_word'
WORD_TO_INDEX = 'word_to_ix'
DATA = 'data'
SENTENCE_LENGTH = 'length of sentences'

def getSentences(path):
	with open('warpeace_input.txt', 'r') as f:
		text = f.read()
		text = re.sub('[\r\n]',' ',text)
		text = re.sub(r'[\x80-\xFF]+', '', text)
		sentences= tokenize.sent_tokenize(text.decode('UTF-8'))
	return sentences


def cleanSentences(sentences):
    clean_sentence = []
    for sent in tqdm(sentences):
        clean_sentence.append(nltk.word_tokenize(re.sub(r'[\',"]+', '', sent).lower()))
    return clean_sentence


def getLengths(cleaned):
    length=[]
    for sent in cleaned:
        length.append(len(sent))
    return length


def dataGenerator(sentences, max_length, word_to_ix):
    size = len(sentences)
    data = np.zeros((size, max_length))
    for i,sent in tqdm(enumerate(sentences)):
        for j,word in enumerate(sent):
            data[i,j] = word_to_ix.get(word)
    return data

	
def savePickle(obj, foldername, filename):
    with open('{0}/{1}.pickle'.format(foldername, filename), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

		
def main(**kwargs):
	
	try:
		sentences = getSentences(kwargs['i'])
	except IOError as e:
		print e
		return
	
	#import ipdb; ipdb.set_trace()
	
	print "Text Loading Succeed.."
	
	cleaned = cleanSentences(sentences)
	lengths = getLengths(cleaned)
	
	unique_words = list(set(chain.from_iterable(cleaned)))
	unique_words = ['PAD']+unique_words
	
	ix_to_word = {ix:word for ix, word in enumerate(unique_words)}
	word_to_ix = {word:ix for ix, word in enumerate(unique_words)}
	
	print "Dictionary Generated.."
	
	data = dataGenerator(cleaned, max(lengths),word_to_ix)
	
	output_folder = kwargs['o']
	
	try:
		savePickle(ix_to_word, output_folder, INDEX_TO_WORD)
		savePickle(word_to_ix, output_folder, WORD_TO_INDEX)
		savePickle(data, output_folder, DATA)
		savePickle(lengths, output_folder, SENTENCE_LENGTH)
	except IOError as e:
		print e
		return
	
	print "Files Saved.."
	
	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Transform a novel txt file into the RNN ready dataset.')
	parser.add_argument('-i', type=str, help='Novel Address')
	parser.add_argument('-o', type=str, help='Output File Folder')
	args = parser.parse_args()
	
	main(**vars(args))