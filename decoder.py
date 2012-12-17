from heap import Heap 
import sys
from math import log
import copy
'''
kenlm_swig_wrapper = "/cs/natlang-sw/Linux-x86_64/NL/LM/NGRAM-SWIG-KENLM"
print "wrapper!!!!",kenlm_swig_wrapper
if ( kenlm_swig_wrapper is None ):
      sys.stderr.write("Error: Environment variable NGRAM_SWIG_KENLM is not set. Exiting!!\n")
      sys.exit(1)
sys.path.insert(1, kenlm_swig_wrapper)
from kenlm import *
path_lm = "/cs/natlang-data/wmt10/lm/eparl_nc_news_2m.en.lm"
lm = readLM(path_lm)
'''
class phrase_table:
    def __init__(self, filename):
        self.phrase_table_file = open(filename, "r")
        self.phrase_table = self.parse_file()

    def parse_file(self):
        phrase_table = {}
        for line in self.phrase_table_file:
            splits = line[:-1].split(' ||| ')
            if phrase_table.has_key(splits[0]):
                phrase_table[splits[0]].append((splits[1], splits[2]))
            else:
                phrase_table[splits[0]] = []
                phrase_table[splits[0]].append((splits[1], splits[2]))
                
        return phrase_table

class Hypothesis:
    def __init__(self, trans_source, dest, p_Lm, p_pt, dis, stack_id, end_d):
        self.p_Lm = pow( 10,float(p_Lm))
        self.p_pt = float(p_pt)
        self.dis = float(dis)
        self.dest = copy.deepcopy(dest)
        self.trans_source = copy.deepcopy(trans_source)
        self.stack_id = stack_id
        self.end_d = end_d
        #print "creating hypothesis ",self.trans_source, self.stack_id
    def get_priority(self):
        return (self.p_Lm * self.p_pt * self.dis)
    def get_mld(self):
        return (self.p_Lm * self.p_pt * self.dis)
        #return (self.p_pt * self.dis)

def generate_gaps(trans_source, end_d):
    #calculating the gaps
    string_gaps = []
    gap = []
    start_index = -1
    for i in range(0, len(trans_source)):
        if trans_source[i][1] == 1 :
            if len(gap) is not 0:
                if abs(start_index - end_d) < dis_limit:
                    string_gaps.append((gap, start_index))
            gap = []
            start_index = -1
        else:
            gap.append(trans_source[i][0])
            if start_index == -1:
                start_index = i
    if len(gap) is not 0:
        if abs(start_index - end_d) < dis_limit:
            string_gaps.append((gap, start_index))
    return string_gaps

def lang_model(destination):
    #return 1.0
    #return KENLangModel.queryLM(destination.split(" "),len(destination.split(" ")))  # for SRILM wrapper
    '''
    l = len(destination.split(" "))
    if l > 5:
        l = 5
    return getNGramProb(lm, destination , l, 'true')
    '''
    return 1.0
def generate_all_hypothesis(hp):
    #This will contains all the possible hypothesis
    hypothesis = []

    string_gaps = generate_gaps(hp.trans_source, hp.end_d)        
    #generating unigrams, bigrams and trigrams
    for gap in string_gaps:
        #length 1
        for i in range(0, len(gap[0])):
            #unigram 
            uni = gap[0][i]
            if pt.phrase_table.has_key(uni):
                for trans in pt.phrase_table[uni]: 
                    uni_trans_source = copy.deepcopy(hp.trans_source)
                    uni_trans_source[gap[1]+i] = (gap[0][i], 1)
                    uni_dest = hp.dest + trans[0] + " "
                    uni_p_Lm = lang_model(uni_dest.rstrip())
                    uni_p_pt = hp.get_mld() * float(trans[1])
                    uni_dis = pow(alpha, abs(hp.end_d-i-gap[1])) 
                    uni_stack_id = hp.stack_id + 1
                    uni_end_d = i + gap[1]
                    hpu = Hypothesis(trans_source = uni_trans_source, dest=uni_dest, p_Lm=uni_p_Lm, p_pt=uni_p_pt, dis=uni_dis, stack_id=uni_stack_id, end_d=uni_end_d)
                    hypothesis.append(hpu)
            #bigram
            if i > 0 :
                bi = gap[0][i-1]+" "+gap[0][i]
                if pt.phrase_table.has_key(bi):
                    for trans in pt.phrase_table[bi]: 
                        bi_trans_source = copy.deepcopy(hp.trans_source)
                        bi_trans_source[gap[1]+i] = (gap[0][i], 1)
                        bi_trans_source[gap[1]+i-1] = (gap[0][i-1], 1)
                        bi_dest = hp.dest + trans[0] + " "
                        bi_p_Lm = lang_model(bi_dest.rstrip())
                        bi_p_pt = hp.get_mld() * float(trans[1])
                        bi_dis = pow(alpha, abs(hp.end_d-i-1-gap[1])) 
                        bi_stack_id = hp.stack_id + 2
                        bi_end_d = i + gap[1]
                        hpb = Hypothesis(trans_source = bi_trans_source, dest=bi_dest, p_Lm=bi_p_Lm, p_pt=bi_p_pt, dis=bi_dis, stack_id=bi_stack_id, end_d=bi_end_d)
                        hypothesis.append(hpb)

            #trigram
            if i > 1 :     
                tri = gap[0][i-2]+" "+gap[0][i-1]+" "+gap[0][i]
                if pt.phrase_table.has_key(tri):
                    for trans in pt.phrase_table[tri]: 
                        tri_trans_source = copy.deepcopy(hp.trans_source)
                        tri_trans_source[gap[1]+i] = (gap[0][i], 1)
                        tri_trans_source[gap[1]+i-1] = (gap[0][i-1], 1)
                        tri_trans_source[gap[1]+i-2] = (gap[0][i-2], 1)
                        tri_dest = hp.dest + trans[0] + " "
                        tri_p_Lm = lang_model(tri_dest.rstrip())
                        tri_p_pt = hp.get_mld() * float(trans[1])
                        tri_dis = pow(alpha, abs(hp.end_d-i-2-gap[1])) 
                        tri_stack_id = hp.stack_id + 3
                        tri_end_d = i + gap[1]
                        hpt = Hypothesis(trans_source = tri_trans_source, dest=tri_dest, p_Lm=tri_p_Lm, p_pt=tri_p_pt, dis=tri_dis, stack_id=tri_stack_id, end_d=tri_end_d)
                        hypothesis.append(hpt)

    return hypothesis

class Decoder:
    def __init__(self, phrase_table):
        self.phrase_table = phrase_table
        self.source = ""
        self.stacks = {}

    def decode(self, source):
        self.source = source.split(" ")
        self.clear_stacks()
        self.init_stacks()
        #initialing the first stack
        trans_source = []
        for i in range(0, len(self.source)):
             trans_source.append((self.source[i], 0))
        hp = Hypothesis(trans_source=trans_source, dest="", p_Lm=1, p_pt=1, dis=1, stack_id=0, end_d=0)
        self.stacks[0].push(1, hp)
        for i in range(0, len(self.source)):
            #popping all the elements from the ith stack
            #print "size of the %s stack =  " % i , self.stacks[i].__len__() 
	    while self.stacks[i].__len__() > 0:
                hp = self.stacks[i].pop()
                for hypothesis in generate_all_hypothesis(hp):
                    stack_no = hypothesis.stack_id
                    if self.stacks[stack_no].__len__() >= beam:
                        #get the root
                        root_prob = self.stacks[stack_no]._heap[0][0]
                        if root_prob < hypothesis.get_priority():
                            self.stacks[stack_no].pop()
                            self.stacks[stack_no].push(hypothesis.get_priority(), hypothesis)
                    else:
                        self.stacks[stack_no].push(hypothesis.get_priority(), hypothesis)
        return self.stacks[len(self.source)]

             
    def clear_stacks(self):
        del self.stacks
        self.stacks = {}
        
    def init_stacks(self):
        for i in range(0, len(self.source)+1):
            self.stacks[i] = Heap() 

    

if __name__ == '__main__':
    global beam, dis_limit, pt, alpha
    if len(sys.argv) == 7:
        phrase_table_filename = sys.argv[1]
        decoder_input_filename = sys.argv[2]
        beam = int(sys.argv[3])
    	alpha = float(sys.argv[4])
	dis_limit = int(sys.argv[5])
	n_best = int(sys.argv[6])
    else:
        print >> sys.stderr, "usage:python %s phrase_table_file decoder_input beam alpha dis_limit n_best" % sys.argv[0]
        sys.exit(-1)
    
    pt = phrase_table(phrase_table_filename)
    decoder = Decoder(pt)
    for line in open(decoder_input_filename, "r"):
        output = decoder.decode(line[:-1].rstrip())
        stack = []
        print ">>>>>>>> ", line[:-1].rstrip(), " <<<<<<<<<"
            #print "length ",output.__len__()
        if output.__len__() == 0:
            print "Could not translate"
            continue
        while output.__len__() > 0:
            obj = output.pop()
            stack.append(obj)
                #print obj.dest, obj.get_priority(), obj.p_Lm
                #print output._heap[i][1].dest, output._heap[i][1].get_priority()
        for i in range(0, n_best):
            if len(stack) == 0:
                break
            obj = stack.pop()
            print obj.dest
            
