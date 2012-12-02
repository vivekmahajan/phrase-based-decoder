## @author baskaran

import os
import sys
import time
from ordereddict import *
import settings

#kenlm_swig_wrapper = os.environ.get("NGRAM_SWIG_KENLM")
kenlm_swig_wrapper = "/cs/natlang-sw/Linux-x86_64/NL/LM/NGRAM-SWIG-KENLM"
print "wrapper!!!!",kenlm_swig_wrapper 
if ( kenlm_swig_wrapper is None ):
         sys.stderr.write("Error: Environment variable NGRAM_SWIG_KENLM is not set. Exiting!!\n")
         sys.exit(1)
sys.path.insert(1, kenlm_swig_wrapper)
from kenlm import *
'''Language Model class for KENLM'''

class KENLangModel(object):

    #print "wrapper!!!!" 
    #kenlm_swig_wrapper = os.environ.get("NGRAM_SWIG_KENLM")
    #print "wrapper!!!!",kenlm_swig_wrapper 
    #if ( kenlm_swig_wrapper is None ):
    #     sys.stderr.write("Error: Environment variable NGRAM_SWIG_KENLM is not set. Exiting!!\n")
    #     sys.exit(1)
    #sys.path.insert(1, kenlm_swig_wrapper)
    #from kenlm import *
    #'''Language Model class for KENLM'''
    LM = None                                       # Class attribute containing LM
    lm_order = None                                 # Class attribute for lm_order
    log_normalizer = 0.434294                       # Normalizer to convert the log-10 value (of KenLM) to natural log
    elider = ''
    lmCache = {}                      # Cache using the OrderedDict (from http://code.activestate.com/recipes/576693/)
    max_cache_size = 100000

    __slots__ = ()

    def __init__(self, lm_order, lmFile, elider_str):
        '''Import the KENLM wrapper module and initialize LM'''

        KENLangModel.lm_order = lm_order
        KENLangModel.elider = elider_str
        if not settings.opts.no_lm_cache:
        #    KENLangModel.lmCache = LRUOrderedDict()
            KENLangModel.lmCache = LMCache()
        self.loadLM(lmFile)

    def __del__(self):
        '''Deletes the LM variable'''

        if not settings.opts.no_lm_cache:
            KENLangModel.lmCache.__clear__()             # Clear the cache
        deleteLM(KENLangModel.LM)

    def loadLM(self, lmFile):
        '''Function for loading the LM'''

        print "Loading LM file %s ... " % (lmFile)
        t_beg = time.time()
        KENLangModel.LM = readLM(lmFile)             # Read lmFile into LM variable
        t_end = time.time()
        print "Time taken for loading LM        : %1.3f sec\n" % (t_end - t_beg)

    @classmethod
    def manageCache(cls):
        '''House-keeping operations for LM Cache'''

        ## Naive way - Just clear the cache for every 'p' sentences (for simple dict)
        KENLangModel.lmCache.__clear__()
        return None

        """
        print "    %% Total entries in the cache :", KENLangModel.lmCache.getSize()

        ## Better way?? Just retain the max_cache_size most recently added entries
        if KENLangModel.lmCache.getSize() > KENLangModel.max_cache_size:
            KENLangModel.lmCache = LRUOrderedDict(KENLangModel.lmCache.popitem() for indx in xrange(KENLangModel.max_cache_size))
            KENLangModel.lmCache.setSize(KENLangModel.max_cache_size)
        """

        """
        ## Inefficient way - Delete all the least recently added entries.
        ##      Because much more entries might have to be deleted than being retained.
        #while curr_cache_size > KENLangModel.max_cache_size:
        #    (key, val) = KENLangModel.lmCache.popitem(last=False)
        #    curr_cache_size -= 1
        """

    ## Functions for querying the LM
    ## Define them as classmethods so that they can be called directly with class
    #@classmethod
    #def queryLM(cls, phr, phr_len):
    #    '''Score a target phrase with the Language Model and return natural log'''

    #    return KENLangModel.queryLMlog10(phr, phr_len) / cls.log_normalizer

    @classmethod
    def queryLM(cls, phr, order):
            '''Score a target phrase with the Language Model and return natural log'''
            if len(phr) > order:  
               return getNGramProb(cls.LM, phr, order,'true')
            else:
               return getNGramProb(cls.LM, phr, len(phr.split()))      
            
    @classmethod
    def queryLMlog10(cls, phr, phr_len):
        '''Score a target phrase with the Language Model and return base-10 log'''

        if settings.opts.no_lm_cache:
            lm_score = getNGramProb(cls.LM, phr, phr_len)
            return lm_score

        lm_score = KENLangModel.lmCache.__getitem__( (phr,) )
        if lm_score is None:
            lm_score = getNGramProb(cls.LM, phr, phr_len)
            KENLangModel.lmCache.__setitem__( (phr,), lm_score )
        return lm_score

    @classmethod
    def calcUNKLMScore(cls, sent):
        '''Calculate the LM score contribution by UNK (OOV) words'''

        return scoreUNK(cls.LM, sent) / cls.log_normalizer

    @classmethod
    def printState(cls, state):
        """  Printing the KENLM state object (for debugging) """

        getHistory(cls.LM, state)

    @classmethod
    def scorePhrnElide(cls, wordsLst, e_len, mgramSpans, statesLst, r_lm_state):
        '''Score all the complete m-grams in a given consequent item'''

        lm_temp = 0.0
        ## Get the forward looking state for current target hypothesis
        if r_lm_state is None:
            r_lm_state = getEmptyState(cls.LM)
            dummy_prob = getNGramProb(cls.LM, ' '.join( wordsLst[e_len-cls.lm_order:] ), getEmptyState(cls.LM), r_lm_state)

        ## Score the complete n-gram phraes
        span_indx = 0
        for (mgram_beg, mgram_end) in mgramSpans:
            lm_hist = statesLst[span_indx]
            ngram_phr = ' '.join( wordsLst[mgram_beg:mgram_end] )

            if lm_hist is None and settings.opts.no_lm_cache:
                ngram_scr = getNGramProb(cls.LM, ngram_phr, cls.lm_order, 'true')
            elif lm_hist is None:
                ngram_scr = KENLangModel.lmCache.__getitem__( (ngram_phr,) )
                if ngram_scr is None:
                    ngram_scr = getNGramProb(cls.LM, ngram_phr, cls.lm_order, 'true')
                    KENLangModel.lmCache.__setitem__( (ngram_phr,), ngram_scr )
            else:
                ngram_scr = getNGramProb(cls.LM, ngram_phr, lm_hist, 'true')
            lm_temp += ngram_scr
            span_indx += 1

        # Finally, elide the string again
        e_tgt = ' '.join(wordsLst[0:cls.lm_order-1] + [cls.elider] + wordsLst[e_len-(cls.lm_order-1):])

        return (lm_temp / cls.log_normalizer, e_tgt, r_lm_state)

    @classmethod
    def getLMHeuCost(cls, wordsLst, e_len):
        """ Compute Heuristic LM score for a given consequent item (by merging one or two antecedents).

            Heuristic LM score is calculated for m-1 words in the beginning and end
            of the candidate. The sentence-boundary markers (<s> and </s>) are first
            appended to the candidate and the heuristic scores are then computed
            separately for three cases, i) no boundary, ii) only left boundary and
            iii) only right boundary. The best (max) score from among the three are
            then returned as the LM heuristic score.
        """

        if (e_len < cls.lm_order): initWrds = wordsLst
        else: initWrds = wordsLst[0:cls.lm_order-1]

        # Compute LM heuristic score for the first m-1 words
        if (wordsLst[0] == "<s>"):                                  # Hypothesis is an S-rule (is_S_rule is True)
            return KENLangModel.helperLMHeu(' '.join(initWrds[1:]), 1, 0) / cls.log_normalizer

        mgram_phr = ' '.join(initWrds)                              # Hypothesis is *not* an S-rule (is_S_rule is False)
        lm_heu_w_edge = KENLangModel.helperLMHeu(mgram_phr, 1, 0)
        lm_heu_wo_edge = KENLangModel.helperLMHeu(mgram_phr, 0, 0)
        lmHueLst = [lm_heu_wo_edge, lm_heu_w_edge, lm_heu_wo_edge]

        # Compute LM heuristic score for the last m-1 words
        if wordsLst[-1] != "</s>":
            if e_len <= cls.lm_order: phr_beg_indx = 1
            else: phr_beg_indx = e_len - cls.lm_order + 1
            lmHueLst[2] += KENLangModel.helperLMHeu(' '.join(wordsLst[phr_beg_indx:]), 0, 1)

        return max(lmHueLst) / cls.log_normalizer                           # Return the max value of LM heu

    @classmethod
    def helperLMHeu(cls, mgram_phr, s_beg, s_end):

        if settings.opts.no_lm_cache:
            mgram_heu_scr = getLMHeuProb(cls.LM, mgram_phr, s_beg, s_end)
            return mgram_heu_scr

        mgram_heu_scr = KENLangModel.lmCache.__getitem__( (mgram_phr, s_beg, s_end) )
        if mgram_heu_scr is None:
            mgram_heu_scr = getLMHeuProb(cls.LM, mgram_phr, s_beg, s_end)
            KENLangModel.lmCache.__setitem__( (mgram_phr, s_beg, s_end), mgram_heu_scr )
        return mgram_heu_scr

class LMCache(object):
    '''Implements a simple LM cache with python dict'''

    __slots__ = "cacheDict"

    def __init__(self):
        self.cacheDict = {}

    def __setitem__(self, key, val):
        self.cacheDict[key] = val

    def __getitem__(self, key, def_val=None):
        return self.cacheDict.get(key, def_val)

    def __clear__(self):
        self.cacheDict.clear()


class LRUOrderedDict(OrderedDict):
    '''Extends the OrderedDict with LRU order'''

    __slots__ = "__dict_size"
    __dict_size = 0

    def __setitem__(self, key, val):
        OrderedDict.__setitem__(self, key, val)
        self.__dict_size += 1

    def __getitem__(self, key, def_val=None):
        return self.get(key, def_val)

    def __clear__(self):
        OrderedDict.clear(self)

    def setSize(self, new_size):
        self.__dict_size = new_size

    def getSize(self):
        return self.__dict_size
