import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        ##raise NotImplementedError

##        dict_of_states_with_scores = dict()

        best_score = float('+inf')
        best_model = None

        for n in range(self.min_n_components, self.max_n_components+1):

            model = self.base_model(n)

            length_of_data = len(self.X)
            num_of_features = len(self.X[0]) # shown also as d
            num_of_states = n

            # Number of free parameters: p = n*(n-1) + (n-1) + 2*n*d = n**2 - 1 + 2*n*d
            num_of_params = num_of_states**2 - 1 + 2*num_of_states*num_of_features

            try:
                logL = model.score(self.X, self.lengths)

                bic = -2*logL + num_of_params * np.log(length_of_data)

##                dict_of_states_with_scores[n] = bic
                
                if bic < best_score:
                    best_score = bic
                    best_model = model

            except Exception:
                pass

##        state_with_best_bic = min(dict_of_states_with_scores.items(), key=lambda x: x[1])[0]

##        return self.base_model(state_with_best_bic)
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        ##raise NotImplementedError

        # #
        # print()
        # print('{}:'.format(self.this_word), len(self.sequences))
        # #

##        dict_of_states_with_scores = dict()

        best_score = float('-inf')
        best_model = None

        for n in range(self.min_n_components, self.max_n_components+1):

            model = self.base_model(n)

            try:
                logL = model.score(self.X, self.lengths)

                # Initialize accumulative score & count of other words
                word_log_score = 0
                word_count = 0

                # Loop through  other words
                for word, data in self.hwords.items():
                    if word != self.this_word:

                        word_X, word_lengths = data

                        try:
                            word_logL = model.score(word_X, word_lengths)
                            word_log_score += word_logL
                            word_count += 1
                        except Exception:
                            pass

                # Compute average score of other words
                a = 1 # regularizer
                avg_score_of_other_words = a / word_count * word_log_score
                dic = logL - avg_score_of_other_words

##                dict_of_states_with_scores[n] = dic
                
                if dic > best_score:
                    best_score = dic
                    best_model = model

            except Exception:
                pass

        # #
        # for item in dict_of_states_with_scores.items():
        #    print(item[0], item[1])
        # #

##        state_with_best_dic = max(dict_of_states_with_scores.items(), key=lambda x: x[1])[0]

##        return self.base_model(state_with_best_dic)
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        ##raise NotImplementedError

        # Initialize variables
        best_score = float('-inf')
        best_model = None
        best_state = None

        # Set maximum number of folds
        num_folds = 5
        # Set minimum number of folds
        my_num_of_folds = min(len(self.sequences), num_folds)

        # Consider case of single sequence
        if my_num_of_folds == 1:
            #self.X, self.lengths = self.sequences
            for n in range(self.min_n_components, self.max_n_components+1):

                try:
                    model = self.base_model(n)
                    logL = model.score(self.X, self.lengths)

                    if logL > best_score:
                        best_score = logL
                        best_model = model

                except Exception:
                    pass

            return best_model
        # Implement KFold CV for multiple sequences 
        kf = KFold(n_splits=my_num_of_folds, shuffle=True, random_state=self.random_state)

        ##
        #print()
        #print('{}:'.format(self.this_word), len(self.sequences))
        ##

##        dict_of_states_with_scores = dict()

        for n in range(self.min_n_components, self.max_n_components+1):
            
##            dict_of_states_with_scores[n] = list()
            list_of_scores = []

            try:
                for train_idx, test_idx in kf.split(self.sequences):

                    # Train
                    self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                    model = self.base_model(n)
                    # Test & Score
                    test_X, test_lengths = combine_sequences(test_idx, self.sequences)
                    logL = model.score(test_X, test_lengths)

##                    dict_of_states_with_scores[n].append(logL)
                    list_of_scores.append(logL)
                
                if np.mean(list_of_scores) > best_score:
                    best_score = np.mean(list_of_scores)
                    #best_model = model
                    best_state = n

            except Exception:
                pass

        ##
        #for item in dict_of_states_with_scores.items():
        #    print(item[0], len(item[1]), item[1])
        ##

        # Choose number of states with maximum average logLikelihood
##        best_n = max(dict_of_states_with_scores.items(), key=lambda x: np.mean(x[1]))[0]

##        return self.base_model(best_n)

        self.X, self.lengths = self.hwords[self.this_word]
        if best_state:
        	best_model = self.base_model(best_state)
        	return best_model
        else:
        	return None