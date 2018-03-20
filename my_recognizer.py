import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
        {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
        both lists are ordered by the test set word_id
        probabilities is a list of dictionaries where each key is a word and value is Log Likelihood
            [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
             ]
        guesses is a list of the best guess words ordered by the test set word_id
            ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    ##raise NotImplementedError

    for test_idx in test_set.get_all_sequences().keys():
        X, length = test_set.get_item_Xlengths(test_idx)
        
        d = dict()

        for word, model in models.items():

            try:
                d[word] = model.score(X, length)
            except Exception:
                d[word] = float("-inf")
            

        probabilities.append(d)

        best_guess = max(d.items(), key=lambda x:x[1])[0]
        guesses.append(best_guess)

    return probabilities, guesses
