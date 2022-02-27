"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submission executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file.

Best of Luck!
"""
from collections import defaultdict, Counter

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import the Twitter corpus and LgramModel
from nltk_model import *  # See the README inside the nltk_model folder for more information

# Import the Twitter corpus and LgramModel
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy, tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy, ", ".join(tweet)))


def compute_accuracy(classifier, data):
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :type data: list(tuple(list(any), str))
    :param data: A list with tuples of the form (list with features, label)
    :rtype float
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct / len(data)


def apply_extractor(extractor_f, data):
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :type extractor_f: (str, str, str, str) -> list(any)
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :type data: list(tuple(str))
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :rtype list(tuple(list(any), str))
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """

    def __init__(self, classifier_class, train_features, **kwargs):
        """

        :type classifier_class: a class object of nltk.classify.api.ClassifierI
        :param classifier_class: the kind of classifier we want to create an instance of.
        :type train_features: list(tuple(list(any), str))
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype dict(any, int)
        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype str
        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n=10):
        self.classifier_obj.show_most_informative_features(n)


# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1 [7 marks]
def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''
    # subset the corpus to only include all-alpha tokens,
    # converted to lower-case (_after_ the all-alpha check)
    #     corpus_tokens = [w.lower() for w in corpus.words if w.isalpha()]
    corpus_tokens = []
    for word in corpus.words():
        if word.isalpha():
            corpus_tokens.append(word.lower())

    lm = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

    # Return a smoothed (using the default estimator) padded bigram
    # letter language model
    return lm


# Question 2 [7 marks]
def tweet_ent(file_name, bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''
    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens, converted
    # to lowercase
    list_of_tweets = xtwc.sents(file_name)
    cleaned_list_of_tweets = []
    for tweet in list_of_tweets:
        cleaned_tweet = []
        for word in tweet:
            if word.isalpha():
                cleaned_tweet.append(word.lower())
        if len(cleaned_tweet) >= 5:
            cleaned_list_of_tweets.append(cleaned_tweet)

    with open("cleaned tweets.txt", "w") as output:
        output.write(str(cleaned_list_of_tweets))

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy
    ent_tweets = []
    for tweet in cleaned_list_of_tweets:
        av_ent = 0
        for word in tweet:
            word_ent = bigram_model.entropy(word, pad_left=True, pad_right=True, perItem=True)
            av_ent += word_ent
        ent_tweets.append((av_ent / len(tweet), tweet))
    return sorted(ent_tweets, key=lambda x: x[0])


# Question 3 [8 marks]
def open_question_3():
    '''
    Question: What differentiates the beginning and end of the list
    of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""We can see the tweets at the top of the list are in English,at an entropy
    around 2.5,and the tweets at the end of the list are in Japanese,at around 17.5.This is a big range,going from a
    good entropy of 2.5,compared to the human 1.3,to 7 times that.This makes sense, as our model was trained in
    English on the Brown corpus,not Japanese,so it has learned bigrams of letters in English,and can predict English
    words,whereas it has never seen Japanese characters so can't predict what comes next.""")[0:500]


# Question 4 [8 marks]
def open_question_4() -> str:
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""Still contain emoji parts,e.g :D ->d.These will
    only be 1 letter,so we can fix this by removing all words<1,leaving a and I.
    Has chars not in the Latin alphabet,e.g Japanese.Can remove these by only keeping words containing ascii chars 97-122.
    Has words in foreign languages,e.g Spanish.Can use a language detection package e.g langdetect,to identify
    and remove non-English words.
    Has misspellings,can use another package e.g TextBlob to correct them.
    Has abbreviations/slang,e.g lol.Can try to create helper function to expand abbreviations and standardize slang
    okayy->ok, ty->thank you""")[0:500]


# Question 5 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweets and their letter bigram entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average letter bigram entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)), list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             non-English tweets and entropies
    '''
    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[
                                         :np.math.floor(len(list_of_tweets_and_entropies) * 0.9)]

    # Extract a list of just the entropy values
    list_of_entropies = [ent_tweet[0] for ent_tweet in list_of_ascii_tweets_and_entropies]

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = mean + standard_deviation
    list_of_not_English_tweets_and_entropies = [ent_tweet for ent_tweet in list_of_ascii_tweets_and_entropies if
                                                ent_tweet[0] > threshold]
    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return mean, standard_deviation, list_of_ascii_tweets_and_entropies, list_of_not_English_tweets_and_entropies


# Question 6 [15 marks]
def open_question_6():
    """
    Suppose you are asked to find out what the average per word entropy of English is.
    - Name 3 problems with this question, and make a simplifying assumption for each of them.
    - What kind of experiment would you perform to estimate the entropy after you have these simplifying assumptions?
       Justify the main design decisions you make in your experiment.
    :rtype: str
    :return: your answer [1000 chars max]
    """
    return inspect.cleandoc("""1:The question says English in general.We cannot get a data set of the whole English
    language.To simplify this,we can pick a dataset to represent it and exemplify its different forms e.g
    speaking/written,formal/informal,modern/historical.
    2:The question doesn't specify what probability is being used to calculate entropy,humans can be used to form a
    prediction,or we can use a model e.g ngrams.
    3:The question says entropy,but we want the entropy of our model's prediction relative to the true probability, this
    is cross entropy.We cannot know the true probability,as again it would require the entire English language,so we
    estimate cross entropy using the what occurs in our corpus.
    To carry out this experiment,I would use the BNC as a corpora,as it contains an extensive,wide range of English,
    cleaned of punctuation and lowercased and train a ngram model using backoff to weight different values of n,where
    exact values could be experimented with.As the BNC is large it also enables us to split the data into multiple
    rounds of training and testing to get a good estimate value for the entropy.
    """)[:1000]


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 7 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data, alpha):
        """
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data):
        """
        Compute the set of all possible features from the (training) data.
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :rtype: set(any)
        :return: The set of all features used in the training data for all classes.
        """
        vocab = []
        for doc in data:
            for feat in doc[0]:
                if feat not in vocab:
                    vocab.append(feat)

        return vocab

    @staticmethod
    def train(data, alpha, vocab):
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :type data: list(tuple(list(any), str))
        :param data: A list of tuples ([f1, f2, ... ], c) with the first element
                     being a list of features and the second element being its class.

        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing

        :type vocab: set(any)
        :param vocab: The set of all features used in the training data for all classes.


        :rtype: tuple(dict(str, float), dict(str, dict(any, float)))
        :return: Two dictionaries: the prior and the likelihood (in that order).
        We expect the returned values to relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0
        classes = {}
        for doc in data:
            if doc[1] not in classes:
                classes[doc[1]] = [1, np.zeros(len(vocab))]
            else:
                classes[doc[1]][0] += 1
            for i in range(len(vocab)):
                if vocab[i] in doc[0]:
                    classes[doc[1]][1][i] += 1

        priors = {}
        likelihoods = {}
        for c, feature_counts in classes.items():
            priors[c] = feature_counts[0] / len(data)
            c_likelihoods = {}
            for i in range(len(vocab)):
                c_likelihoods[vocab[i]] = (feature_counts[1][i] + alpha) / (sum(feature_counts[1]) + alpha * len(vocab))
            likelihoods[c] = c_likelihoods

        return priors, likelihoods

    def prob_classify(self, d):
        """
        Compute the probability P(c|d) for all classes.
        :type d: list(any)
        :param d: A list of features.
        :rtype: dict(str, float)
        :return: The probability p(c|d) for all classes as a dictionary.
        """
        d_probs = {}
        prob_d = 0
        for c, c_likelihoods in self.likelihood.items():
            feats_given_c = 1
            for feat in d:
                if feat in c_likelihoods:
                    feats_given_c *= c_likelihoods[feat]
            d_probs[c] = self.prior[c] * feats_given_c
            prob_d += self.prior[c] * feats_given_c

        for c in d_probs.keys():
            d_probs[c] = d_probs[c] / prob_d

        return d_probs

    def classify(self, d):
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :type d: list(any)
        :param d: A list of features.
        :rtype: str
        :return: The most likely class.
        """
        d_probs = self.prob_classify(d)
        return max(d_probs, key=d_probs.get)


# Question 8 [10 marks]
def open_question_8() -> str:
    """
    How do you interpret the differences in accuracy between the different ways to extract features?
    :rtype: str
    :return: Your answer of 500 characters maximum.
    """
    return inspect.cleandoc("""Preposition is most informative, but all words are informative in predicting attachment
    Logistic regression better here""")[:500]


# Feature extractors used in the table:
# see your_feature_extractor for documentation on arguments and types.
def feature_extractor_1(v, n1, p, n2):
    return [v]


def feature_extractor_2(v, n1, p, n2):
    return [n1]


def feature_extractor_3(v, n1, p, n2):
    return [p]


def feature_extractor_4(v, n1, p, n2):
    return [n2]


def feature_extractor_5(v, n1, p, n2):
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 9.1 [5 marks]
# from sklearn.feature_extraction.text import TfidfVectorizer
# tf=TfidfVectorizer()
# # text_tf= tf.fit_transform(data['Phrase'])
#
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.tokenize import RegexpTokenizer
# #tokenizer to remove unwanted elements from out data like symbols and numbers
# token = RegexpTokenizer(r'[a-zA-Z0-9]+')
# cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
# # text_counts= cv.fit_transform(data['Phrase'])

def pos_tag(word):
    tag = nltk.pos_tag([word])
    return tag[0][1]

def stem(word):
    ps = nltk.stem.porter.PorterStemmer()
    return ps.stem(word)


def your_feature_extractor(v, n1, p, n2):
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.
    :type v: str
    :param v: The verb.
    :type n1: str
    :param n1: Head of the object NP.
    :type p: str
    :param p: The preposition.
    :type n2: str
    :param n2: Head of the NP embedded in the PP.
    :rtype: list(any)
    :return: A list of features produced by you.
    """
    # features = [("v", v), ("n1", n1), ("p", p), ("n2", n2), ("p n2", (p, n2)), ("n1 p", (n1, p)), ("v p", (v, p)), ("v n2", (v, n2)), ("v n1 n2", (v, n1, n2))]
    # pos_v = pos_tag(v)
    # v = stem(v)
    # n1 = stem(n1)
    # n2 = stem(n2)
    features = [("v", v), ("n1", n1), ("p", p), ("n2", n2), ("p n1", (p, n1)), ("p n2", (p, n2)), ("v p", (v, p)), ("v n2", (v, n2)),
                ("p n1 n2", (p, n1, n2)), ("len n1 n2", len(n1 + n2)), ("n1 proper", n1[0].isupper()), ("n2 proper", n2[0].isupper())]
    return features


# Question 9.2 [10 marks]
def open_question_9():
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick 3 examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.
    :rtype: str
    :return: Your answer of 1000 characters maximum.
    """
    return inspect.cleandoc("""1.
    2.
    3. Not in top 30, but 0.2% accuracy increase, length of n1+n2. Do not know why informative, only guess is that when
    n1 and n2 are both numbers, so shorter, it is more likely to be a verb.""")[:1000]


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""


def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_4, answer_open_question_3, answer_open_question_6, \
        answer_open_question_8, answer_open_question_9
    global ascci_ents, non_eng_ents

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features

    print("*** Part I***\n")

    print("*** Question 1 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 2 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:20]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-20:]
    ppEandT(worst10_ents)

    print("*** Question 3 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    print("*** Question 4 ***")
    answer_open_question_4 = open_question_4()
    print(answer_open_question_4)

    print("*** Question 5 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Tweets considered non-English')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

    print("*** Question 6 ***")
    answer_open_question_6 = open_question_6()
    print(answer_open_question_6)

    print("*** Part II***\n")

    print("*** Question 7 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 8 ***")
    answer_open_question_8 = open_question_8()
    print(answer_open_question_8)

    # This is the code that generated the results in the table of the CW:
    #
    # A single iteration of suffices for logistic regression for the simple feature extractors.

    # extractors_and_iterations = [feature_extractor_1, feature_extractor_2, feature_extractor_3, feature_extractor_4, feature_extractor_5]
    #
    # print("Extractor    |  Accuracy")
    # print("------------------------")

    # for i, ex_f in enumerate(extractors_and_iterations, start=1):
    #     training_features = apply_extractor(ex_f, ppattach.tuples("training"))
    #     dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    #
    #     a_logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=6, trace=0)
    #     lr_acc = compute_accuracy(a_logistic_regression_model, dev_features)
    #     print(f"Extractor {i}  |  {lr_acc*100}")
    #
    # for i, ex_f in enumerate(extractors_and_iterations, start=1):
    #     training_features = apply_extractor(ex_f, ppattach.tuples("training"))
    #     dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    #
    #     naive_bayes_model = NaiveBayes(training_features, 0.1)
    #     nb_acc = compute_accuracy(naive_bayes_model, dev_features)
    #     print(f"Extractor {i}  |  {nb_acc*100}")

    print("*** Question 9 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc * 100}")

    answer_open_question_9 = open_question_9()
    print("Answer to open question:")
    print(answer_open_question_9)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
