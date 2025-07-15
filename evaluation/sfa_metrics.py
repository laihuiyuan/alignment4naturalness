# -*- coding:utf-8 _*-

# based on Esther's sfa.py, calculating PTF, CDU and SynTTR

import spacy
import argparse
from scipy import spatial
from nltk import FreqDist
from collections import defaultdict


def create_lemma_freq_dict(spacy_model, lines):
    """
    Create FreqDist object for lemmas of adjectives, nouns and verbs that occur in the text
    """
    words = []
    for line in lines:
        object = spacy_model(line)
        lemmas = [token.lemma_ for token in object]
        pos_tags = [token.pos_ for token in object]
        for l, p in zip(lemmas, pos_tags):
            if p in ["ADJ", "NOUN", "VERB"]:
                words.append(l)

    return FreqDist(words)


def create_bi_dict(dicttextfile):
    """
    Create a dictionary of English words with possible Dutch translations
    """

    bi_dict = dict()

    all = ""
    for line in dicttextfile:
        if "/" in line:
            all = all + "@" + line
        else:
            all = all + "#" + line
    test_all = all

    for string in test_all.split("@"):
        english = string.split("/")[0].rstrip()
        s = []
        dutch_part = string.split("/")[-1]
        dutch_part = dutch_part.replace(".", ",")
        for item in dutch_part.split(","):
            n = ""
            for c in item:
                if c in "abcdefghijklmnopqrstuvwqxyzĳöëïäü ":
                    n += c
            n = n.replace("ĳ", "ij")
            if n:
                s.append(n.strip())
        bi_dict[english] = set(s)

    return bi_dict


def ptf(vectors):
    """
    Compute average primary translation prevalence over source words
    """
    ptfs = []
    for vector in vectors:
        if sum(vector) > 1:  # word is translated to Dutch
            ptfs.append(max(vector)/sum(vector))

    return sum(ptfs) / len(ptfs)


def cdu(vectors):
    """
    Compute average cosine distance between a vector and a distribution
    where 'each translation option would be equally prevalent'
    """
    c_dists = []
    for vector in vectors:
        if sum(vector) > 1:  # word is translated to Dutch

            # Construct 'equal' vector
            average_vec_val = round(sum(vector) / len(vector))
            equal_vec = [average_vec_val] * len(vector)

            # Compute cosine distance
            cos_dist = 1 - spatial.distance.cosine(vector, equal_vec)
            c_dists.append(cos_dist)

        return sum(c_dists) / len(c_dists)


def syn_ttr(types, lines):
    """
    Compute TTR where the types are only the translation options from the bidict and the
    tokens are appearances of these types in the running text
    """

    relevant_types = set()
    token_count = 0
    for line in lines:
        for word in line.split():
            if word in types:
                relevant_types.add(word)
                token_count += 1

    return len(relevant_types) / token_count


def compute_sfa(eval_data, dict_data):

    # Count frequencies for lemmas in texts
    sp = spacy.load("nl_core_news_lg")
    lemma_freqs = create_lemma_freq_dict(sp, eval_data)
    
    # Retrieve translation options
    bi_dict = create_bi_dict(dict_data)

    # Count occurrences of translation options
    count_dict = defaultdict(list)
    for en, nl_syns in bi_dict.items():
        for synonym in nl_syns:
            count = lemma_freqs[synonym]
            count_dict[en].append((synonym, count))


    # Construct vector
    vectors = []
    for values in count_dict.values():
        vectors.append([v[1] for v in values])

    # Get translation types
    types = set([element for items in bi_dict.values() for element in items])


    return syn_ttr(types, eval_data), ptf(vectors), cdu(vectors)

