import nltk


# gets noun phrases
# TODO take care of NNP.
def get_nouns(document):
    data = []
    sentence = nltk.word_tokenize(document)
    sentence = nltk.pos_tag(sentence)
    is_plural = False
    for _node in sentence:
        if _node[1] == "NNS":
            is_plural = True
        if _node[1] in ["NN", "NNS"]:
            data.append(_node[0])
    return filter_nouns(data), is_plural


# TODO
def filter_nouns(nouns):
    return nouns

