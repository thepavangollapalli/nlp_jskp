import nltk
import sys
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger

stanford_ner_jar = 'stanford-ner-2018-10-16/stanford-ner.jar'
stanford_ner_model = 'stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
stanford_pos_jar = 'stanford-postagger-2018-10-16/stanford-postagger.jar'
stanford_pos_model = 'stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'

ner_tagger = StanfordNERTagger(stanford_ner_model, stanford_ner_jar, encoding="utf8")
pos_tagger = StanfordPOSTagger(stanford_pos_model, stanford_pos_jar, encoding="utf8")

#class to store all sentences
class Sentences:
    def __init__(self, passage):
        self.sentences = nltk.tokenize.sent_tokenize(passage)
        self.tokenized = self.tokenize(self.sentences)
        self.ner = self.ner(self.tokenized)
        self.pos = self.pos(self.tokenized)

    def tokenize(self, sentences):
        tokens = list()
        for s in self.sentences:
            tokens.append(nltk.word_tokenize(s))
        return tokens

    def ner(self, tokens):
        nerTags = ner_tagger.tag_sents(tokens)
        # print(nerTags)
        return nerTags

    def pos(self, tokens):
        posTags = list()
        for s in tokens:
            posTags.append(nltk.pos_tag(s))
        return posTags

class Sentence:
    def __init__(self, s, n):
        self.currSent = s.sentences[n]
        self.tokenized = s.tokenized[n]
        self.ner = s.ner[n]
        self.pos = s.pos[n]
        self.len = len(self.tokenized)

    def __repr__(self):
        return str(self.ner)



def what(sentence):
    q = list()
    found = False
    for i in range(1, len(sentence.tokenized) - 1):
        if found:
            if sentence.pos[i][1] in ['NN', 'NNP', 'NNS', 'PRP'] and sentence.tokenized[i-1] == 'and':
                q.pop()
                break
            q.append(sentence.tokenized[i])
        if sentence.pos[i][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'] and sentence.ner[i-1][1] != 'PERSON' and sentence.pos[i-1][1] in ['NN','NNP','NNS','PRP'] and not found:
            q.append("What")
            q.append(sentence.tokenized[i])
            found = True
    if(len(q) > 0):
        q.append("?")
    return q

def who(sentence):
    q = list()
    found = False
    for i in range(1, sentence.len - 1):
        if not found and sentence.pos[i][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
            q.append("Who")
            q.append(sentence.tokenized[i])
            found = True
        if found:
            if sentence.pos[i][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'] and sentence.ner[i-1][1] in ['PERSON', 'ORGANIZATION']:
                q.pop()
                q.append("Who")
                q.append(sentence.tokenized[i])
            else:
                q.append(sentence.tokenized[i])
    q.append("?")
    return q

if __name__ == '__main__':
    txtFile = sys.argv[1]
    n = int(sys.argv[2])
    sentences = []
    result = list()
    counter = 0
    with open(txtFile, 'r') as f:
            content = f.read()
            f.close()
    sentences = Sentences(content)
    for i in range(0, len(sentences.sentences)):
        result.append(' '.join(what(Sentence(sentences, i))))
    for q in result:
        if(q != ""):
            if(counter<n):
                print("Q" + str(counter) + ": " + q)
                counter += 1
            else:
                break
    while(counter<n):
        print("Q" + str(counter) + ": " + '')
        counter += 1




