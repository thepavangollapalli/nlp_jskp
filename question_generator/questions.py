#!/usr/bin/env python -W ignore::DeprecationWarning

#Mute depreciation warning from scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import nltk
import sys
import copy
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger

stanford_ner_jar = '/home/coreNLP/stanford-ner-2018-10-16/stanford-ner.jar'
stanford_ner_model = '/home/coreNLP/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
stanford_pos_jar = '/home/coreNLP/stanford-postagger-full-2018-10-16/stanford-postagger.jar'
stanford_pos_model = '/home/coreNLP/stanford-postagger-full-2018-10-16/models/english-bidirectional-distsim.tagger'
stanford_when_model = '/home/coreNLP/stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz'

ner_tagger = StanfordNERTagger(stanford_ner_model, stanford_ner_jar, encoding="utf8")
pos_tagger = StanfordPOSTagger(stanford_pos_model, stanford_pos_jar, encoding="utf8")
when_tagger = StanfordNERTagger(stanford_when_model, stanford_ner_jar, encoding="utf8")

#class to store all sentences
class Sentences:
    def __init__(self, passage):
        self.sentences = nltk.tokenize.sent_tokenize(passage)
        self.tokenized = self.tokenize(self.sentences)
        self.ner = self.ner(self.tokenized)
        self.pos = self.pos(self.tokenized)
        self.whenTags = self.whenTags(self.tokenized)
        self.overall_questions = self.get_overall_questions()

    def tokenize(self, sentences):
        tokens = list()
        for s in self.sentences:
            tokens.append(nltk.word_tokenize(s))
        return tokens

    def ner(self, tokens):
        nerTags = ner_tagger.tag_sents(tokens)
        return nerTags

    def pos(self, tokens):
        pos_tags = list()
        for s in tokens:
            pos_tags.append(nltk.pos_tag(s))
        return pos_tags

    def whenTags(self, tokens):
        whenTags = when_tagger.tag_sents(tokens)
        return whenTags

    def get_overall_questions(self):
        result = list()
        for i in range(0, len(self.sentences)):
            sentence = Questions(self, i)
            for question in sentence.sentence_questions:
                result.append(question)
        return result

class Questions:
    def __init__(self, s, n):
        self.currSent = s.sentences[n]
        self.tokenized = s.tokenized[n]
        self.ner = s.ner[n]
        self.pos = s.pos[n]
        self.whenTags = s.whenTags[n]
        self.len = len(self.tokenized)
        self.proN = self.proN(self.ner)
        self.sentence_questions = self.get_questions()

    def __repr__(self):
        return str(self.ner)

    def create_question(self, q):
        if(len(q) == 0):
            return q
        return " ".join(q) + "?"

    def proN(self, ner_tags):
        pn = dict()
        for (w, t) in ner_tags:
            if t == "PERSON":
                if(w not in pn):
                    pn[w] = 1
                else:
                    pn[w] += 1
        maxWord = ""
        maxCount = 0
        for k in pn:
            if(maxCount < pn[k]):
                maxCount = pn[k]
                maxWord = k
        return maxWord

    def get_questions(self):
        return [self.what(), self.who(), self.when(), self.yesNo(), self.where(), self.why()]

    def what(self):
        noun_tags = {'NN', 'NNP', 'NNS', 'PRP'}
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'DT', 'MD'}

        q = list()
        found = False
        for i in range(1, len(self.tokenized) - 1):
            if found:
                if self.pos[i][1] in noun_tags and self.tokenized[i-1] == 'and':
                    q.pop()
                    break
                q.append(self.tokenized[i])
            if not found and self.pos[i][1] in verb_tags and self.ner[i-1][1] != 'PERSON' and self.pos[i-1][1] in noun_tags:
                q.append("What")
                q.append(self.tokenized[i])
                found = True
        return self.create_question(q)

    def who(self):
        verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'}
        ner_subjects = {'PERSON', 'ORGANIZATION'}

        q = list()
        found = False
        for i in range(1, self.len - 1):
            if found:
                if self.pos[i][1] in verb_tags and self.ner[i-1][1] in ner_subjects:
                    q.pop()
                    break
                q.append(self.tokenized[i])
            if not found and self.pos[i][1] in verb_tags:
                q.append("Who")
                q.append(self.tokenized[i])
                found = True
        return self.create_question(q)

    def when(self):
        q = list()
        found = False
        time = {'today', 'tomorrow', 'yesterday'}
        otherTime = {"by", "after", "before", "during", "when", "while", 'on', 'in', 'last'}
        lastTime = {"after", "before", "during", "when", "while"}
        for i in range(1, self.len ):
            if not found:
                if ((self.ner[i-1][1] == 'PERSON' and i < self.len - 1) and 
                   ((self.pos[i][1] == 'MD' and self.pos[i+1][1] == 'VB') or
                    (self.pos[i][1] == 'VBD' and self.pos[i+1][1] == 'VBG'))):
                    q.append("When")
                    q.append(self.tokenized[i])
                    q.append(self.tokenized[i-1])
                    found = True
                else:
                    if (self.ner[i-1][1] == 'PERSON' and i < self.len and
                        (self.pos[i][1] in ['VBZ', 'VBD'])):
                        if self.pos[i][1] == 'VBZ':
                            verb = "will"
                        else:
                            verb = "did"
                        new = self.tokenized[i]
                        q.append("When")
                        q.append(verb)
                        q.append(self.tokenized[i-1])
                        q.append(new)
                        found = True
            else:
                if (self.whenTags[i][1] in ['DATE', 'TIME'] or self.tokenized[i]
                    in time):
                    if self.tokenized[i-1] in otherTime:
                        q.pop()
                    return self.create_question(q)
                elif (self.tokenized[i] in lastTime):
                    return self.create_question(q)
                elif i != self.len - 1:
                    q.append(self.tokenized[i])
                    return self.create_question(q)
                else:
                    return self.create_question(q)

    def yesNo(self):
        posTag = copy.deepcopy(self.pos)
        flip = False
        indication = {"is", "are", "does", "were", "can", "were", "will", 
                            "has", "had", "have", "could", "would", "should"}
        verbs = list()
        q = list()
        for i in range(len(posTag)):
            word = posTag[i][0]
            tag = posTag[i][1]
            if word in indication: 
                verbs.append(i)
        if len(verbs) == 0:
            posTag.insert(0, ('did', 'VBD'))
            flip = True
        else:
            posTag.insert(0, posTag.pop(verbs[0]))
        q.append(posTag[0][0].title())
        for i in range(len(posTag)-1):
            w = posTag[i+1][0]
            wnew = ""
            if i == 0:
                if posTag[1][1] == "PRP":
                    w = self.proN
                elif posTag[1][1] == "NNP":
                    w = w.lower()
            if flip:
                try: 
                    wnew = en.verb.present(w)
                except: 
                    wnew = w
            else:
                wnew = w
            q.append(wnew)
        q.pop()
        return self.create_question(q)

    def where(self):
        nerTag = copy.deepcopy(self.ner)
        sentence = self.tokenized
        posTag = copy.deepcopy(self.pos)
        q = list()
        verbs = {"VBD", "VBZ"}
        nouns = {"NN", "NNP", "PRP"}
        wh = {"at", "in"}
        sub = False
        loc = False
        subI = 0 
        locI = 0
        vals = dict()
        for w in range(len(sentence)):
            if(loc):
                if (posTag[w][1] in verbs):
                    vals["verb"] = (sentence[w], w)
                    locI = w  
                    break
            if(sub):
                if (posTag[w][1] in verbs):
                    vals["verb"] = (sentence[w], w)
                if (nerTag[w][1] == "LOCATION" and sentence[w-1] in wh):
                    subI = w - 1 
            if(not sub and not loc): 
                if(nerTag[w][1] == "LOCATION"):
                    loc = True 
                elif(posTag[w][1] in nouns):
                    sub = True
                    vals["sub"] = (sentence[w], w)
        if subI == 0 and locI == 0:
            return q
        elif sub:
            w1 = vals["verb"][1]
            bc1 = ''.join(str(e) for e in sentence[w1+1:subI])
            bc1 = bc1.replace(",", " ")

            w0 = vals["sub"][1]
            bc0 = ''.join(str(e) for e in sentence[w0-1:w1])
            bc0 = bc0.replace(",", " ")
            if (vals["verb"][0] in {"is", "was"}):
                q.append("Where")
                q.append(vals["verb"][0])
                q.append(bc0)
                q.append(bc1)
            elif(nerTag[vals["verb"][1]][1] == "VBD"):
                q.append("Where did")
                q.append(vals["sub"][0])
                q.append(en.verb.present(vals["verb"][0]))
                q.append(bc1)
            else:
                q.append("Where does")
                q.append(vals["sub"][0])
                q.append(en.verb.present(vals["verb"][0]))
                q.append(bc1)
        elif loc:
            w = vals["verb"][1]
            bc = ''.join(str(e) + " " for e in sentence[locI:])
            bc = bc.replace(".", "")
            bc = bc.replace("!", "")
            q.append("Where")
            q.append(bc)
        return self.create_question(q)

    def why(self):
        n = self.ner
        p = self.pos
        vI = None
        sI = None
        sentence = self.tokenized
        nouns = {"NN", "NNP", "PRP", "NNS"}
        verbs = {"VBD", "VBZ", "VBP"}
        vs = False
        ss = False
        done = 0
        nerTag = copy.deepcopy(n)
        posTag = copy.deepcopy(p)
        dt = None
        q = list()
        if("because" in sentence or "since" in sentence):
            for w in range(self.len):
                if sentence[w] in {"because", "since"}:
                    done = w
                if posTag[w][1] in nouns and not ss:
                    sI = w
                    ss = True
                    if(w-1 >= 0):
                        if(posTag[w-1][1] == "DT"):
                            dt = sentence[w-1]
                        else:
                            dt = None
                if posTag[w][1] in verbs and not vs:
                    vs = True
                    vI = w
            if sI != None and vI!=None:
                bc = ''.join(str(e) + " " for e in sentence[vI+1:done])
                bc = bc.replace(".", "")
                bc = bc.replace("!", "")
                q.append("Why")
                q.append(sentence[vI])
                if(dt != None):
                    q.append(dt)
                q.append(sentence[sI])
                q.append(bc)
        return self.create_question(q)

def clean(result):
    clean = list()
    for q in result:
        if(q != [] and q!= None and q!=""):
            clean.append(q)
    return clean

if __name__ == '__main__':
    txtFile = sys.argv[1]
    n = int(sys.argv[2])
    with open(txtFile, 'r', encoding="latin1", errors='surrogateescape') as f:
            content = f.read()
            f.close()
    sentences = Sentences(content)
    counter = 0
    result = sentences.overall_questions
    result = clean(result)
    while(counter<n):
        if(result == []):
            print("")
            counter += 1
        else:
            print(result[counter%len(result)])
            counter += 1

