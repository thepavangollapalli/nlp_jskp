import os
import re
import sys
import nltk
import math
import numpy
import timex
import string
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser

if __name__ == '__main__':
    Preprocess(questionDoc=sys.argv[1], document=sys.argv[2]).run()

class Preprocess(object):
    def __init__(self, questionDoc, document):
        self.questionDoc = questionDoc
        self.document = document
        self.questions = []
        self.wordWeights = dict()
        self.porterStem = PorterStemmer()

    def addQuestions(self):
        with open(self.questionDoc, 'r') as file:
            qs = file.read()
            file.close()
        for q in qs.splitlines():
            self.questions.append(q)

    def processSentences(self):
        with open(self.document, 'r', encoding = "ISO-8859-1") as file:
            ss = file.read()
            file.close()
        return nltk.tokenize.sent_tokenize(ss.replace("\n", " . "))

    def answerBinary(self,question,sentence):
        answer = "Yeet"
        question_tags = nltk.pos_tag(ntlk.word_tokenize(question))
        nouns_verbs = []
        for (word,tag) in question_tags:
            if("NN" in tag or "JJ" in tag):
                nouns_verbs.append(word)
        sentence_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
        negations = ["not","doesn't","isn't","wasn't","didn't"]
        nv_count = 0 #count of recognized nouns and verbs from question
        negate = False
        for (word,tag) in sentence_tags:
            if(word in nouns_verbs):
                nv_count+=1
            if(word in negations):
                negate = True
        #if the proportion of recognized nouns/verbs is too low or there is a negation in the sentence we say no
        if float(nv_count)/len(nouns_verbs) <= .34 or negate:
            answer = "Nope"
        return answer

