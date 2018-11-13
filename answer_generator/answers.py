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
            file.close
        return nltk.tokenize.sent_tokenize(ss.replace("\n", " . "))
