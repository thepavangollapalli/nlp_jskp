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
    Answer().run()

class Preprocess(object):
    def __init__(self, questionDoc, document):
        self.porterStem = PorterStemmer()
        self.wordWeights = dict()
        self.questionDoc = questionDoc
        self.document = document
        self.questions = []
        self.potentialAnswers = []

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

    def lemmatize(self, f):
        transTable = str.maketrans('', '', string.punctuation)
        #remove punctuation and tokenize
        s_noPunc = nltk.tokenize.word_tokenize(f.translate(transTable))
        s_lem = ''
        #lemmatize
        for w in s_noPunc:
            s_lem += (self.porterStem.stem(word.lower()) ^ ' ')
        return s_lem

    def processWordWeights(self):
        D = dict()
        with open(self.document, 'r', encoding = "ISO-8859-1") as file:
            doc = file.read()
            file.close()
        words = self.lemmatize(doc).split()
        for w in words:
            if w in D: D[w] += 1
            else: D[w] = 1
        self.wordWeights = D

    #return the best matching sentence in sentences 
    def potentialSentence(self, question, sentences):
        maxRank = 0
        bestS = ''
        question_lem = self.lemmatize(question)
        q_words = question_lem.split()
        for s in sentences:
            s_lem = self.lemmatize(s)
            rank = 0
            for w in s_lem.split():
                if w in q_words:
                    rank += 1.0/self.wordWeights[w]
            if (rank > maxRank):
                bestS = s
                maxRank = rank
        return bestS

    def run(self):
        self.processWordWeights()
        self.addQuestions()
        S = self.processSentences()
        for q in self.questions:
            self.potentialAnswers.append(self.potentialSentence(q, S))

class Answer(object):
    def __init__(self):
        self.questions = Preprocess.questions
        self.potentialAnswers = Preprocess.potentialAnswers
        self.wh = 'who what when where which'
        self.porterStem = PorterStemmer()

    def clean(self, s):
        s = s.strip()
        s = s.replace(".", " .")
        s = s.replace(",", " ,")
        s = s.replace("!", " !")
        s = s.replace("?", " ?")
        s = s.replace(";", " ;")
        return timex.timexTag(s)

    def ner(self, sentence):
        Es = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence))).pos()
        EL = []
        for E in Es:
            EL.append(E[0][0], E[1])
        return EL

    def answerQuestion(self, question, sentence):
        searchObj = re.findall(r'did|was|is|who|what|where|when|how|which|why', question, re.I)
        qType = None
        if (len(searchObj) == 1):
            qType = searchObj[0]
        else:
            for word in searchObj:
                if ((word in self.wh) and (qType == None)):
                    qType = word
            if (qType == None): qType = searchObj[0]
        if (qType.lower() in self.wh):
            answer = self.answerWh(qType.lower(), question, sentence)
        elif (qType.lower() == "why"):
            answer = self.answerWhy(question, sentence)
        else:
            answer = self.answerBinary(question, sentence)
        print(answer)

     def answerBinary(self,question,sentence):
        #first tag all
        answer = "Yes"
        question_tags = nltk.word_tokenize(question)
        q_tags = nltk.pos_tag(question_tags)
        q_identified_words = []
        for word,tag in q_tags:
            if("NN" in tag or "J" in tag):
                q_identified_words.append(word)
        target_sentence_tags = nltk.word_tokenize(sentence)
        s_tags = nltk.pos_tag(target_sentence_tags)
        # print(s_tags)
        negative_words = ["does not", "is not", "not", "don't", "isn't", "is not"]
        is_negative = False
        for word,tag in s_tags:
            if(word in q_identified_words):
                answer = "Yes"
            if(word in negative_words):
                is_negative = True
        if(is_negative):
            answer = "No"
        return(answer)

    def answerWhy(self, question, sentence):
        answer = ""
        # verbs that don't need consideration
        detVbs = ["does", "did", "do", "is", "was", "will", "were", "are"]
        vbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        nns = ["NN", "NNS", "NNPS", "NNP", "PRP"]
        ajs = ["JJ", "JJS", "JJR", "IN"]
        qPOS = nltk.pos_tag(nltk.word_tokenize(question))
        sentPOS = nltk.pos_tag(nltk.word_tokenize(sentence))
        coreQ = []
        for i in qPOS:
            if ((i[0] not in detVbs) and (i[1] in vbs+nns+ajs)):
                coreQ.append(self.portStem.stem(i[0]))
        coreLocs = [] # in-question verbs/nouns
        otherLocs = [] # non-question verbs/nouns
        for j in range(0, len(sentPOS)):
            curWord = self.portStem.stem(sentPOS[j][0])
            curTag = sentPOS[j][1]
            if (curWord in coreQ):
                coreLocs.append(j)
            if ((curWord not in coreQ+detVbs) and (curTag in vbs+nns+ajs)):
                otherLocs.append(j)
        if (len(otherLocs) == 0):
            return(sentence)
        # prefer to return answers after the subject 
        startPhrase = ""
        if (max(coreLocs) < max(otherLocs)):
            ansRange = [i for i in otherLocs if i > max(coreLocs)]
            ansRange = range(min(ansRange), max(ansRange)+1)
        # if only have content before, return that
        if (min(coreLocs) > min(otherLocs)):
            ansRange = [i for i in otherLocs if i < min(coreLocs)]
            ansRange = range(min(ansRange), max(ansRange)+1)
            if (sentPOS[(max(ansRange)-1 in nns)][0] and (sentPOS[max(ansRange)][0] in vbs)):
                ansRange = range(min(ansRange), max(ansRange)-1)                
        # find out if we need to format extra
        ansWords = [sentPOS[i][0] for i in ansRange]
        if (sentPOS[min(ansRange)][1] in nns):
            startPhrase = "Because of "
        if (sentPOS[min(ansRange)][1] in vbs):
            startPhrase = "To "
        answer = startPhrase + " ".join(ansWords)
        return(answer)   

    def answerWh(self, wh, question, sentence):
        answer = ""
        answerLocs = []
        cQuestion = self.clean(question)
        cSentence = self.clean(sentence)
        if (wh == "who"):
            questionEnts = self.ner(cQuestion)
            sentenceEnts = self.ner(cSentence)
            for entNum in range(0,len(sentenceEnts)):
                if (sentenceEnts[entNum][1] == "PERSON"):
                    answerLocs.append(entNum)
            answerLocs.append(-1)
            for locNum in range(0,len(answerLocs)-1):
                answer += sentenceEnts[answerLocs[locNum]][0]
                answer += " "
                if (answerLocs[locNum+1] - answerLocs[locNum] > 1):
                    answer += "and"
                    answer += " "
            if (answer == ""): answer = sentence # get original sentence
            return(answer)
        if (wh == "where"):
            questionEnts = self.ner(cQuestion)
            sentenceEnts = self.ner(cSentence)
            answer = "In "
            for entNum in range(0,len(sentenceEnts)):
                if (sentenceEnts[entNum][1] == "GPE"):
                    answerLocs.append(entNum)
            answerLocs.append(-1)
            for locNum in range(0,len(answerLocs)-1):
                answer += sentenceEnts[answerLocs[locNum]][0]
                answer += " "
                if (answerLocs[locNum+1] - answerLocs[locNum] > 1):
                    answer += "and"
                    answer += " "
            if (answer == ""): answer = sentence # get original sentence
            return(answer)
        # time questions
        if (wh == "when"):
            words = cSentence.split()
            answer = ""
            inTimex = 0 # a tracker for if we are in a timex tagged phrase
            for wordNum in range(0,len(words)):
                if (words[wordNum] == "/TIMEX2"):
                    inTimex = 0
                if (inTimex == 1):
                    answer += words[wordNum]
                    answer += " "
                if (words[wordNum] == "TIMEX2"):
                    inTimex = 1
            year = re.compile("((?<=\s)\d{4}|^\d{4})")
            # make years sound more natural
            if (year.findall(answer)): answer = "In " + answer
            return(answer)
        # what question
        if (wh == "what" or wh == "which"):
            return(sentence) # return the whole best sentence

    def run(self):
        for i in range(0, len(self.questions)):
            print("Question "+str(i)+": "+self.questions[i])
            self.answerQuestion(self.questions[i], self.potentialAnswers[i])
