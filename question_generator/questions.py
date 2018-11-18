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

    def get_overall_questions(self):
        result = self.where()
        for i in range(0, len(self.sentences)):
            sentence = Sentence(self, i)
            for question in sentence.sentence_questions:
                result.append(question)
        return result

    def where(self):
        ner_tags = self.ner
        pos_tags = self.pos
        sentence_list = self.tokenized

        subjectFirst = False
        locationFirst = False
        subjectFirstIndex = 0
        locationFirstIndex = 0
        word_location = dict()

        for s in range(len(sentence_list)):
            sentence = sentence_list[s]
            for w in range(len(sentence)):
                word = sentence[w]
                if subjectFirst:
                    if (pos_tags[w][1] in ["VBD", "VBZ"]):
                        word_location["verb"] = (word, w)
                    if(ner_tags[s][w][1] == "LOCATION" and sentence[w-1] in ["at", "in"]):
                        subjectFirstIndex = w - 1
                        break
                if locationFirst:
                    if (pos_tags[s][w][1] in ["VBD", "VBZ"]):
                            word_location["verb"] = (word, w)
                            locationFirstIndex = w
                            break
                if (not (subjectFirst or locationFirst)):
                    if(ner_tags[s][w][1] == "LOCATION"):
                            locationFirst = True
                    elif(pos_tags[s][w][1] in ["NN", "NNP", "PRP"]):
                            subjectfirst = True
                            word_location["subject"] = (word, w)

        if not (subjectFirst or locationFirst):
            return []
        elif subjectFirst:
            verb_location = word_location["verb"][1]
            bigchunk1 = "".join(str(e) for e in sentence [verb_location+1:subjectFirstIndex])
            bigchunk1 = bigchunk1.replace(",", " ")

            subject_location = word_location["sub"][1]
            bigchunk0 = "".join(str(e) for e in sentence[w0-1:w1])
            bigchunk0 = bigchunk0.replace(",", " ")
            if(word_location["verb"][0] in ["is", "was"]):
                return ["Where " + word_location["verb"][0] + " " + bigchunk0 + " " + bigchunk1 + "?"]
            elif(ner_tags[word_location["verb"][1]][1] == "VBD"):
                return ["Where did " + word_location["subject"][0] + en.verb.present(word_location["verb"][0]) + " " + bigchunk1 + "?"]
            else:
                return ["Where does " + word_location["subject"][0] +  en.verb.present(word_location["verb"][0])+ " " + bigchunk1 + "?"]
        elif locationFirst:
            bigchunk = "".join(str(e) + " " for e in
                            sentence[locationFirstIndex:])
            bigchunk = bigchunk.replace(".", "")
            bigchunk = bigchunk.replace("!", "")
            return ["Where " + bigchunk + "?"]
        else:
            return []

class Sentence:
    def __init__(self, s, n):
        self.currSent = s.sentences[n]
        self.tokenized = s.tokenized[n]
        self.ner = s.ner[n]
        self.pos = s.pos[n]
        self.len = len(self.tokenized)
        self.sentence_questions = self.get_questions()

    def __repr__(self):
        return str(self.ner)

    def create_question(self, q):
        if(len(q) == 0):
            return q
        return " ".join(q) + "?"

    def get_questions(self):
        return [self.what(), self.who()]

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


if __name__ == '__main__':
    txtFile = sys.argv[1]
    n = int(sys.argv[2])
    with open(txtFile, 'r', encoding="latin1") as f:
            content = f.read()
            f.close()
    sentences = Sentences(content)
    counter = 0
    result = sentences.overall_questions
    for q in result:
        if(q != "" and counter < n):
            print(q)
            counter += 1
        else:
            break
    while(counter<n):
        print('')
        counter += 1




