import re
import math
from collections import defaultdict
class NaiveBayes:
    def __init__(self):
        self.word_count = {}
        self.class_count = {}
        self.vocab = set()
        self.total_docs = 0

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()
    
    def fit(self, texts, labels):
        self.total_docs = len(labels)
        
        for label in labels:
            self.class_count[label] = self.class_count.get(label, 0) + 1
            
        for text, label in zip(texts, labels):

            if label not in self.word_count:
                self.word_count[label] = defaultdict(int)

            words = self.tokenize(text)

            for word in words:
                self.word_count[label][word] += 1
                self.vocab.add(word)
                
    def predict(self, text):
        words = self.tokenize(text)
        scores = {}
        for c in self.class_count:
            scores[c] = math.log(self.class_count[c]/self.total_docs)
            total_words = sum(self.word_count[c].values())

            for word in words:
                word_freq = self.word_count[c].get(word, 0)
                prob = (word_freq +1)/(total_words + len(self.vocab))
                scores[c]+= math.log(prob)
        return max(scores, key=scores.get)        

