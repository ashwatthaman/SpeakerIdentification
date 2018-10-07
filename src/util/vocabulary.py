from collections import Counter,defaultdict
import pickle,codecs,re

class Vocabulary():

    def __init__(self):
        pass

    @staticmethod
    def new(sent_arr,vocab_size=None,thre=None,categ=False,char=False):
        self=Vocabulary()
        word_arr=[]#=set()
        self.char = char
        for sent in sent_arr:
            # print("sent:{}".format(sent))
            if char:
                [word_arr.append(char) for char in list(sent)]
            else:
                [word_arr.append(word) for word in sent.split(" ")]
        if vocab_size is None:
            if thre is None:
                vocab_size=len(set(word_arr))
        elif len(set(word_arr)) < vocab_size:
                vocab_size = len(set(word_arr))

        if thre is not None:
            wordcnt = {word: cnt for word, cnt in Counter(word_arr).most_common() if cnt>=thre}
            # vocab_size=len(wordcnt)
        else:
            wordcnt = {word: cnt for word, cnt in Counter(word_arr).most_common(vocab_size - 3)}
            # wordcnt = {word: cnt for word, cnt in Counter(word_arr).most_common(vocab_size)}

        word2id = defaultdict(lambda :0)
        if not categ:
            for wi,word in enumerate(wordcnt):
                word2id[word]=wi+3
            word2id['<unk>'] = 0
            word2id['<s>'] = 1
            word2id['</s>'] = 2
        else:
            for wi,word in enumerate(wordcnt):
                word2id[word]=wi
        self.id2word = {word2id[word]:word for word in word2id}
        self.__size=len(word2id)
        self.word2id=word2id
        return self

    def __len__(self):
        return self.__size

    def stoi(self, s):
        return self.word2id[s]

    def itos(self, i):
        return self.id2word[i]

    def expand(self,vocab,addname=None):
        len_vocab=len(self)
        for wi,word in enumerate([word for word in vocab.word2id if word not in self.word2id]):
            # print("{}:{}".format(word,wi+len_vocab))
            self.word2id[word]=wi+len_vocab
            self.id2word[wi+len_vocab]=word
        self.__size=len(self.word2id)

    def makeData(self,sent_arr):
        # print(sent_arr.__class__.__name__)
        # print(sent_arr)
        # print(len(sent_arr))
        if self.char:
            split = lambda s:list(s)
        else:
            split = lambda s:s.split(" ")
        if sent_arr.__class__.__name__=='generator':
            return [[self.word2id[word] for word in split(sent)] for sent in sent_arr]
        if sent_arr[0].__class__.__name__=="str":
            return [[self.word2id[word] for word in split(sent)] for sent in sent_arr]
        elif sent_arr[0].__class__.__name__=="list":
            return [[self.word2id[word] for word in sent] for sent in sent_arr]

    def save(self, filename):
        # with codecs.open(filename, 'w', encoding="utf-8") as fp:
        print("save_size:{}".format(self.__size))
        with codecs.open(filename, 'w',encoding="utf-8") as fp:
            print(self.__size, file=fp)
            for i in range(self.__size):
                print(self.itos(i), file=fp)

    @staticmethod
    def load(filename):
        # with codecs.open(filename, "r", encoding="utf-8") as fp:
        with codecs.open(filename, "r",encoding="utf-8") as fp:
            self = Vocabulary()
            self.__size = int(next(fp))
            print("load_size:{}".format(self.__size))
            self.word2id = defaultdict(lambda: 0)
            self.id2word = [''] * self.__size
            for i in range(self.__size):
                s = next(fp).strip()
                if s:
                    self.word2id[s] = i
                    self.id2word[i] = s
        return self

def normalizeSent(sent,normalize=True):
    if sent.__class__.__name__ != "str":
        return ""
    if normalize:
        patt = re.compile("https?://[A-Za-z\./]*")
        for match in re.findall(patt,sent):
            sent=sent.replace(match," <url> ")
        # del_letters = "“ ” - &amp;t &amp; ' ’ … * : \" ( ) &gt; &gt [ ] ^ { } ~ ".split(" ")
        del_letters = "“ ” - &amp;t &amp; ' ’ … * : \" ( ) [ ] ^ { } ~ ".split(" ")
        for del_lett in del_letters:sent = sent.replace(del_lett,"")
        with_spaces = ". , ! ? : _ / \n".split(" ")
        for with_sp in with_spaces:sent = sent.replace(with_sp," {} ".format(with_sp))
        while "  " in sent:sent = sent.replace("  "," ")
        sent = sent.lower().strip()
        # if ";" in sent:
        #     print(sent)
        # sent = sent.replace("/"," / ")

    return sent


def vocabularize(sent_arr_arr,vocab_size=None,vocab=None,normalize=True):
    sent_all=[]
    if sent_arr_arr[0].__class__.__name__=="str":
        sent_arr_arr = [[sent] for sent in sent_arr_arr]
    for si in range(len(sent_arr_arr)):
        # print(sent_arr_arr[si])

        sent_arr_arr[si]=[normalizeSent(sent,normalize) for sent in sent_arr_arr[si]]
        sent_all+=sent_arr_arr[si]
    print(Counter([len(sent_arr) for sent_arr in sent_arr_arr]).most_common())
    if vocab is None:
        vocab = Vocabulary.new(sent_all,vocab_size)
    sent_new_arr_arr=[vocab.makeData(sent_arr) for sent_arr in sent_arr_arr]
    return sent_new_arr_arr,vocab

def characterize(sent_arr_arr,vocab_size=None,vocab=None,normalize=True):
    sent_all=[]
    if sent_arr_arr[0].__class__.__name__=="str":
        sent_arr_arr = [[sent] for sent in sent_arr_arr]
    for si in range(len(sent_arr_arr)):
        # print(sent_arr_arr[si])

        sent_arr_arr[si]=[normalizeSent(sent,normalize) for sent in sent_arr_arr[si]]
        sent_all+=sent_arr_arr[si]
    print(Counter([len(sent_arr) for sent_arr in sent_arr_arr]).most_common())
    if vocab is None:
        vocab = Vocabulary.new(sent_all,vocab_size,char=True)
    sent_new_arr_arr=[vocab.makeData(sent_arr) for sent_arr in sent_arr_arr]
    return sent_new_arr_arr,vocab

def categorize(categ_arr_arr,attr_arr,categ_size_arr=None,categ=None):
    if categ_size_arr is None:
        categ_size_arr= [len(set(categ_arr)) for categ_arr in categ_arr_arr]
    for ci in range(len(categ_arr_arr)):
        categ_arr_arr[ci]=["{}:{}".format(attr_arr[ci],categ) for categ in categ_arr_arr[ci]]
    if categ is None:
        categ=Vocabulary.new(categ_arr_arr[0],categ_size_arr[0])
        [categ.expand(Vocabulary.new(categ_arr,categ_size)) for categ_arr,categ_size in zip(categ_arr_arr[1:],categ_size_arr[1:])]
    categ_new_arr_arr=categ.makeData([[categ_arr_arr[ri][ci] for ri in range(len(categ_arr_arr))] for ci in range(len(categ_arr_arr[0]))])
    return categ_new_arr_arr,categ

def labelize(label_arr,label_size,label_vocab=None):
    if label_size is None:
        label_size = len(set(label_arr))

    if label_vocab is None:
        categ=Vocabulary.new(label_arr,label_size,categ=True)
    categ_new_arr_arr=categ.makeData(label_arr)
    return categ_new_arr_arr,categ

