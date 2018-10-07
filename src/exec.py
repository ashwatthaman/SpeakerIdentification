import sys,os
sys.path.append("../")
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from lstm_word_char import LSTMWordChar
from traintest import trainDev#,cosSimCategVec
import argparse

class Args():
    def __init__(self,train=True,spm=True,cv=1):
        dataname = "serif"
        self.train=train
        self.dataname = dataname
        self.cv = cv
        self.spm = spm
        # self.source ="./{}/all_{}16000_fixed.txt".format(dataname,dataname)
        # self.category="./{}/all_chara.txt".format(dataname)
        self.outtype="cls"
        self.epoch = 30
        self.n_char = 3500
        # self.n_vocab = 13269
        self.n_vocab = 25000
        # self.n_vocab = 22858
        self.embed = 64
        self.hidden= 128
        self.bgm_h = 54
        self.categ_size=221
        #points directory to transfer gensim w2v model
        if spm:
            self.premodel="../../pretrain_models/W2V/model/w2v_wiki_v16000_d{}.w2v".format(self.embed)
        else:
            self.premodel="../../pretrain_models/W2V/model/w2v_wiki_mecab_d{}.w2v".format(self.embed)
        self.prechar="../../pretrain_models/W2V/model/w2v_wiki_v16000_d{}_char.w2v".format(self.embed)

        # self.premodel=""

        self.n_latent = 60#0
        self.layer = 1
        self.batchsize=64
        self.sample_size = 10
        self.kl_zero_epoch = 15
        self.dropout = 0.5
        self.n_class = 2
        if train:
            self.gpu = -1#0
        else:
            self.gpu = -1
        self.gradclip = 5
        if spm:
            self.model_name_base = "LSTMWordCharSpm_{}_e{}_h{}_cv{}".format(dataname,self.embed,self.hidden,self.cv)
        else:
            self.model_name_base = "LSTMWordCharMecab25000_{}_e{}_h{}_cv{}".format(dataname,self.embed,self.hidden,self.cv)
        if not os.path.exists(self.dataname):os.mkdir(self.dataname)
        if not os.path.exists("{}/model".format(self.dataname)):os.mkdir("{}/model".format(self.dataname))
        self.model_dir = CDIR + "./{}/model/".format(self.dataname)
        # self.model_name_base = "AttSiamNLTK_{}_e{}_h{}_cv{}".format(dataname,self.embed,self.hidden,self.cv)
        self.model_name_base+= "_e{}"
        self.model_name = self.model_dir + self.model_name_base+'.npz'  # +"_man"

def sampleTrain():
    args = Args(True,spm=False)
    model = LSTMWordChar(args)
    x_tr, y_tr, x_dv, y_dv, x_te, y_te = model.getTrDvTe(args)
    model.vocab.save("./vocab_{}.bin".format(args.n_vocab))
    trainDev(args, model, x_tr, y_tr, x_dv, y_dv, x_te, y_te)
    model.getTestData(args)

def sampleTest(dirname):
    args = Args(False,spm=False)
    args.batchsize=1
    model = LSTMWordChar(args)
    model.loadModel(args)
    model.getTestData(args,dirname)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i","--input",
                        help="input file",
                        default="./test/"
                        )
    args = parser.parse_args()
    # sampleTrain(spm)
    sampleTest(args.input)

