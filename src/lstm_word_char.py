from chainer import links as L
from chainer import functions as F
import chainer
import os,sys
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")

from data_loader import load_data_from_file
from model_common import NNChainer,NstLSTM
import random
random.seed(623)
import numpy as np
xp = np

try:
    import cupy as xp
except ImportError:
    import numpy as xp
except ModuleNotFoundError:
    import numpy as xp

class LSTMWordChar(NNChainer):
    def __init__(self,args):
        self.setArgs(args)
        super(LSTMWordChar, self).__init__(
                embed_char = L.EmbedID(self.n_char,self.n_embed),
                # embed_word = L.EmbedID(self.n_vocab,self.n_embed),
                embed = L.EmbedID(self.n_vocab,self.n_embed),
                lstm = L.NStepBiLSTM(self.n_layers,2*self.n_embed,self.out_size,dropout=self.drop_ratio),
                h2w = L.Linear(2*self.out_size,2),
        )
        self.setDevice(args)

    def setArgs(self,args):
        if args.train:
            input_list,label_list,char,vocab = load_data_from_file(args)
            self.input_list = input_list
            self.label_list = label_list
        else:
            from util.vocabulary import Vocabulary
            vocab=Vocabulary.load("./serif/vocab2.txt")
            char=Vocabulary.load("./serif/char2.txt")
            # vocab=Vocabulary.load("./serif/vocab_25000.bin")
            # char=Vocabulary.load("./serif/char_3500.bin")

        self.n_char = args.n_char
        self.pwd = CDIR
        self.bgm_h = args.bgm_h
        # self.n_vocab = args.n_vocab
        print("n_vocab",args.n_vocab)
        print("vocab_len",len(vocab))
        self.vocab = vocab
        self.char = char
        self.premodel=args.premodel
        super().setArgs(args)
        print("self_n_vocab",self.n_vocab)

    def setVocab(self,args):
        pass
        # self.vocab = self.vocab

    def extractPretext(self,tupl_x):
        # return [tupl[0] for tupl in tupl_x]
        # return tupl_x[0]
        return [t_e+[2] for t_e in tupl_x[0]]

    def extractPostext(self,tupl_x):
        # return [tupl[1] for tupl in tupl_x]
        # return tupl_x[2]
        return [t_e + [2] for t_e in tupl_x[2]]

    def extractPreWords(self,tupl_x):
        # return tupl_x[3]
        return [t_e + [2] for t_e in tupl_x[3]]

    def extractPosWords(self,tupl_x):
        return [t_e + [2] for t_e in tupl_x[4]]
        # return tupl_x[4]

    def extractTeacher(self,t_list):
        for ri in range(len(t_list[0])):
            t_list[0][ri]=t_list[0][ri]+[0]
            t_list[1][ri]=t_list[1][ri]+[0]
        return t_list

    def extractVisCols(self):
        return "preline,posline,話者予測,話者正解\n"


    # def extractVisAttr(self,tupl_x,t_list,y_list):
    def extractVisAttr(self,tupl_x,t_list):
        y = self.predict(tupl_x)
        t_list=self.extractTeacher(t_list)
        y_list = y.data.argmax(1).tolist()
        prec_list=self.extractPretext(tupl_x)
        posc_list=self.extractPostext(tupl_x)
        prew_list=self.extractPreWords(tupl_x)
        posw_list=self.extractPosWords(tupl_x)
        t1 = [t_e2 for t_e in t_list[0] for t_e2 in t_e]
        len_t1_list = [len(t_e) for t_e in t_list[0]]
        len_t2_list = [len(t_e) for t_e in t_list[1]]
        t2 = [t_e2 for t_e in t_list[1] for t_e2 in t_e]

        y_pre = y_list[:len(t1)]
        section_pre = np.array(len_t1_list[:-1], dtype=np.int32)
        sections = np.cumsum(section_pre)  # CuPy does not have cumsum()
        y_pre_list = np.split(y_pre,sections)

        y_pos = y_list[len(t1):]
        section_pos = np.array(len_t2_list[:-1], dtype=np.int32)
        sections = np.cumsum(section_pos)  # CuPy does not have cumsum()
        y_pos_list = np.split(y_pos,sections)
        line_list = []


        for pre_list,pos_list, y_pre,y_pos, t_pre,t_pos in zip(prec_list,posc_list, y_pre_list,y_pos_list, t_list[0],t_list[1]):
            pretxt = " ".join([self.char.itos(id) for id in pre_list])
            postxt = " ".join([self.char.itos(id) for id in pos_list])
            y_pre =y_pre.tolist()
            y_pos =y_pos.tolist()

            washa_p = [self.char.itos(cid) for pi,(cid,yid) in enumerate(zip(pre_list+pos_list,y_pre+y_pos)) if yid==1]
            washa_t = [self.char.itos(cid) for pi,(cid,tid) in enumerate(zip(pre_list+pos_list,t_pre+t_pos)) if tid==1]
            line_str = "\"{}\",\"{}\",\"{}\",\"{}\"\n".format(pretxt,postxt,washa_p,washa_t)
            line_list.append(line_str)
        t_list=t1+t2
        return line_list,t_list,y_list


    def getTrDvTe(self,args,test_ratio=0.1):
        kl = int(1./test_ratio)

        xs_list,y_list = self.input_list,self.label_list
        print("xs_len",len(xs_list))
        print("y_len",len(y_list))
        assert len(y_list)==len(xs_list)
        ind_arr = list(range(len(y_list)))
        random.shuffle(ind_arr)
        xs_list = [xs_list[ind] for ind in ind_arr]
        y_list = [y_list[ind] for ind in ind_arr]

        test_inds = [ind for ii,ind in enumerate(ind_arr) if ii%kl==args.cv]
        dev_inds  = [ind for ii,ind in enumerate(ind_arr) if ii%kl==(args.cv+1)%kl]
        tr_inds   = [ind for ii,ind in enumerate(ind_arr) if ii%kl!=args.cv and ii%kl!=(args.cv+1)%kl]

        assert len(set(tr_inds).intersection(set(dev_inds)))  ==0
        assert len(set(tr_inds).intersection(set(test_inds))) ==0
        assert len(set(dev_inds).intersection(set(test_inds)))==0

        xs_tr = [xs_list[tri] for tri in tr_inds]
        xs_dv = [xs_list[dei] for dei in dev_inds]
        xs_te = [xs_list[tei] for tei in test_inds]

        y_tr = [y_list[tri] for tri in tr_inds]
        y_dv = [y_list[dei] for dei in dev_inds]
        y_te = [y_list[tei] for tei in test_inds]

        print("tr:{},dv:{},te:{}".format(len(xs_tr),len(xs_dv),len(xs_te)))
        return xs_tr,y_tr,xs_dv,y_dv,xs_te,y_te

    def encode(self,xs_c,xs_w):
        xs_emb = self.makeEmbedBatch(xs_c,xs_w)
        _,_,ys_c = self.lstm(None,None,xs_emb)
        return ys_c

    def makeEmbedBatch(self,xs_c,xs_w,reverse=False):
        # print("len_x_c",[len(x_c) for x_c in xs_c])
        # print("len_x_w",[len(x_w) for x_w in xs_w])
        xs_c_new=[];xs_w_new=[]
        for x_c,x_w in zip(xs_c,xs_w):
            if len(x_c)!=len(x_w):
                print(len(x_c),len(x_w))
                print([self.char.itos(c_e) for c_e in x_c])
                print([self.vocab.itos(w_e) for w_e in x_w])
        if reverse:
            xs_c = [xp.asarray(x[::-1],dtype=xp.int32) for x in xs_c]
            xs_w = [xp.asarray(x[::-1],dtype=xp.int32) for x in xs_w]
        elif not reverse:
            xs_c = [xp.asarray(x,dtype=xp.int32) for x in xs_c]
            xs_w = [xp.asarray(x,dtype=xp.int32) for x in xs_w]
        section_pre = np.array([len(x) for x in xs_c[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        emb_c = self.embed_char(F.concat(xs_c, axis=0))
        if len(self.premodel)>5 and self.epoch_now<2:
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                emb_w = self.embed(F.concat(xs_w, axis=0))
        else:
            emb_w = self.embed(F.concat(xs_w, axis=0))
        emb = F.concat([emb_c,emb_w],axis=1)
        xs = F.split_axis(emb, sections, axis=0)
        return xs

    def __call__(self,tupl):
        tupl_x,t = tupl#[0];t =  tupl[1]
        t = self.extractTeacher(t)
        t_pre = [t_pre_e2 for t_pre_e in t[0] for t_pre_e2 in t_pre_e]
        t_pos = [t_pos_e2 for t_pos_e in t[1] for t_pos_e2 in t_pos_e]
        t_all = xp.array(t_pre+t_pos,dtype=xp.int32)
        ys_w = self.predict(tupl_x)
        loss = F.softmax_cross_entropy(ys_w, t_all,ignore_label=-1)  # /len(t_all)
        return loss

    def predict(self,tupl_x):
        x_pre_char = self.extractPretext(tupl_x)
        x_pre_word = self.extractPreWords(tupl_x)
        x_pos_char = self.extractPostext(tupl_x)
        x_pos_word = self.extractPosWords(tupl_x)

        # print("x_pre",[self.vocab.itos(v_e) for v_e in x_pre[0]])
        # print("x_pre",[len(v_e) for v_e in x_pre])
        # print("x_pos",[self.vocab.itos(v_e) for v_e in x_pos[0]])
        # print("x_pos",[len(v_e) for v_e in x_pos])
        y_pre = self.encode(x_pre_char,x_pre_word)
        y_pos = self.encode(x_pos_char,x_pos_word)
        # print("y_pre",[p_e.shape for p_e in y_pre])
        # print("y_pos",[p_e.shape for p_e in y_pos])
        # y_c = F.concat([y_pre,y_pos],axis=0)
        y_c = F.concat(y_pre+y_pos,axis=0)
        ys_w = self.h2w(F.tanh(y_c))
        # print('ys_w',ys_w.shape)
        return ys_w

    def getTestData(self,args,dirname):
        import glob,subprocess,codecs
        from preprocess import preprocessLine
        serif_delim = "「"
        # file_arr = [file for file in glob.glob(CDIR+"./test/*.txt") if "spm" not in file and "mecab" not in file]
        file_arr = [file for file in glob.glob(dirname+"*.txt") if "spm" not in file and "mecab" not in file]
        def sepCharas(self,pre_line,pos_line,y_list):
            chara_list = [self.char.itos(wid) if y==1 else " " for wid,y in zip(pre_line+pos_line,y_list)]
            print("chara_list",chara_list)
            pre_chara_list = chara_list[:len(pre_line)]

            pre_charas = "".join(pre_chara_list)
            while "  " in pre_charas:pre_charas=pre_charas.replace("  "," ")
            # pre_chara_list = pre_charas.split(" ")
            pre_chara_list = [pre for pre in pre_charas.split(" ") if len(pre)>0]

            print("prechara_list",pre_chara_list)

            pos_chara_list = chara_list[len(pre_line):]

            pos_charas = "".join(pos_chara_list)
            while "  " in pos_charas:pos_charas=pos_charas.replace("  "," ")
            pos_chara_list = [pos for pos in pos_charas.split(" ") if len(pos)>0][::-1]
            print("poschara_list",pos_chara_list)

            # print(pre_chara_list[0:1]+pos_chara_list[0:1]+pre_chara_list[1:]+pos_chara_list[1:])
            return pre_chara_list[0:1]+pos_chara_list[0:1]+pre_chara_list[1:]+pos_chara_list[1:]

        def asteriskSerif(line):
            if serif_delim in line:
                return [self.vocab.stoi("*")],[self.char.stoi("*")]
            else:
                w_id=[self.vocab.stoi(word) for word in line.split(" ") for _ in range(len(word))]
                c_id=[self.char.stoi(char) for char in line.replace(" ", "")]
                return w_id,c_id

        for file in file_arr:
            if not os.path.exists(file.replace(".txt","_mecab.txt")):
                subprocess.call("mecab -O wakati -b 50000 {} > {}".format(file,file.replace(".txt","_mecab.txt")),shell=True)


        for file in file_arr:
            prespm_list = [];serifspm_list = [];posspm_list = []
            preline_list = [];serifline_list = [];posline_list = []
            linenum_list = []
            if args.spm:
                write_file=file.replace(".txt", "_spm.txt")
            else:
                write_file = file.replace(".txt", "_mecab.txt")
            line_list = [line.replace("▁","").strip() for line in codecs.open(write_file,encoding="utf-8").readlines()]
            fr=[]
            for line in line_list:
                if serif_delim in line:
                    if line.index(serif_delim)<10:
                        fr.append(line)
                    else:fr+=[l_e+'。' for l_e in line.split('。') if len(l_e)>3]
                else:fr+=[l_e+'。' for l_e in line.split('。') if len(l_e)>3]


            for li,line in enumerate(fr):
                if serif_delim not in line:continue
                washa = line.split("「")[0]
                serif = preprocessLine(line.replace("{}「".format(washa),"「"))
                pre_line = preprocessLine(fr[li-1],split_erase=False)
                pos_line =preprocessLine(fr[li+1], split_erase=False)

                if serif_delim not in pre_line or serif_delim not in pos_line:

                    pre_spm,pre_line = asteriskSerif(pre_line)
                    prespm_list.append(pre_spm);preline_list.append(pre_line)
                    pos_spm,pos_line = asteriskSerif(pos_line)
                    posspm_list.append(pos_spm);posline_list.append(pos_line)

                    serifspm_list.append([self.vocab.stoi(word) for word in serif.split(" ") for _ in range(len(word))])
                    serifline_list.append([self.char.stoi(char) for char in serif.replace(" ","")])

                    linenum_list.append(li)
            input_list = [(c1,c2,c3,w1,w2,li) for c1,c2,c3,w1,w2,li in zip(preline_list,serifline_list,posline_list,prespm_list,posspm_list,linenum_list)]
            linenum_dict={}

            for inp in input_list:
                pre_line = inp[0]+[2]
                pos_line = inp[2]+[2]

                pre_spm = inp[3]+[2]
                pos_spm = inp[4]+[2]
                line_num = inp[5]

                inp = [[i_e] for i_e in inp]
                ys_w = self.predict(inp)
                y_list = ys_w.data.argmax(1).tolist()

                print('')
                print('char',''.join([self.char.itos(cid) for cid,y in zip(pre_line+pos_line,y_list)]))
                print('vocab',''.join([self.vocab.itos(wid) for wid,y in zip(pre_spm+pos_spm,y_list)]))
                print('name候補',''.join([self.char.itos(wid) for wid,y in zip(pre_line+pos_line,y_list) if y==1]))

                chara=sepCharas(self,pre_line,pos_line,y_list)
                print("cccchara",chara)
                if len(chara)>0:
                    linenum_dict[line_num]=chara[0]

            #話者名を付与
            fw = codecs.open(write_file.replace(".txt","_charatag.txt"),"w",encoding="utf-8")
            serif_lines = set([li for li,line in enumerate(fr) if serif_delim in line])

            for li,line in enumerate(fr):
                if serif_delim in line:
                    line = serif_delim+line.split(serif_delim)[-1]
                if li in linenum_dict:
                    fw.write(linenum_dict[li]+line+"\n")
                else:
                    fw.write(line+"\n")
            fw.close()

            # 話者名を付与 連続するセリフ文も補間。
            fw = codecs.open(write_file.replace(".txt","_charatag_complemented.txt"),"w",encoding="utf-8")
            # print("li_list",li_list)
            for li,line in enumerate(fr):
                li_list = sorted(list(linenum_dict.keys()))

                if li in linenum_dict:
                    if li+1 in serif_lines and li+1 not in linenum_dict:
                        if li_list.index(li)==0:chara_ind=li_list.index(li)+1
                        elif li_list.index(li)==len(li_list):chara_ind=li_list.index(li)-1
                        elif abs(li_list[li_list.index(li)-1]-li)>abs(li_list[li_list.index(li)+1]-li):
                            chara_ind=li_list.index(li)+1
                            # chara = linenum_dict[li_list[li_list.index(li)+1]]
                        else:
                            chara_ind=li_list.index(li)-1
                            # chara = linenum_dict[li_list[li_list.index(li)-1]]
                        chara = linenum_dict[li_list[chara_ind]]
                        linenum_dict[li+1]=chara

            for li,line in enumerate(fr):
                if serif_delim in line:
                    line = serif_delim+line.split(serif_delim)[-1]
                if li in linenum_dict:
                    fw.write(linenum_dict[li]+line+"\n")
                else:
                    fw.write(line+"\n")
            fw.close()

