from chainer import Variable
from chainer import cuda
from chainer import functions as F
from chainer import links as L

import numpy as np
if cuda.cudnn_enabled:
    import cupy as xp
else:
    global xp
    xp=np

from chainer import Chain
import os,sys
from chainer import serializers
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
from util.vocabulary import Vocabulary,vocabularize,labelize

import chainer

import util.generators as gens
import random
from gensim.models import word2vec

class NNChainer(Chain):

    def __init__(self,**args):
        super(NNChainer, self).__init__(**args)

    # def setParams(self,args):
    def setArgs(self,args):
        self.n_vocab = args.n_vocab
        self.n_embed = args.embed
        self.n_layers = args.layer
        self.out_size = args.hidden
        self.drop_ratio = args.dropout
        if args.gpu>=0:
            import cupy as xp

        self.setBatchSize(args.batchsize)
        self.setVocab(args)
        self.setMaxEpoch(args.epoch)
        self.setEpochNow(0)

    def setAttr(self,attr_list,attr_visi,txt_attr):
        self.attr_list = attr_list
        self.visible_attr_list = attr_visi
        self.txt_attr = txt_attr

    def extractVisAttr(self,tupl):

        visi_inds = [self.attr_list.index(vis) for vis in self.visible_attr_list]
        txt_inds = [self.attr_list.index(tat) for tat in self.txt_attr]

        # print("visi_inds:{}".format(visi_inds))
        txtize = lambda x_txt: " ".join([self.vocab.itos(id) for id in x_txt])
        # return [[tupl[ti][vi] for vi in visi_inds] for ti in range(len(tupl)) ]
        return [[txtize(tupl[ti][vi]) if vi in txt_inds else tupl[ti][vi] for vi in visi_inds] for ti in range(len(tupl))]

    def setDevice(self,args):
        global xp
        if args.gpu>=0:
            import cupy as cp
            xp = cp
            try:
                xp.cuda.Device(args.gpu).use()
            except cp.cuda.runtime.CUDARuntimeError:
                xp.cuda.Device().use()
            try:
                self.to_gpu(args.gpu)
            except xp.cuda.runtime.CUDARuntimeError:
                self.to_gpu()
        else:
            xp = np


    def setText(self,text_arr):
        self.text_arr = text_arr

    def setVocab(self, args):
        vocab_name = "./{}/vocab_{}.bin".format(args.dataname, args.dataname)
        set_vocab = set()

        sent_new_arr,vocab = vocabularize(self.text_arr,vocab_size=args.n_vocab,normalize=False)
        self.setText(sent_new_arr)
        n_vocab = len(set_vocab) + 3
        print("n_vocab:{}".format(n_vocab))
        print("arg_vocab:{}".format(args.n_vocab))
        # src_vocab = Vocabulary.new(self.text_arr, args.n_vocab)
        vocab.save(vocab_name)
        self.vocab = vocab
        return vocab


    def setEpochNow(self, epoch_now):
        self.epoch_now = epoch_now

    def setMaxEpoch(self, epoch):
        self.epoch = epoch

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def setData(self,tt_list,cat_list):
        self.tt_list =tt_list[:]
        self.cat_list=cat_list[:]

    def loadModel(self,args,load_epoch=None):
        first_e = 0
        model_name = ""
        max_epoch = args.epoch if load_epoch is None else load_epoch
        for e in range(max_epoch):
            model_name_tmp = args.model_name.format(e)
            print("model_tmp",model_name_tmp)
            if os.path.exists(model_name_tmp):
                model_name = model_name_tmp
                self.setEpochNow(e+1)

        if os.path.exists(model_name):
            serializers.load_npz(model_name, self)
            print("loaded_{}".format(model_name))
            first_e = self.epoch_now
        else:
            print("loadW2V")
            if os.path.exists(args.premodel):
                self.loadW(args.premodel)
            else:
                print("wordvec model doesnt exists.")
        return first_e

    def makeEmbedBatch(self,xs,reverse=False):
        # print('xs0:{}'.format(xs))
        if reverse:
            xs = [xp.asarray(x[::-1],dtype=xp.int32) for x in xs]
        elif not reverse:
            xs = [xp.asarray(x,dtype=xp.int32) for x in xs]
        # section_pre = xp.array([len(x) for x in xs[:-1]], dtype=xp.int32)
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs

    def makeCategEmbed(self,categ):
        categ = [xp.asarray(x,dtype=xp.int32) for x in categ]
        section_pre = np.array([len(x) for x in categ[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        categ = F.split_axis(self.categ_embed(F.concat(categ, axis=0)), sections, axis=0)
        return categ


    def getBatchGen(self,args):
    #     tt_now_list = [[self.vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source)]
    #     cat_now_list = [[self.categ_vocab.stoi(cat[0])] for cat in gens.word_list(args.category)]
        ind_arr = list(range(len(self.tt_list)))
        random.shuffle(ind_arr)
        tt_now = (self.tt_list[ind] for ind in ind_arr)
        cat_now = (self.cat_list[ind] for ind in ind_arr)
        tt_gen = gens.batch(tt_now, args.batchsize)
        cat_gen = gens.batch(cat_now, args.batchsize)
        for tt,cat in zip(tt_gen,cat_gen):
            yield (tt,cat)

    def loadW(self, premodel_name):
        src_vocab = self.vocab
        src_w2ind = {}
        src_ind2w = {}
        src_size = len(self.vocab)
        # print(src_size)
        for vi in range(src_size):
            src_ind2w[vi] = src_vocab.itos(vi)
            src_w2ind[src_ind2w[vi]] = vi
        # print("pre:{}".format(self.embed.W.data[0][:5]))
        # self.embed.W = Variable(xp.array(transferWordVector(src_w2ind, src_ind2w, premodel_name), dtype=xp.float32))
        self.embed.W.data = xp.array(transferWordVector(src_w2ind, src_ind2w, premodel_name), dtype=xp.float32)
        # print("pos:{}".format(self.embed.W.data[0][:5]))


    def setClassWeights(self,cat_list):
        count_hash = getPosNegWeightHash(cat_list)
        self.weights = xp.array([count_hash[ci] for ci in range(self.class_n)], dtype=xp.float32)


    def decode(self, ys_c,t,pred,decfunc,**args):
        t_all_list=[];ys_d_list=[]
        batch_size = len(t)
        t_pred,t_vec = self.makeDecInput(t)
        seq_len = max([len(t_vec[ti]) for ti in range(len(t_vec))])
        # print("t_len:{}".format([len(t_vec[ti]) for ti in range(len(t_vec))]))
        if pred:
            t = [[1] for _ in range(batch_size)]
        for ti in range(seq_len):
            if pred:
                vec_e = self.makeEmbedBatch(t)
            else:
                vec_e = [F.reshape(F.concat([t_e[ti]],axis=0),(1,self.n_embed)) for t_e in t_vec]
            pred_e = [xp.asarray([p_e[ti]],dtype=xp.int32) for p_e in t_pred]
            ys_d = decfunc(vec_e,ys_c,**args)

            if pred:
                y_w = self.h2w(ys_d)
                t = y_w.data.argmax(axis=1).tolist()
                t = [[t_e] for t_e in t]

            t_all = xp.array([p_e for t_each in pred_e for p_e in t_each.tolist()], dtype=xp.int32)
            # print("t_all_shape:{}".format(t_all.shape))
            # print("ys_d_shape:{}".format(ys_d.shape))
            ys_d_list.append(ys_d)
            t_all_list.append(t_all)
        ys_w = self.h2w(F.concat(ys_d_list, axis=0))
        #ys_w: (batch_size*max_seq_len,vocab_len)
        y_v_list = xp.reshape(ys_w.data.argmax(axis=1),(seq_len,batch_size))
        y_v_list = xp.transpose(y_v_list)

        y_tr_len_list = [len([t_e for t_e in t_p.tolist() if t_e!=-1]) for t_p in t_pred]
        ys_w_disp_list = [y_v[:y_len] for y_v,y_len in zip(y_v_list,y_tr_len_list)]

        ys_w_disp = ys_w_disp_list[0].tolist()
        # print("t_pred:{}".format(t_pred[0]))
        # print("t:{}".format([self.vocab.itos(int(t_e)) for t_e in t_pred[0]]))
        # print("ys_w_disp_list:{}".format(ys_w_disp_list))

        # print("t:{}".format([self.vocab.itos(int(t_e)) for t_e in t_pred[0] if t_e!=-1]))
        # print("y:{}\n".format([self.vocab.itos(y_v) for y_v in ys_w_disp]))

        t_all = xp.array([t_e for t_all in t_all_list for t_e in t_all.tolist()],dtype=xp.int32)
        return ys_w,t_all,ys_w_disp_list


class NstLSTM(L.NStepLSTM):
    def __init__(self, n_layer, in_size, out_size, dropout=0.5):
        n_layers = 1
        super(NstLSTM, self).__init__(n_layers, in_size, out_size, dropout)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(NstLSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(NstLSTM, self).to_gpu()#device)
        if self.cx is not None:
            self.cx.to_gpu()#device)
        if self.hx is not None:
            self.hx.to_gpu()#device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))

        hy, cy, ys = super(NstLSTM, self).__call__(self.hx, self.cx, xs)
        self.hx, self.cx = hy, cy
        return ys

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)

def predictRandom(prob):
    probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
    # print("probab:{}".format(probability))
    probability /= np.sum(probability)
    index = np.random.choice(range(len(probability)), p=probability)
    return index

def transferWordVector(w2ind_post, ind2w_post, premodel_name):
    premodel = word2vec.Word2Vec.load(premodel_name).wv
    vocab = premodel.vocab
    word = ""
    error_count=0
    print("ind2len:" + str(len(ind2w_post)))
    for ind in range(len(ind2w_post)):
        try:
            vocab[ind2w_post[ind]]
            word = ind2w_post[ind]
        except:
            error_count += 1

    # sims = premodel.most_similar("the",topn=5)
    sims = premodel.most_similar(word,topn=5)
    if '<unk>' not in vocab:
        unk_ind = vocab[word]
    else:
        unk_ind = vocab["<unk>"]
    print("unk_ind:"+str(unk_ind))
    print("errcount:"+str(error_count))
    W = [premodel.syn0norm[vocab.get(ind2w_post[ind],unk_ind).index].tolist() for ind in range(len(ind2w_post))]
    return W
