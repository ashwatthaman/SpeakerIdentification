from chainer import serializers,optimizers
from chainer import functions as F
import chainer
import codecs,sys,json
import numpy as np

# from evaluation import evalDiscrete,rankCommon
import os,math
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")

def trainDev(args,encdec,x_train,y_train,x_dev,y_dev,x_test,y_test):
    def trainSub(encdec,optimizer,train):
        loss_sum = 0
        for tupl in encdec.getBatchGen(args):
            loss = encdec(tupl)
            assert not math.isnan(loss.data)
            loss_sum += loss.data
            assert not math.isnan(loss_sum)
            if train:
                encdec.cleargrads()
                loss.backward()
                optimizer.update()
        return loss_sum#"""

    def trainSubSuper(encdec,optimizer,args,x_train,y_train,x_dev,y_dev,x_test,y_test,iter_unit,es_now,eval_dev_max,early_stop=3):
        ep_unit = len(x_train) // iter_unit
        if ep_unit>0:
            x_train_arr=[x_train[ri*iter_unit:(ri+1)*iter_unit] for ri in range(ep_unit)]
            y_train_arr=[y_train[ri*iter_unit:(ri+1)*iter_unit] for ri in range(ep_unit)]
        else:
            x_train_arr=[x_train]
            y_train_arr=[y_train]
        for iter_i,(x_tr,y_tr) in enumerate(zip(x_train_arr,y_train_arr)):
            encdec.setData(x_tr, y_tr)
            loss_sum=trainSub(encdec,optimizer,train=True)
            encdec.setData(x_dev, y_dev)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                eval_dev, eval_test = devtest(args, encdec, x_test, y_test)
            print("\n   epoch{}_{}:loss_sum:{},eval_dev:{},eval_test:{}".format(encdec.epoch_now,iter_i, round(float(loss_sum), 4),round(float(eval_dev), 4),round(float(eval_test), 4)))
            if eval_dev_max is None or eval_dev > eval_dev_max:
                eval_dev_max = eval_dev
                print('update_model:{}'.format(encdec.epoch_now))
                print('model_name:{}'.format(args.model_name))
                serializers.save_npz(args.model_name.format(encdec.epoch_now)+'.npz', encdec)
                es_now = 0
            else:
                es_now += 1
            if es_now > early_stop:
                print('earlyStopped')
                break
        return eval_dev_max,es_now

    def devtest(args, encdec,  x_test, y_test):
        if args.outtype=="reg":
            eval_dev = 100.0 / trainSub(encdec.copy(),None, train=False)
            print('   eval_dev:{}'.format(eval_dev))
            encdec.setData(x_test, y_test)
            eval_test = testRank(args, encdec.copy())
            print('   eval_test:{}'.format(round(eval_test,4)))
        else:
            print('   dev:')
            eval_dev = testClassify(args, encdec.copy())
            encdec.setData(x_test, y_test)
            print('   test:')
            eval_test = testClassify(args, encdec.copy())
        return eval_dev, eval_test
    first_e=encdec.loadModel(args)
    early_stop=4;es_now=0
    lr=0.001
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup(encdec)
    # with chainer.using_config('train', False), chainer.no_backprop_mode():
    #     encdec.setData(x_dev, y_dev)
    #     eval_dev_max, eval_test = devtest(args, encdec, x_test, y_test)
    eval_dev_max=0.0;eval_test=0.0
    for e_i in range(encdec.epoch_now, args.epoch):
        encdec.setEpochNow(e_i)
        # print('{}:e_now:{}:{}'.format(args.dataname,encdec.epoch_now,args.epoch))
        eval_dev_max,es_now=trainSubSuper(encdec,optimizer,args,x_train,y_train,x_dev,y_dev,x_test,y_test,iter_unit=65536,es_now=es_now,eval_dev_max=eval_dev_max,early_stop=early_stop)
        if es_now > early_stop:
            print('earlyStopped')
            break

def testClassify(args,model):
    filename = args.model_name_base.format(model.epoch_now)
    if not os.path.exists(model.pwd + "./each_{}/{}/output/".format(args.dataname,args.dataname)):
        os.mkdir(model.pwd + "./each_{}/{}/output/".format(args.dataname,args.dataname))
    fw = codecs.open(model.pwd + "./each_{}/{}/output/result_{}.csv".format(args.datname,args.dataname, filename), "w",encoding="utf-8")
    logw = codecs.open(model.pwd + "./each_{}/log/log_{}.json".format(args.dataname,filename), "w", encoding="utf-8")
    log_hash = {}
    log_hash["args"] = args.__dict__
    fw.write(model.extractVisCols())
    all_t=[];all_y=[]
    for tupl in model.getBatchGen(args):
        x_arr=tupl[0];t_list=tupl[1]
        line_list,t_list,y = model.extractVisAttr(x_arr, t_list)
        [fw.write(line) for line in line_list]
        all_y+=y
        all_t+=t_list
    eval_val,eval_hash=evalDiscrete(all_t, all_y)
    log_hash["eval"] = eval_hash
    logw.write(json.dumps(log_hash, default=lambda x: x.__class__.__name__))
    logw.close()
    fw.close()
    return eval_val#,eval_hash


