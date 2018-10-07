import pandas,re,sys
import sys,os
from preprocess import load_same_name_dict,isSpeakerInSerif
sys.path.append("../")
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")

from util.vocabulary import characterize,vocabularize

def load_data_from_file(args):
    if args.spm:
        df = pandas.read_csv(CDIR+"./mldata/washa_predict_spm.csv")
    else:
        df = pandas.read_csv(CDIR + "./mldata/washa_predict_mecab.csv")

    df = df.fillna("")
    serif_spm_cols=["pre_line_spm","serif_spm","pos_line_spm"]
    serif_cols=["pre_line_spm","serif","pos_line"]

    norm = lambda s:s#.replace("\n",'').replace('\r','').replace('　','').replace(' ','').replace('…','...').replace('‥','..')#.replace('″','′′');
    title_list = [title.split("/")[-1] for title in df["title"]]
    washa_list = df["washa"]
    preline_list=[norm(preline) for preline in df["pre_line"]]
    serif_list = [norm(serif) for serif in df["serif"]]
    posline_list=[norm(posline) for posline in df["pos_line"]]
    prespm_list=df["pre_line_spm"]
    serifspm_list = df["serif_spm"]
    posspm_list=df["pos_line_spm"]
    input_list, label_list, char, vocab=load_data(args, title_list, washa_list, preline_list, serif_list, posline_list, prespm_list, serifspm_list,posspm_list)
    return input_list,label_list,char,vocab

def load_data(args,title_list,washa_list,preline_list,serif_list,posline_list,prespm_list,serifspm_list,posspm_list):
    prechar_ids, poschar_ids = make_charids_of_line(title_list, washa_list, preline_list, serif_list, posline_list)

    lenpre_list=[[len(preword) for preword in prespm.split(" ")] for prespm in prespm_list]
    lenpos_list=[[len(posword) for posword in posspm.split(" ")] for posspm in posspm_list]

    (pre_cids,serif_cids,pos_cids),char=characterize([list(preline_list),list(serif_list),list(posline_list)],normalize=False,vocab_size=args.n_char)
    (pre_wids,serif_wids,pos_wids),vocab=vocabularize([list(prespm_list),list(serifspm_list),list(posspm_list)],normalize=False,vocab_size=args.n_vocab)


    label_list = [(prechar,poschar) for prechar,poschar in zip(prechar_ids,poschar_ids)]

    copy_prespm_list, copy_posspm_list = make_copy_spm(pre_wids, lenpre_list, pos_wids, lenpos_list)
    assert len(pre_cids)==len(copy_prespm_list)
    input_list = [(c1,c2,c3,w1,w2) for c1,c2,c3,w1,w2 in zip(pre_cids,serif_cids,pos_cids,copy_prespm_list,copy_posspm_list)]

    vocab.save("vocab2.txt")
    char.save("char2.txt")
    return input_list,label_list,char,vocab

def make_charids_of_line(title_list,washa_list,preline_list,serif_list,posline_list):
    prechar_ids = []
    poschar_ids = []

    title_samename_dict = {title: load_same_name_dict(title) for title in set(title_list)}
    for title,washa,preline,serif,posline in zip(title_list,washa_list,preline_list,serif_list,posline_list):
        samename_dict=title_samename_dict[title]
        prechar = [0]*len(preline)
        poschar = [0]*len(posline)
        if isSpeakerInSerif("",posline,washa,samename_dict):
        # if washa_tmp in posline:
            washa_list_cls = [washa_tmp for washa_tmp in [washa]+samename_dict.get(washa,[]) if washa_tmp in posline]
            washa_list_cls = sorted(washa_list_cls,key=lambda x:len(x),reverse=True)
            # print("washa_list",washa_list_cls)
            washa_tmp = washa_list_cls[0]
            wind = posline.index(washa_tmp)
            # print(posline[wind:wind+len(washa)])
            poschar=[1 if ri>=wind and ri<wind+len(washa_tmp) else 0 for ri in range(len(posline))]

        elif isSpeakerInSerif(preline,"",washa,samename_dict):
            washa_list_cls = [washa_tmp for washa_tmp in [washa]+samename_dict.get(washa,[]) if washa_tmp in preline]
            washa_list_cls = sorted(washa_list_cls,key=lambda x:len(x),reverse=True)
            # print("washa_list",washa_list_cls)
            washa_tmp = washa_list_cls[0]

            wind = preline.rindex(washa_tmp)
            # print(preline[wind:wind+len(washa)])
            prechar=[1 if ri>=wind and ri<wind+len(washa_tmp) else 0  for ri in range(len(preline))]

        assert len(preline)==len(prechar)
        assert len(posline)==len(poschar)
        prechar_ids.append(prechar)
        poschar_ids.append(poschar)
    return prechar_ids,poschar_ids

def make_copy_spm(pre_wids,lenpre_list,pos_wids,lenpos_list):
    copy_prespm_list=[]
    copy_posspm_list=[]
    assert len(pre_wids)==len(lenpre_list)
    # for pre_wid,prelen,prechar,preline_df,prespm_df in zip(pre_wids,lenpre_list,sentc_arr_arr[0],preline_list,prespm_list):
    for pre_wid,prelen in zip(pre_wids,lenpre_list):
        copy_prespm = [ps_e for ps_e,pl_e in zip(pre_wid,prelen) for _ in range(pl_e)]
        copy_prespm_list.append(copy_prespm)
    assert len(pos_wids)==len(lenpos_list)
    for posspm,poslen in zip(pos_wids,lenpos_list):
    # for posspm,poslen in zip(posspm_list,lenpos_list):
        assert len(posspm)==len(poslen)
        copy_posspm = [ps_e for ps_e,pl_e in zip(posspm,poslen) for _ in range(pl_e)]
        copy_posspm_list.append(copy_posspm)
    return copy_prespm_list,copy_posspm_list


if __name__=="__main__":
    loadData()