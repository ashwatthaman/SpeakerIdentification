import codecs,re,glob,subprocess,csv,json,pandas,os
CDIR=os.path.abspath(__file__)
CDIR=CDIR.replace(CDIR.split("/")[-1],"")
serif_delim = "「"
#セリフ分の前後がセリフ文か、地の文であれば話者名が含まれているものの割合はどの程度か？
#集計したいもの
# 両方がセリフ
# atLeastどちらかが地の文で、話者名なし
# atLeastどちらかが地の文で、話者名あり

def loadStop():
    fr = codecs.open("stopchara","r",encoding="utf-8")
    stch_set=set([line.replace("\n","") for line in fr])
    # print(stch_arr)
    return stch_set

def preprocessLine(line,split_erase=True):
    line = re.sub("\[.*?\]", "", line)
    line = re.sub("［.*?］", "", line)
    line = line.replace("",'').replace("\n",'').replace('\r','').replace('　','').replace('…','...').replace('‥','..').replace('″','');
    if split_erase:
        line=line.replace(' ', '')
    if len(line)==0:line="</s>"
    return line

def load_same_name_dict(title):
    df = pandas.read_csv(CDIR+"./mldata/same_person.txt")
    col_list ="作品,記名,他候補名".split(",")
    df =df[df["作品"]==title]
    speaker_dict = {}
    for kimei,koho in zip(df["記名"],df["他候補名"]):
        if kimei not in speaker_dict:
            speaker_dict[kimei]=[]
        speaker_dict[kimei].append(koho)
    return speaker_dict

def isSpeakerInSerif(preline,posline,speaker,speaker_dict):
    for speaker in [speaker]+speaker_dict.get(speaker,[]):
        if speaker in preline+posline:
            return True
    return False

def get_inpout_from_file(fname):
    speaker_dict = load_same_name_dict(fname)
    lines = codecs.open(fname, "r", encoding="sjis").readlines()
    get_train_input_output(lines,speaker_dict)

def get_train_input_output(lines,speaker_dict):

    def isSpeaker(speaker,stop_set,thre=10):
        if speaker not in stop_set and len(speaker)<thre:
            return True
        else:return False
    fr=[]
    for line in lines:
        if serif_delim in line:
            if line.index(serif_delim) < 10:
                fr.append(line)
            else:
                fr += [l_e + '。' for l_e in line.split('。') if len(l_e) > 3]
        else:fr += [l_e + '。' for l_e in line.split('。') if len(l_e) > 3]
    stop_set = loadStop()
    cnt_list=[]

    pre_list=[];serif_list=[];pos_list=[];speaker_list=[]
    nopre_list=[];noserif_list=[];nopos_list=[];nospeaker_list=[]

    for li,line in enumerate(fr):
        if serif_delim not in line:continue
        speaker = line.split("「")[0]
        if not isSpeaker(speaker,stop_set):continue
        serif    = preprocessLine(line.replace("{}「".format(speaker),"「"))
        pre_line = preprocessLine(fr[li-1])
        pos_line = preprocessLine(fr[li+1])

        #セ文をあれにする。
        if serif_delim in pre_line and serif_delim in pos_line:
            cnt_list.append("Serif")

        # elif speaker in pre_line+pos_line:
        elif isSpeakerInSerif(pre_line,pos_line,speaker,speaker_dict):
            if serif_delim in pre_line:
                if pre_line.index(serif_delim) < 10:
                    pre_line = '*'
            if serif_delim in pos_line:
                if pos_line.index(serif_delim) < 10:
                    pos_line = '*'
            cnt_list.append("In")
            speaker_list.append(speaker);pre_list.append(pre_line);serif_list.append(serif);pos_list.append(pos_line)

        else:
            cnt_list.append("Out")
            # speaker_list.append(speaker);pre_list.append(pre_line);serif_list.append(serif);pos_list.append(pos_line)
            nospeaker_list.append(speaker);nopre_list.append(pre_line);noserif_list.append(serif);nopos_list.append(pos_line)

    writeNoSpeakerCSV(fname,nospeaker_list,nopre_list,noserif_list,nopos_list)

    from collections import Counter
    cnt_dict = {tupl[0]:tupl[1] for tupl in Counter(cnt_list).most_common()}
    # print([(attr,cnt_dict[attr]) for attr in ["preSerif","posSerif","preIn","posIn","preJi","posJi"]])
    try:
        print([(attr,cnt_dict[attr]) for attr in ["Serif","In","Out"]])
    except KeyError:
        return [],[],[],[]
    return speaker_list,pre_list,serif_list,pos_list

def writeNoSpeakerCSV(title,speaker_list,pre_list,serif_list,pos_list):
    title = title.split("/")[-1].replace(".txt.csv","")
    title_list = [title]*len(speaker_list)
    df_dict = {"title":title_list,"speaker":speaker_list,"pre_line":pre_list,"serif":serif_list,"pos_line":pos_list}
    df = pandas.DataFrame(df_dict)
    df = df[["title","speaker","pre_line","serif","pos_line"]]
    df.to_csv("./mldata/nospeaker/nospeaker_predict_{}.csv".format(title))


def write(speaker_list,pre_list,serif_list,pos_list,title_list,vocab_n=16000):

    df_dict = {"title":title_list,"speaker":speaker_list,"pre_line":pre_list,"serif":serif_list,"pos_line":pos_list}
    df = pandas.DataFrame(df_dict)
    df = df[["title","speaker","pre_line","serif","pos_line"]]
    df.to_csv("./mldata/speaker_predict.csv")

    # df = pandas.read_csv("./mldata/speaker_predict.csv",encoding="utf-8")
    serif_list = list(df["serif"])+list(df["pre_line"])+list(df["pos_line"])
    print('serif_list_len',len(serif_list))
    print('serif_df_len',len(df['serif']))
    print('pre_df_len',len(df['pre_line']))
    print('pos_df_len',len(df['pos_line']))

    #'''
    fw = codecs.open("./mldata/alltxt.txt","w",encoding="utf-8")
    [fw.write(serif.replace("\n","")+"\n") for serif in serif_list]
    fw.close()

    subprocess.call("mecab -O wakati -b 50000 ./mldata/alltxt.txt  > ./mldata/alltxt_mecab.txt",shell=True)
    subprocess.call("nkf -w8 --overwrite ./mldata/alltxt_mecab.txt",shell=True)
    writeEach("./mldata/alltxt_mecab.txt",df_dict)

def writeEach(writefile,df_dict):
    # all_lines = codecs.open("./mldata/alltxt_spm{}.txt".format(vocab_n),encoding='utf-8').readlines()
    all_lines = codecs.open(writefile,encoding='utf-8').readlines()
    all_lines = [line.replace("\n","").replace("▁","").strip() for line in all_lines]
    print("alllines",len(all_lines))
    serif_lines = all_lines[:len(all_lines)//3]
    pre_lines = all_lines[len(all_lines)//3:2*len(all_lines)//3]
    pos_lines = all_lines[2*len(all_lines)//3:]

    print('serif_list_len', len(serif_lines))
    print('pre_list_len', len(pre_lines))
    print('pos_list_len', len(pos_lines))
    assert len(serif_lines)==len(pre_lines)==len(pos_lines)

    # df_dict = {"title":title_list,"speaker":speaker_list,"pre_line":pre_list,"serif":pos_list}
    df_dict["pre_line_spm"]=pre_lines
    df_dict["serif_spm"]=serif_lines
    df_dict["pos_line_spm"]=pos_lines
    df = pandas.DataFrame(df_dict)
    df = df[["title","speaker","pre_line","serif","pos_line","pre_line_spm","serif_spm","pos_line_spm"]]
    if 'spm' in writefile:
        df.to_csv("./mldata/speaker_predict_spm.csv")
    else:
        df.to_csv("./mldata/speaker_predict_mecab.csv")


if __name__=="__main__":
    pass