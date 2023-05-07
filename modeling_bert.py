# encoding:utf-8
import json
from collections import defaultdict
import numpy as np
import re
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from crf import CRF
from nltk.corpus import stopwords
from dataprocess_core import EnglishTextUtil
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel,BertModel

MODEL_NAME="/home/liudq/bert_chn/bert_en"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME) # encode_plus()

START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
EVENTTYPE_IDX_MAP = {"N": 0, "Life": 1, "Movement": 2, "Transaction": 3, "Business": 4, "Conflict": 5, "Contact": 6,
                 "Personnel": 7, "Justice": 8, START_TAG: 9, STOP_TAG: 10}
# POSTAG_IDX_MAP = {"": 0, "n": 1, "v": 2, "a": 3, "<PAD>": 4}
STOPCHARS = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
pos_tags=open("postag.txt",encoding='utf').read().split('\n')
POSTAG_IDX_MAP={}
for i,p in enumerate(pos_tags):
    POSTAG_IDX_MAP[p]=i

dp_tags=open("dps.txt",encoding='utf').read().split('\n')
DP_MAP={}
for i,p in enumerate(dp_tags):
    DP_MAP[p]=i



def pad_datas(inputdata,use_bert=False):
    wordlist = inputdata["words"]
    taglist = inputdata.get("tags")


    def tokenize_and_preserve_labels(sentence, text_labels):
        tokenized_sentence = []
        labels = []
        if bool(text_labels):
            for word, label in zip(sentence, text_labels):
                tokenized_word = tokenizer.tokenize(word)  # id
                n_subwords = len(tokenized_word)  # 1
                # 将单个字分词结果追加到句子分词列表
                tokenized_sentence.extend(tokenized_word)
                # 标签同样添加n个subword，与原始word标签一致
                labels.extend([label] * n_subwords)
            return tokenized_sentence, labels
        else:
            return tokenizer.tokenize(' '.join(sentence)),[]


    if use_bert:
        id_list=[]
        label_list=[]
        mask_list=[]
        for i,words in enumerate(wordlist):
            wordpiece,labels=tokenize_and_preserve_labels(words,taglist)
            # bert_feats.append(wordpiece)
            # labels.append(label)
            tokenized_sentence = ["[CLS]"] + wordpiece + ["[SEP]"]  # add special tokens
            labels.insert(0, START_TAG)  # 给[CLS] token添加O标签
            labels.insert(-1, STOP_TAG)  # 给[SEP] token添加O标签
            maxlen = 256
            if (len(tokenized_sentence) > maxlen):
                # 截断
                tokenized_sentence = tokenized_sentence[:maxlen]
                labels = labels[:maxlen]
            else:
                tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(maxlen - len(tokenized_sentence))]
                labels = labels + [STOP_TAG for _ in range(maxlen - len(labels))]
            attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
            ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
            label_ids = [EVENTTYPE_IDX_MAP.get(tag, 0) for tag in labels]
            id_list.append(ids)
            mask_list.append(attn_mask)
            label_list.append(label_ids)
        return {'ids':  torch.tensor(id_list, dtype=torch.long),'mask': torch.tensor(mask_list, dtype=torch.long),'targets': label_list}


def preprocess_texts(texts, vocab,use_bert=False):
    texts_words = []
    texts_postags = []
    texts_heads=[]
    texts_rels=[]
    etu_processer = EnglishTextUtil()
    for text in texts:
        textinfo = etu_processer.core_process(text)  # 四元组 word, norm_word, tag, offset, wordnet_tag
        words = [w for w, l, p in textinfo["words"]]

        texts_words.append(words)

    outpudata = pad_datas({"words": texts_words, "tags": []},use_bert=True)
    return outpudata

# 载入数据
def load_eventdata(datafile,use_bert=False):
    with open(datafile, 'r') as r:
        source_datas = [json.loads(line) for line in r]
    print("texts", len(source_datas))

    out_wordslist = []
    out_taglist = []
    out_postaglist = []  # 词性标注
    out_headlist=[]
    out_rellist=[]
    for sentence_data in source_datas:
        text = sentence_data["text"]
        anchor = sentence_data["anchor"]
        if not bool(anchor):continue
        main_event_type = sentence_data.get('type')
        sub_event_type = sentence_data.get('subtype')
        wordsinfo = sentence_data.get('words')
        words = [item[0] for item in wordsinfo]  # 词形变换后的结果

        tags = ['N' for _ in range(len(words))]  # 缺省标签

        for k, (word, normword, postag, anchorflag) in enumerate(wordsinfo):
            if anchorflag:
                tags[k] = main_event_type

        out_wordslist.append(words)
        out_taglist.append(tags)

    print("loaded datas", len(out_wordslist), len(out_taglist))

    # 计算词表，计算最大序列长度，词表中前2个保留给
    special_words = [START_TAG, STOP_TAG, UNK]
    max_seqlen = max([len(words) for words in out_wordslist]) + len(special_words) - 1 #去掉UNK

    inputdata = {"words": out_wordslist, "postags": out_postaglist, "tags": out_taglist,"heads": out_headlist,"rels":out_rellist}
    padded_datas = pad_datas(inputdata,use_bert=use_bert)
    if use_bert:
        return padded_datas

# 把数据集划分成训练集、验证集和测试集， inputdatas是dict，每个元素是一个list
def split_dataset(inputdatas, train_ratio, val_ratio):
    keys = list(inputdatas.keys())
    total = len(inputdatas[keys[0]])
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # 随机序号列表
    idxes = list(range(total))
    np.random.seed(42)
    np.random.shuffle(idxes)
    np.random.shuffle(idxes)
    np.random.shuffle(idxes)

    trainset = dict()
    valset = dict()
    testset = dict()
    for key, datas in inputdatas.items():
        trainset[key] = [datas[i] for i in idxes[0:train_size]]
        valset[key] = [datas[i] for i in idxes[train_size:train_size+val_size]]
        testset[key] = [datas[i] for i in idxes[train_size+val_size:]]

    print("split_dataset", total, train_size, val_size)
    return trainset, valset, testset

class BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.batch_first=True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.lstm = nn.LSTM(768, 100, num_layers=1, bidirectional=True, batch_first=self.batch_first)
        self.crf = CRF(config.num_labels, batch_first=self.batch_first)
        self.classifier = nn.Linear(200, config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None,labels=None):
        embeds = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            head_mask=head_mask)[0]
        # sequence_output = self.dropout(sequence_output)
        sequence_output, (ht, hc) = self.lstm(embeds)
        logits = self.classifier(sequence_output)
        outputs = (logits,)

        if labels is not None:  # 训练的情况
            # , mask=attention_mask
            # print("logits", logits)
            # print("labels", labels)
            loss = self.crf(emissions=logits, tags=labels)
            predicts = self.crf.decode(logits)
            # print("train res:", labels.shape, predicts.shape, "label", labels[0])
            # print("predicts", predicts[0][0])
            outputs = (-1 * loss,) + outputs
        else:
            # print("sequence_output", sequence_output.shape)
            # print("logits", logits.shape, logits)
            tags = self.crf.decode(logits)
            # print("tags", tags)
            tags = tags.squeeze(0).cpu().numpy().tolist()
            outputs = tags
        #                     tags = self.model.crf.decode(logits, inputs['attention_mask'])
        #                     tags = tags.squeeze(0).cpu().numpy().tolist()
        return outputs



EMBEDDING_DIM = 100
HIDDEN_DIM = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=3e-5 
crf_learning_rate=1e-3

def eval_dataset(model, dataset, use_postag, use_dp):
    batch_size = 60
    datakeys = list(dataset.keys())
    # print("eval dataset", datakeys, len(dataset[datakeys[0]]))
    wordsids = dataset["ids"]
    mask= dataset["mask"]
    tagids=dataset["targets"]

    start_tagid = EVENTTYPE_IDX_MAP[START_TAG]
    stop_tagid = EVENTTYPE_IDX_MAP[STOP_TAG]

    model.eval()

    total_errcnt = 0    # 词语级别错误数
    word_acc = 0  # 词语级别的正确率
    sentence_correct_count = 0  # 句子级别的正确数
    for start in range(0, len(wordsids), batch_size):
        end = min(len(wordsids), start + batch_size)
        batch_wordsids = wordsids[start:end]
        batch_masks = mask[start:end]
        targets =tagids[start:end]
        wordid_feats = torch.stack(batch_wordsids, 0).to(device)
        mask_feat = torch.stack(batch_masks, 0).to(device)
        predicts = model(wordid_feats, attention_mask=mask_feat)


        for ti, pi in zip(targets, predicts):
            # print("++>", ti)
            # print("==>", pi)
            accarr = [1 if targ == pred else 0 for targ, pred in zip(ti, pi)
                   if (targ not in [start_tagid, stop_tagid])]  # and (targ != 0 or pred != 0)

            # class_accarr = [1 if targ == pred else 0 for targ, pred in zip(ti, pi)
            #           if (targ not in [start_tagid, stop_tagid])]  # and (targ != 0 or pred != 0)

            # recall = 0 if origin == 0 else (right / origin)
            # precision = 0 if found == 0 else (right / found)
            # f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)

            acc = sum(accarr) / len(accarr)
            total_errcnt += int(len(accarr) - sum(accarr))
            if sum(accarr) == len(accarr):
                sentence_correct_count += 1
            # print("**", acc, sum(accarr), len(accarr), accarr)
            word_acc += acc
    sentence_acc = sentence_correct_count / len(wordsids)  # 句子级别正确率
    word_acc = word_acc / len(wordsids)  # 单词级别正确率
    return word_acc, sentence_acc, total_errcnt


def train(use_postag=False,use_dp=False,use_bert=False):
    datafile = "train.json"

    # 全部数据
    inputdatas = load_eventdata(datafile,use_bert=use_bert)

    # 划分数据集
    trainset, devset, testset = split_dataset(inputdatas, 0.8, 0.1)
    num_tags = len(EVENTTYPE_IDX_MAP)


    model = BiLSTM_CRF.from_pretrained(MODEL_NAME, num_labels=num_tags)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lr},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate}
    ]

    optimizer =  torch.optim.Adam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    # param_optimizer = list(model.named_parameters())
    # opti_params = [p for n, p in param_optimizer if p.requires_grad]
    # 
    # optimizer = torch.optim.Adam(opti_params, lr=3e-4)
    # optimizer2 = optimizer

    batch_size = 30
    epoches = 50

    model = model.to(device)
    model.train()
    # print("model", model)

    datakeys = list(trainset.keys())
    print("trainset", datakeys, len(trainset[datakeys[0]]))
    wordsids = trainset["ids"]
    mask= trainset["mask"]
    targets=trainset["targets"]

    best_acc = 0
    best_epoch = 0

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(epoches):  # again, normally you would NOT do 300 epochs, it is toy data
        epoch_loss = 0
        model.train()
        for start in range(0, len(wordsids), batch_size):
            end = min(len(wordsids), start+batch_size)
            batch_wordsids = wordsids[start:end]
            batch_masks=mask[start:end]
            batch_tagids = targets[start:end]
            # print(batch_masks)
            # print(batch_tagids)
            # print(batch_wordsids)

            wordid_feats = torch.stack(batch_wordsids,0).to(device)
            mask_feat= torch.stack(batch_masks,0).to(device)
            # print()
            lables = torch.tensor(batch_tagids,dtype=torch.long).to(device)
            loss, outputs = model(wordid_feats,attention_mask=mask_feat,labels=lables)
            # Step 3. Run our forward pass.
            # print('feats', feats.shape, feats[0])
            # print('targets', targets.shape, targets[0])


            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            if epoch < 200:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # else:
            #     optimizer2.zero_grad()
            #     loss.backward()
            #     optimizer2.step()

            epoch_loss += loss

        word_acc, sentence_acc, errcnt = eval_dataset(model, devset, use_postag,use_dp)
        print("epoch %d train loss %.3f eval word acc: %.5f  sentence acc: %.5f errcnt %d [best:%d %.5f]"
                %(epoch, epoch_loss, word_acc, sentence_acc, errcnt, best_epoch, best_acc))

        # 保存最好的模型
        if word_acc > best_acc:
            # torch.save(model, 'myModel.pt')
            best_acc = word_acc
            best_epoch = epoch
        if epoch%5==0:
            torch.save(model, 'myModel.pt')
    word_acc, sentence_acc, errcnt = eval_dataset(model, testset, use_postag,use_dp)
    print("test word acc: %.5f, sentence acc: %.5f, errcnt %d" %(word_acc, sentence_acc, errcnt))
    # torch.save(model, 'myModel.pt')

    return model, vocab


def load_model(modelfile='myModel.pt'):
    model = torch.load(modelfile)
    return model

def predict(model=None, vocab=None, use_postag=True,use_bert=True):
    if model is None or vocab is None:
        model = load_model()

    # 模型
    model.eval()
    model.to(device)

    # Check predictions after training
    print("Please input...")
    while True:
        text = input()
        featdatas = preprocess_texts([text], vocab,use_bert=use_bert)
        wordsids =  featdatas ["ids"]
        mask =  featdatas ["mask"]
        wordid_feats = wordsids.to(device)
        mask_feat = mask.to(device)

        # print("feat", feat)
        with torch.no_grad():
            res = model(wordid_feats, attention_mask=mask_feat)
            print(res)


def batch_test(model=None, vocab=None, use_bert=True):
    if model is None or vocab is None:
        vocab, model = load_model()

    # 模型
    model.eval()
    model.to(device)

    datafile = "eventdata.json"
    vocab, max_seqlen, inputdatas = load_eventdata(datafile)
    wordslist = inputdatas["words"]
    texts = [" ".join(words) for words in wordslist]

    for text in texts:
        featdatas = preprocess_texts([text], vocab,use_bert=True)

        wordsids =  featdatas ["ids"]
        mask =  featdatas ["mask"]
        wordid_feats = wordsids.to(device)
        mask_feat = mask.to(device)
        with torch.no_grad():
            res = model(wordid_feats, attention_mask=mask_feat)
            # if res[0]!=9 and res[-1]!=10:
            flag = False
            for i in range(1, len(res) - 1):
                if res[i] != 0:
                    flag = True
            if flag:
                print(text, res)

import sys
if len(sys.argv) > 1 and sys.argv[1] == "train":
    model, vocab = train(use_postag=False,use_dp=False,use_bert=True)
    predict(model, vocab)
elif len(sys.argv) > 1 and sys.argv[1] == "test":
    model, vocab = None, None
    batch_test(model, vocab,use_bert=True)
else:
    model, vocab = None, None
    predict(model, vocab,use_postag=False,use_bert=True)

    """sudo /home/liudq/anaconda3/bin/python modeling.py
    # ori:test word acc: 0.94138, sentence acc: 0.36990, errcnt 369
    pos:test word acc: 0.95776, sentence acc: 0.38769, errcnt 297
    sep:test word acc: 0.96073, sentence acc: 0.35692, errcnt 277
    """
