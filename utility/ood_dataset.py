import copy
import random
import os
from flair.data import DT, FlairDataset, Sentence, Tokenizer
from flair.tokenization import SegtokTokenizer, SpaceTokenizer

def create_ood_test_set(corpus, ood_corpus,  random_seed=123, ood_ratio=0.5, ori_save_folder=None, ood_save_folder=None):
    tokenizer_method = SpaceTokenizer()
    if not os.path.exists(ood_save_folder):
        os.makedirs(ood_save_folder)
    # build corpus type entity
    ori_corpus_entities = find_all_entities(corpus)
    ori_corpus_entities_set = extract_set(ori_corpus_entities, 0)

    # build ood corpus entity
    ood_corpus_entities = find_all_entities(ood_corpus)
    ood_corpus_entities_set = extract_set(ood_corpus_entities, 0)

    # cal div set ood_corpus/corpus
    ood_uni_entities_set = ood_corpus_entities_set - ori_corpus_entities_set
    ood_uni_entities_len_dict = build_len_based_dict(ood_uni_entities_set)


    apply_ood = True
    mid = corpus.test[0]

    # need operation
    if apply_ood:
        ori_mid_text = mid.text
        ori_mid_label = mid.get_spans('ner')
        span_num = len(ori_mid_label)

        rep_num = random.randint(1, span_num) # every sent will have this rep-num entities replaced.
        rep_id_list = random.sample(range(span_num), rep_num) # random extract two elements to replace

        new_mid_text = copy.deepcopy(ori_mid_text)
        new_mid_text_list = new_mid_text.split(" ")
        # new_mid_label = copy.deepcopy(ori_mid_label)
        new_mid_label = []
        for ori_one_label in ori_mid_label:
            new_mid_label.append(ori_one_label)
        for rep_id in rep_id_list:
            ori_tri = ori_mid_label[rep_id]
            ori_tri_text_len = len(ori_tri.text.split(" "))
            if ori_tri_text_len in ood_uni_entities_len_dict.keys():
                sampled_ood_tri_text = random.sample(ood_uni_entities_len_dict[ori_tri_text_len], 1)[0]  # generate a random type
            else:
                continue
            ori_tri_lw_word_index, ori_tri_hg_word_index = read_word_level_index(str(ori_tri))
            new_mid_text_list[ori_tri_lw_word_index:ori_tri_hg_word_index] = sampled_ood_tri_text.split(" ")[:]

            # construct label
            mid_sent = ' '.join(new_mid_text_list)
            mid_sent = Sentence(mid_sent, use_tokenizer=tokenizer_method)
            mid_span = mid_sent[ori_tri_lw_word_index:ori_tri_hg_word_index]
            mid_span.add_label('ner', value='OOD', score=1.0)
            new_mid_label[rep_id] = mid_span # have wrong start-end position

        # after for loop, the text and new labels are ready.
        new_mid_text = ' '.join(new_mid_text_list)


        new_mid_sent = Sentence(new_mid_text, use_tokenizer=tokenizer_method)
        for label in new_mid_label:
            cur_tri_lw_word_index, cur_tri_hg_word_index = read_word_level_index(str(label))
            cur_span = new_mid_sent[cur_tri_lw_word_index:cur_tri_hg_word_index]
            cur_span.add_label('ner', value=label.get_label('ner').value, score=label.get_label('ner').score) # calibrate the wrong start-end position
            # new_mid_sent.add_label('ner', cur_span)


        new_mid_label_list = ['O'] * len(new_mid_text_list)
        for label in new_mid_label:
            cur_tri_lw_word_index, cur_tri_hg_word_index = read_word_level_index(str(label))
            for m in range(cur_tri_lw_word_index, cur_tri_hg_word_index):
                new_mid_label_list[m] = label.get_label('ner').value
        return new_mid_sent, new_mid_text_list, new_mid_label_list 









    return corpus

def find_all_entities(corpus):
    res = []
    tr = corpus.train
    val = corpus.dev
    te = corpus.test
    res.extend(find_entities_once(tr))
    res.extend(find_entities_once(val))
    res.extend(find_entities_once(te))
    return res

def find_entities_once(list_sent):
    res = []
    for sent in list_sent:
        text_label_eles = sent.get_labels('ner')
        for text_label_ele in text_label_eles:
            mid_ele = (text_label_ele.data_point.text, text_label_ele.value, text_label_ele.score)
            res.append(mid_ele)
    return res

def extract_set(list_cell, id):
    res = []
    for cell in list_cell:
        mid = cell[id]
        res.append(mid)
    res = set(res)
    return res

def build_len_based_dict(entity_set):
    entity_list = list(entity_set)
    res = {}
    for ele in entity_list:
        # ele_len = len(ele) # this is character-level length
        ele_len = len(ele.split(" "))
        if ele_len not in res.keys():
            res[ele_len] = [ele]
        else:
            res[ele_len].append(ele)
    return res


def read_word_level_index(un_identifier):
    # un_identifier = predicted_span.unlabeled_identifier  # 'Span[1:2]: "asian"'
    lf_braket_idx = un_identifier.index('[')
    rg_braket_idx = un_identifier.index(']')
    comma_idx = un_identifier.index(':')
    low_idx = int(un_identifier[lf_braket_idx + 1:comma_idx])
    high_idx = int(un_identifier[comma_idx + 1:rg_braket_idx])
    return low_idx, high_idx

def count_removed_class_dataset(corpus, remove_word_list, return_new_set=False):
    exclued_num, inclued_num = 0, 0
    exclued_set = []
    inclued_set = []


    tr_ex_num, tr_in_num, tr_ex_set, tr_in_set = count_remove_one_class(corpus.train, remove_word_list, return_new_set)
    dev_ex_num, dev_in_num, dev_ex_set, dev_in_set = count_remove_one_class(corpus.dev, remove_word_list, return_new_set)
    te_ex_num, te_in_num, te_ex_set, te_in_set = count_remove_one_class(corpus.test, remove_word_list, return_new_set)
    exclued_num = exclued_num + tr_ex_num + dev_ex_num + te_ex_num
    inclued_num = inclued_num + tr_in_num + dev_in_num + te_in_num
    if return_new_set == True:
        exclued_set.extend(tr_ex_set)
        exclued_set.extend(dev_ex_set)
        exclued_set.extend(te_ex_set)

        inclued_set.extend(tr_in_set)
        inclued_set.extend(dev_in_set)
        inclued_set.extend(te_in_set)


    if return_new_set == True:
        return exclued_num, inclued_num, exclued_set, inclued_set
    else:
        return exclued_num, inclued_num #, new_tr, new_dev, new_te

def count_remove_one_class(tr_set, remove_word_list, return_new_set=False):
    exclued_num, inclued_num = 0, 0
    exclued_set = []
    inclued_set = []
    for sent in tr_set:
        labels = sent.get_labels('ner')
        has_remove_word = False
        for remove_word in remove_word_list:
            for label in labels:
                if label.value == remove_word:
                    has_remove_word = True
                    inclued_num += 1
                    if return_new_set:
                        inclued_set.append(sent)
                    break
            if has_remove_word == True:
                break
        if has_remove_word == False:
            exclued_num += 1
            if return_new_set:
                exclued_set.append(sent)
    return exclued_num, inclued_num, exclued_set, inclued_set


def replace_lo_label_ood(included_set, leave_out_label_list):
    res = []
    for sent in included_set:
        for sent_label in sent.get_labels('ner'):
            for lo_label in leave_out_label_list:
                if sent_label.value == lo_label:
                    sent_label.value = 'OOD'
        res.append(sent)
    return res

def merge_ood_corpus(corpus, excluded_set, included_set, exclude_split_ratio):
    random_seed = 1
    exclu_len = len(excluded_set)
    inclu_len = len(included_set)

    random.Random(random_seed).shuffle(excluded_set) # set random seed = 1


    tr_set = excluded_set[0:int(exclu_len*exclude_split_ratio[0])]
    dev_set = excluded_set[int(exclu_len*exclude_split_ratio[0]) : int(exclu_len*exclude_split_ratio[1])]
    te_set = excluded_set[int(exclu_len*exclude_split_ratio[1]) : int(exclu_len*exclude_split_ratio[2])]
    te_set.extend(included_set)
    random.Random(random_seed).shuffle(te_set)


    # handle data structure of corpus
    corpus._train = tr_set
    corpus._dev = dev_set
    corpus._test = te_set


    return corpus