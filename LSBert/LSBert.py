
import argparse
import csv
import os
import random
import math
import sys
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM

from sklearn.metrics.pairwise import cosine_similarity as cosine


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import numpy as np
import torch
import nltk
from scipy.special import softmax
#from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def get_score(sentence,tokenizer,maskedLM):
    '''
    :param sentence: the (part of the) sentence
    :param tokenizer: the BERT tokenizer
    :param maskedLM: the BERT model
    :return:
    '''
    tokenized_input = tokenizer.tokenize(sentence)

    len_sen = len(tokenized_input)

    START_TOKEN = '[CLS]'
    SEPARATOR_TOKEN = '[SEP]'

    tokenized_input.insert(0, START_TOKEN)
    tokenized_input.append(SEPARATOR_TOKEN)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_input)

    sentence_loss = 0
    
    for i,word in enumerate(tokenized_input):

        if(word == START_TOKEN or word==SEPARATOR_TOKEN):
            continue

        original_word = tokenized_input[i]
        tokenized_input[i] = '[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_input)])
        mask_input = mask_input.to('cpu')
        with torch.no_grad():
            att, pre_word =maskedLM(mask_input)
        word_loss = cross_entropy_word(pre_word[0].cpu().numpy(),i,input_ids[i])
        sentence_loss += word_loss
        tokenized_input[i] = orignial_word
        
    return np.exp(sentence_loss/len_sen)


def LM_score(difficult_word, difficult_word_context, substitution_candidates, tokenizer, model):
    '''
    :param difficult_word: the difficulted, masked, word
    :param source_context: the context of the difficult word
    :param substitution_candidates: the candidates for substitution 
    :param tokenizer: the BERT tokenizer
    :param model: the BERT model
    :return:
    '''
    new_sentence = ''

    for context in source_context:
        new_sentence += context + " "
    
    new_sentence = new_sentence.strip()
    print("new sentence: ", new_sentence)
    LM = []

    for substibution in substitution_selection:
        
        sub_sentence = new_sentence.replace(difficult_word,substibution)

        
        #print(sub_sentence)
        score = get_score(sub_sentence,tokenizer,maskedLM)

        #print(score)
        LM.append(score)

    return LM


def preprocess_SR(difficult_word, generated_subs, embedding_dict, embedding_vector, word_count):
    '''
    :param difficult_word: the difficulted, masked, word
    :param generated_subs: the generated simplifications
    :param embedding_dict: the words in the embedding model
    :param embedding_vector: the corresponding vectors in the embedding model
    :param word_count: the word frequencies
    :return:
    '''
    selected_subs = []
    similarity_scores=[]
    count_scores=[]
    frequency_score = 10

    # If it is in the frequent words dict, it gets the corresponding count
    if difficult_word in word_count:
        frequency_score = word_count[difficult_word]

    in_embedding = True

    # Look up the embedding value of the complex word
    if(difficult_word not in embedding_dict):
        in_embedding = False
    else:
        embedding_value = embedding_vector[embedding_dict.index(difficult_word)].reshape(1,-1)

    # Iterating over the candidate substitutions and attribution values
    for sub in generated_subs:

        # If the substitution is so rare that it's not in the word_count dict, it is not taken into account
        if sub not in word_count:
            continue
        # Otherwise, the count feature is set to its count val
        else:
            sub_count = word_count[sub]
           
        # If there is an embedding value for the difficult word and candidate word, the similarity is calculated
        # If there is an enmbedding value for the difficult word, but not for the candidate word, the candidate is discarded
        if in_embedding:
            if sub not in embedding_dict:
                continue

            token_embedding_index = embedding_dict.index(sub)
            similarity = cosine(embedding_value, embedding_vector[token_embedding_index].reshape(1,-1))

            similarity_scores.append(similarity)

        selected_subs.append(sub)
        count_scores.append(sub_count)

    return selected_subs,similarity_scores,count_scores


def substitution_ranking(difficult_word, difficult_word_context, candidate_words, embedding_vocab, embedding_vectors, word_count,
                         tokenizer, model, annotations):
    '''
    :param difficult_word: the difficult word that has been masked
    :param difficult_word_context: the words in the context of the difficult word
    :param candidate_words: the BERT-generated simplifications
    :param embedding_vocab: the words in the embedding model
    :param embedding_vectors: the corresponding vectors in the embedding model
    :param word_count: the frequency file
    :param tokenizer: the used BERT tokenizer
    :param model: the used BERT MLM
    :param annotations: the annotations that humans have given as a simplification for the target word
    :return: pre_word:
    '''


    substitution_candidates, similarity_scores, frequency_scores = preprocess_SR(difficult_word, candidate_words, embedding_vocab, embedding_vectors, word_count)

    # If there are no candidates left, just return the difficult word
    if len(substitution_candidates) == 0:
        return difficult_word

    # If there are cosine scores calculated:
        if len(similarity_scores) > 0:
            seq = sorted(similarity_scores, reverse=True)
            similarity_rank = [seq.index(v) + 1 for v in similarity_scores] # This describes for each subs candidate the position in the ranking

        sorted_count = sorted(count_scores, reverse=True)

        count_rank = [sorted_count.index(v) + 1 for v in count_scores] # This describes for each subs candidate the position in the ranking

    lm_score = LM_score(difficult_word, difficult_word_context, substitution_candidates, tokenizer, model)

    # print(lm_score)

    rank_lm = sorted(lm_score)
    lm_rank = [rank_lm.index(v) + 1 for v in lm_score]

    bert_rank = []
    for i in range(len(substitution_candidates)):
        bert_rank.append(i + 1)

    if len(sis_scores) > 0:
        all_ranks = [bert + sis + count + LM for bert, sis, count, LM in zip(bert_rank, sis_rank, count_rank, lm_rank)]
    else:
        all_ranks = [bert + count + LM for bert, count, LM in zip(bert_rank, count_rank, lm_rank)]
    # all_ranks = [con for con in zip(context_rank)]

    pre_index = all_ranks.index(min(all_ranks))
    pre_word = substitution_candidates[pre_index]

    return pre_word


def substitution_generation(difficult_word, predicted_tokens,probabilities, ps, selection_size=10):
    '''
    :param difficult_word: the difficult, masked, target word
    :param pre_tokens: 20 most likely tokens generated by BERT
    :param pre_scores: the probabilities of those 20 generated substitutions
    :param ps: the porter stemmer
    :param num_selection: the number of likely substitutions to return
    :return:
    '''
    selected_tokens = []

    difficult_word_stem = ps.stem(difficult_word)

    assert selection_size <= len(predicted_tokens)

    # Loop over all predicted tokens
    for i in range(len(predicted_tokens)):
        predicted_token = predicted_tokens[i]

        # If BERT predicts a subword, it is not taken into account
        if predicted_token[0:2] == "##":
            continue

        # If BERT predicts the actual word, it is not taken into account
        if (predicted_token == difficult_word):
            continue

        # If the stem of the predicted word is the same as that of the actual, it is not taken in to account
        predicted_token_stem = ps.stem(predicted_token)
        if (predicted_token_stem == difficult_word_stem):
            continue

        # If the predicted token is very similar to the actual, it is not taken into account
        if (len(predicted_token_stem) >= 3) and (predicted_token_stem[:3] == difficult_word_stem[:3]):
            continue

        # If the predicted token is not a subword and is different enough, it is added to the actual
        selected_tokens.append(predicted_token)

        # If enough tokens have been deleted, it's enough
        if (len(selected_tokens) == selection_size):
            break

    # If none are good enough for the criteria, the first ones until the selection size are chosen
    if (len(selected_tokens) == 0):
        selected_tokens = predicted_tokens[0:selection_size + 1]

    assert len(selected_tokens) > 0

    return selected_tokens


def convert_whole_word_to_feature(bert_sent, mask_position, seq_length, tokenizer):
    '''
    If a single nltk token is tokenized into multiple subwords for BERT,
    this function transforms a data file into a list of `InputFeature`s.
    :param final_tokens: [CLS] sentence [SEP] masked sentence [CLS]
    :param mask_position: index of the difficult word/ mask
    :type: mask_position: list
    :param seq_length: maximum length of BERT sequence
    :param tokenizer: used BERT tokenizer
    :return:
    '''


    final_tokens = ["[CLS]"] #This will be filled with the tokens?
    input_type_ids = []
    input_type_ids.append(0)
    for token in bert_sent:         # Build the sentence back up
        final_tokens.append(token)
        input_type_ids.append(0)    # And a corresponding list (first time zeroes) todo: what for?

    final_tokens.append("[SEP]")    # The sentence ends with [SEP]
    input_type_ids.append(0)

    for token in bert_sent:
        final_tokens.append(token)  # Add them a second time
        input_type_ids.append(1)    # But then with 1s

    final_tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    index = 0
    count = 0

    # The number of subwords that make up the token:
    len_masked_subwords = len(mask_position)

    # Replacing the all subwords with one [MASK] token
    while count in range(len_masked_subwords):
        index = len_masked_subwords - 1 - count

        pos = mask_position[index]
        if index == 0:
            final_tokens[pos] = '[MASK]'
        else:
            del final_tokens[pos]
            del input_type_ids[pos]

        count += 1

    # Convert the final tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=final_tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def convert_token_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    '''
       If a single nltk token is tokenized into a single token by BERT,
       this function transforms a data file into a list of `InputFeature`s.
       :param final_tokens: [CLS] sentence [SEP] masked sentence [CLS]
       :param mask_position: index of the difficult word/ mask
       :type: mask_position: list
       :param seq_length: maximum length of BERT sequence
       :param tokenizer: used BERT tokenizer
       :return:
       '''
    # tokens_a = tokenizer.tokenize(sentence)
    # print(mask_position)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    if isinstance(mask_position, list):
        for pos in mask_position:
            true_word = true_word + tokens[pos]
            tokens[pos] = '[MASK]'
    else:
        true_word = tokens[mask_position]
        tokens[mask_position] = '[MASK]'

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def extract_context(words, mask_index, window):
    '''
    This function extracts the context of the difficult word in a sentence
    :param words: nltk tokenized sentence
    :type words: list
    :param mask_index: index of the difficult word
    :param window: size of context window
    :type window: int
    :return: context: the words surrounding the difficult words
    '''

    sent_length = len(words)

    half_window = int(window/2)

    # Check that the difficult word is located inside the sentence
    assert mask_index>=0 and mask_index<sent_length

    context = ""

    # If the sentence is shorter than the window, the whole sentence is returned
    if sent_length<=window:
        context = words

    # if the mask is in the first half and the
    elif mask_index < sent_length - half_window and mask_index >= half_window:
        context = words[mask_index - half_window : mask_index + half_window + 1]
    elif mask_index<half_window:
        context = words[0:window]
    elif mask_index>=sent_length-half_window:
        context = words[sent_length-window:sent_length]
    else:
        print("Wrong!")

    return context


def convert_sentence_to_token(sentence, seq_length, tokenizer):
    '''
    Function to align the raw texts with BERT tokenizer
    :param sentence: original sentence
    :param seq_length: maximal sequence length that can be fed to BERT
    :param tokenizer: BERT tokenizer corresponding to used model
    :return: bert_sent: subword tokenized sentence by BERT
    :return: nltk_sent: tokenized sentence by NLTK
    :return: position2: list with the token: subword mapping of BERT- nltk
    '''

    print("sentence: ", sentence)
    # Use BERT tokenizer to tokenize text
    bert_sent = tokenizer.tokenize(sentence.lower())
    # print("BERT tokenized sent:", bert_sent)

    # The bert text must be smaller than the maximal length-2 (because of the the CLS tokens)
    assert len(bert_sent) < seq_length - 2

    # Then tokenize the sentence with nltk
    nltk_sent = nltk.word_tokenize(sentence.lower())
    # print("nltk tokenized sent", nltk_sent)

    position2 = []

    token_index = 0

    # This is the position where the new sentence starts?
    start_pos = len(bert_sent) + 2

    pre_word = ""

    # Loop over the nltk tokenized words, make some corrections
    for i, nltk_word in enumerate(nltk_sent):
        # print("i ", i)
        # print("word ", nltk_word)
        # print("position2", position2)
        # print("token_index",token_index)
        # print("start_pos", start_pos)
        # print("pre_word", pre_word)

        if nltk_word == "n't" and pre_word[-1] == "n":
            nltk_word = "'t"
            # print(1,word)

        if bert_sent[token_index] == "\"":
            len_token = 2
            # print(2,tokenized_text[token_index] )
        else:
            len_token = len(bert_sent[token_index])
            # print(3,tokenized_text[token_index] )

        if bert_sent[token_index] == nltk_word or len_token >= len(nltk_word):
            # print(4,tokenized_text[token_index])
            position2.append(start_pos + token_index)
            pre_word = bert_sent[token_index]

            token_index += 1
        else:
            new_pos = []
            new_pos.append(start_pos + token_index)

            new_word = bert_sent[token_index]

            while new_word != nltk_word:

                token_index += 1

                new_word += bert_sent[token_index].replace('##', '')

                new_pos.append(start_pos + token_index)

                if len(new_word) == len(nltk_word):
                    break
            token_index += 1
            pre_word = new_word

            position2.append(new_pos)
    # print(bert_sent, nltk_sent, position2)
    return bert_sent, nltk_sent, position2


def read_eval_dataset_lexmturk(data_path, is_label=True):
    # Todo: check meaning of is_label
    '''
    Function to read in the lex.mturk.txt data set.
    :param data_path: location of the lex.mturk.txt file
    :param is_label: indicates if you are interested in how the difficult words have been annotated (in case of evaluation I guess)
    :return: sentences: list of sentences
    :return: difficult_words: list of the difficult words
    :return: substitutinos: list of lists of the annotated simplifications
    '''
    # To read in the lex.mturk dataset
    sentences = []
    difficult_words = []
    substitutions = []
    id = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if id == 1:  # This is the header, we want to skip that
                    continue
                if not line:
                    break
                sentence, words = line.strip().split('\t', 1)
                # print(sentence)
                difficult_word, labels = words.strip().split('\t', 1)
                label_list = labels.split('\t')

                sentences.append(sentence)
                difficult_words.append(difficult_word)

                # Adding every suggested label to a list (once)
                one_labels = []
                for lab in label_list:
                    if lab not in one_labels:
                        one_labels.append(lab)

                substitutions.append(one_labels)
            else:
                # If you don't want the labels
                if not line:
                    break
                sentence, difficult_word = line.strip().split('\t')
                sentences.append(sentence)
                difficult_words.append(difficult_word)
    return sentences, difficult_words, substitutions


def read_eval_index_dataset(data_path, is_label=True):
    '''
    Function to read in the BenchLS data set.
    :param data_path: location of the  file
    :param is_label: indicates if you are interested in how the difficult words have been annotated (in case of evaluation I guess)
    :return: sentences: list of sentences
    :return: difficult_words: list of the difficult words
    :return: substitutions: list of lists of the annotated simplifications
    '''
    sentences = []
    difficult_words = []
    substitutions = []

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()

            if not line:
                break
            # Collect the sentences and words
            sentence, words = line.strip().split('\t', 1)
            # Split the words into the difficult word and possible simplifications
            difficult_word, labels = words.strip().split('\t', 1)
            label_list = labels.split('\t')

            sentences.append(sentence)
            difficult_words.append(difficult_word)

            # The label annotation have indeces with them, that are unnecessary, they are thus removed
            one_labels = []
            for lab in label_list[1:]:
                if lab not in one_labels:
                    lab_id, lab_word = lab.split(':')
                    one_labels.append(lab_word)

            substitutions.append(one_labels)

    return sentences, difficult_words, substitutions


def getWordCount(word_count_path):
    '''
    :param word_count_path: location of the frequency file
    :return word2count: dictionary of words and their frequency
    :rtype: dict
    '''
    # Makes a dictionary word : freq
    word2count = {}
    with open(word_count_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        N = 0
        for i in lines:
            i = i.strip()
            if (len(i) > 0):
                i = i.split()
                if (len(i) == 2):
                    word2count[i[0]] = float(i[1])
                else:
                    print(i)
    return word2count


def getWordmap(wordVecPath):
    '''

    :param wordVecPath: location of word embedding model
    :returns:
    words: list of all words in word embedding model
    vectors: list of corresponding word vectors
    '''

    # I think that this function creates a list of all words, and a list of all embedding vectors as a np

    words = []
    vectors = []
    f = open(wordVecPath, 'r', encoding="utf-8")
    lines = f.readlines()

    for (n, line) in enumerate(lines):
        if (n == 0):
            print("Word embedding of size: ", line)
            continue
        word, vector = line.rstrip().split(' ', 1)

        vector = np.fromstring(vector, sep=' ')

        vectors.append(vector)

        words.append(word)

        # if(n==200000):
        #    break
    f.close()
    return (words, vectors)


def main():
    ''''Parsing the input, and running the first functions'''
    parser = argparse.ArgumentParser()

    # Directory of evaluation data (BenchLS/ Lexmturk/ NNSeval)
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data dir.")

    # Name of used BERT model
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese, or one of the added ones.")

    # Location of output file
    parser.add_argument("--output_SR_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory of writing substitution selection.")

    # Location of the word embedding model
    parser.add_argument("--word_embeddings",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of word embeddings")

    # Location of the word frequency file
    parser.add_argument("--word_frequency",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of word frequency.")

    ### Other parameters: ###

    # Location for caching
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    # The maximum total input sequence length after WordPiece tokenization
    parser.add_argument("--max_seq_length",
                        default=250,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # Running evaluations or not
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    # Uncased or cased model
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Batch size for evaluation
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    # Number of training epochs
    parser.add_argument("--num_selections",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")

    # Number of evaluation epochs?
    parser.add_argument("--num_eval_epochs",
                        default=1,
                        type=int,
                        help="Total number of training epochs to perform.")

    # Proportion of training to perform linear learning rate warmup for. ?
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    # Using CUDA?
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    # Local Rank?
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # Random Seed
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # Float Precision
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    # Scaling loss stabillity
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

### Location of execution: ###
    # Change in case of GPU (Check original code)
    device = "cpu"
    n_gpu = torch.cuda.device_count()
    logger.info("Device/ Training info : \n device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

### Opening/ Loading Files ###
    # Cache Location:
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),'distributed_{}'.format(args.local_rank))

    # Evaluation file:
    evaluation_file_name = args.eval_dir.split('/')[-1][:-4]

    # Output File:
    output_sr_file = open(args.output_SR_file, "a+")

    # Word Frequency file (wikipedia & a children's book):
    print("Loading the frequency file:")
    word_count_path = args.word_frequency
    # This is a dictionary with the shape word: frequency
    word_count = getWordCount(word_count_path)

    # Loading the stemmer
    ps = PorterStemmer()

    # Number of candidates for substitution
    num_selection = args.num_selections

### Start initialization of the model ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # For when there is a a GPU:
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForMaskedLM.from_pretrained(args.bert_model, output_attentions=True, cache_dir=cache_dir)

    if args.fp16:
        model.half()
    model.to(device)

### Loading in Embeddings ###
    print("Loading embeddings ...")
    embedding_path = args.word_embeddings
    # the vocabulary of the embedding model in a list, and the corresponding emb values in another
    embedding_vocab, embedding_vectors = getWordmap(embedding_path)
    print("Done loading embeddings")

### Toward Generating the Substitutions:###
    SS = []
    substitution_words = []
    bre_i = 0
    window_context = 11

    # If we want to do evaluation on the candidates and (something with the execution) TODO:What does it do?
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if evaluation_file_name=='lex.mturk':
            # Retrieve the sentences, difficult words and annotated labels
            eval_sents, difficult_words, annotated_subs = read_eval_dataset_lexmturk(args.eval_dir)
        else:
            eval_sents, difficult_words, annotated_subs = read_eval_index_dataset(args.eval_dir)

### Running the evaluation
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_sents))

    # Pytorch model in evaluation mode:
    model.eval()

    eval_size = len(eval_sents)

    # Loop over the evaluation sentences:
    for i in range(eval_size):
        print('Sentence {} rankings: '.format(i))

        # Making a mapping between BERT's subword tokenized sent and nltk tokenized sent
        bert_sent, nltk_sent, bert_token_positions = convert_sentence_to_token(eval_sents[i], args.max_seq_length, tokenizer)

        assert len(nltk_sent) == len(bert_token_positions)

        # Look up the location of the difficult word:
        mask_index = nltk_sent.index(difficult_words[i])
        # And retrieve the words surrounding it
        mask_context = extract_context(nltk_sent, mask_index, window_context)

        len_tokens = len(bert_sent)
        # This has to do with the alignment between the BERT and NLTK tokenizer again, looking up where the diffucult word is:
        mask_position = bert_token_positions[mask_index]

        # If the mask is at a sub-word-tokenized token
        if isinstance(mask_position, list):
            # This is an instance of the feature class
            feature = convert_whole_word_to_feature(bert_sent, mask_position, args.max_seq_length, tokenizer)
        else:
            feature = convert_token_to_feature(bert_sent, mask_position, args.max_seq_length, tokenizer)
        tokens_tensor = torch.tensor([feature.input_ids])

        # Something with masking/ attention
        token_type_ids = torch.tensor([feature.input_type_ids])

        # Something with masking
        attention_mask = torch.tensor([feature.input_mask])

        # And on their way to the CUDA/ CPU
        tokens_tensor = tokens_tensor.to('cpu')
        token_type_ids = token_type_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')

### Make predictions for
        with torch.no_grad():
            all_attentions, prediction_scores = model(tokens_tensor, token_type_ids, attention_mask)

        # BERT's candidates are generated (2 times the required amount) todo:why?
        if isinstance(mask_position, list): # if tokenized by BERT as multiple subwords
            predicted_top = prediction_scores[0, mask_position[0]].topk(args.num_selections * 2)

        else:
            predicted_top = prediction_scores[0, mask_position].topk(args.num_selections * 2)
            # print("predicted_top2", predicted_top[0].cpu().numpy())

        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())

        # A hard cut on the selection, leaving maximum num_selection candidates
        candidate_words = substitution_generation(difficult_words[i], predicted_tokens, predicted_top[0].cpu().numpy(), ps,
                                                  num_selection)
        print("candidate words", candidate_words)
        SS.append(candidate_words)

        pre_word = substitution_ranking(difficult_words[i], mask_context, candidate_words, embedding_vocab, embedding_vectors,word_count,tokenizer,model,annotated_subs[i])





if __name__=="__main__":
    main()
