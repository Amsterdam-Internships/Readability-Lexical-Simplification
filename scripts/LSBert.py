import logging
import argparse
import random
from nltk import PorterStemmer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import *
from transformers import BertForPreTraining, BertTokenizer
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch.nn.functional as F

import nltk
nltk.download('punkt')


def main():
    """Parsing the input, and running the first functions"""
    global complex_words
    parser = argparse.ArgumentParser()

    # Language of the model to be run:
    parser.add_argument("--language",
                        default = "nl",
                        required=True,
                        help = "The language of the model")

    # Directory of evaluation data (BenchLS/ Lexmturk/ NNSeval)
    # #ToDo Fill in the exact datasets
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data directory.")

    # Location of output file
    parser.add_argument("--output_SR_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory of writing substitution selection.")

    # # Location of the word embedding model
    # parser.add_argument("--word_embeddings",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The path of word embeddings")

    # Location of the word frequency file
    # parser.add_argument("--word_frequency",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The path of word frequency.")

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

    # Number of training epochs
    parser.add_argument("--num_selections",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()

    ### Location of execution: ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Opening/ Loading Files ###
    # Cache Location:
    cache_dir = args.cache_dir

    # Evaluation file:
    evaluation_file_name = args.eval_dir.split('/')[-1][:-4]

    # Output File:
    output_sr_file = open(args.output_SR_file, "a+")

    # Loading the stemmer
    ps = PorterStemmer()

    # Number of candidates for substitution
    num_selection = args.num_selections

    ### Start initialization of the model ###

    ### Loading in the Model & Tokenizer ###

    print("Loading the model and tokenizer")

    if args.language == "nl":
        tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
        model = AutoModelForMaskedLM.from_pretrained("GroNLP/bert-base-dutch-cased")

        embedding_path = "../models/wikipedia-320.txt"
        word_count_path = "../datasets/dutch_frequencies.txt"

    if args.language == "eng":

        # used_model = "bert-base-uncased"
        # used_model = "bert-base-cased"
        # used_model = "bert-large-uncased-whole-word-masking"
        # used_model = "../models/combined_FT_2"
        used_model = "../models/finetuned_on_gpu"
        # used_model = "D:/Thesis/combined_FT_2"
        # used_model = "D:/Thesis/finetuned_on_gpu"



        if used_model.startswith("bert"):
            tokenizer = AutoTokenizer.from_pretrained(used_model)
            model = AutoModelForMaskedLM.from_pretrained(used_model)

        else:
            model = BertForPreTraining.from_pretrained(used_model)
            tokenizer = BertTokenizer.from_pretrained(used_model)


        embedding_path = "../models/crawl-300d-2M-subword.vec"
        word_count_path = "../datasets/frequency_merge_wiki_child.txt"

    word_count = getWordCount(word_count_path)  # This is a dictionary with the shape word: frequency

    model.to(device)

    ### Loading in Embeddings ###
    print("Loading embeddings ...")
    embedding_vocab, embedding_vectors = getWordmap(embedding_path) # the vocabulary of the embedding model in a list, and the corresponding emb values in another
    print("Done loading embeddings")

    ### Toward Generating the Substitutions:###
    candidates_list = []
    substitution_words = []
    bre_i = 0
    window_context = 11

    ### Retrieve the sentences, complex words and annotated labels ###
    if evaluation_file_name == 'lex.mturk' or evaluation_file_name == 'small_example_dutch':  # Specifically for these files (with header etc)
        eval_sents, complex_words, annotated_subs = read_eval_dataset_lexmturk(args.eval_dir)
    else:
        eval_sents, complex_words, annotated_subs = read_eval_index_dataset(args.eval_dir)

    ### Running the evaluation ###
    logger.info("***** Running evaluation *****")

    # Pytorch model in evaluation mode:
    model.eval()

    eval_size = len(eval_sents)

    ### Loop over the evaluation sentences: ###
    now = datetime.now()
    now_string = now.strftime("%d-%m-%Y-%H:%M")
    used_model = used_model.replace("../models/", "")

    with open("../results/"+used_model+now_string+".txt", "w", encoding="UTF-8") as outfile:

        for i in range(eval_size):
            logger.info("***** next sentence *****")

            print( f"_______________________\nSENTENCE{i} \n")
            sentence = eval_sents[i]
            print(sentence)

            outfile.write(str(sentence)+"\t")

            # Making a mapping between BERT's subword tokenized sent and nltk tokenized sent
            bert_sent, nltk_sent, bert_token_positions = convert_sentence_to_token(eval_sents[i], args.max_seq_length, tokenizer)

            assert len(nltk_sent) == len(bert_token_positions)
            complex_word = complex_words[i]
            print("complex word", complex_word, "\n")

            mask_index = nltk_sent.index(complex_words[i])  # the location of the complex word:
            outfile.write(nltk_sent[mask_index])

            mask_context = extract_context(nltk_sent, mask_index, window_context)  # the words surrounding it

            len_tokens = len(bert_sent)
            bert_mask_position = bert_token_positions[mask_index]  # BERT index of mask

            if isinstance(bert_mask_position, list):  # If the mask is at a sub-word-tokenized token
                # This is an instance of the feature class
                feature = convert_whole_word_to_feature(bert_sent, bert_mask_position, args.max_seq_length, tokenizer)
            else:
                feature = convert_token_to_feature(bert_sent, bert_mask_position, args.max_seq_length, tokenizer)
            tokens_tensor = torch.tensor([feature.input_ids])

            # Something with masking/ attention
            token_type_ids = torch.tensor([feature.input_type_ids])

            # Something with masking
            attention_mask = torch.tensor([feature.input_mask])

            # And on their way to the CUDA/ CPU
            tokens_tensor = tokens_tensor.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            ### Make predictions ###
            with torch.no_grad():
                output = model(tokens_tensor, attention_mask=attention_mask, token_type_ids=token_type_ids)
                prediction_scores = output[0]

            if isinstance(bert_mask_position, list):
                predicted_top = prediction_scores[0, bert_mask_position[0]].topk(num_selection*2)
            else:
                predicted_top = prediction_scores[0, bert_mask_position].topk(num_selection*2)

            pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1])
            pre_probs = F.softmax(predicted_top.values, dim=-1)

            # A hard cut on the selection, leaving maximum num_selection candidates
            candidate_words = substitution_generation(complex_words[i], pre_tokens, pre_probs,
                                                      ps,
                                                      num_selection)

            for cand in candidate_words:
                outfile.write(cand + "\t")

            # print("candidate words", candidate_words)

            candidates_list.append(candidate_words)

            highest_preds = substitution_ranking(complex_words[i], mask_context, candidate_words, embedding_vocab,
                                                 embedding_vectors, word_count, tokenizer, model, annotated_subs[i])
            for word in highest_preds:
                outfile.write("\t"+word)
            outfile.write("\n")

            predicted_word = highest_preds[0]
            substitution_words.append(predicted_word)
            print("predicted word: ", predicted_word)


    potential, precision, recall, f_score = evaluation_SS_scores(candidates_list, annotated_subs)
    print("The score of evaluation for substitution selection")
    output_sr_file.write(str(used_model))
    output_sr_file.write(str(args.num_selections))
    output_sr_file.write('\t')
    output_sr_file.write(str(potential))
    output_sr_file.write('\t')
    output_sr_file.write(str(precision))
    output_sr_file.write('\t')
    output_sr_file.write(str(recall))
    output_sr_file.write('\t')
    output_sr_file.write(str(f_score))
    output_sr_file.write('\t')
    print(potential, precision, recall, f_score)

    precision, accuracy, changed_proportion = evaluation_pipeline_scores(substitution_words, complex_words,
                                                                         annotated_subs)
    print("The score of evaluation for full LS pipeline")
    print(precision, accuracy, changed_proportion)
    output_sr_file.write(str(precision))
    output_sr_file.write('\t')
    output_sr_file.write(str(accuracy))
    output_sr_file.write('\t')
    output_sr_file.write(str(changed_proportion))
    output_sr_file.write('\n')

    output_sr_file.close()


if __name__ == "__main__":
    main()
