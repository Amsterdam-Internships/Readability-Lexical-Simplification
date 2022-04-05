import logging
import argparse
import random
import spacy

from nltk import PorterStemmer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import *
from transformers import BertForPreTraining, BertTokenizer
from nltk.stem import WordNetLemmatizer

# python -m spacy download nl_core_news_sm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch.nn.functional as F

nlp = spacy.load("nl_core_news_sm")

# https://stackoverflow.com/questions/67500193/cleaning-lemmatizing-dutch-dataset-using-spacy
def lemmatizer(texts):
    texts = [text.replace("\n", "").strip() for text in texts]
    docs = nlp.pipe(texts)
    cleaned_lemmas = [[t.lemma_ for t in doc] for doc in docs]

    return cleaned_lemmas

def main():
    """Parsing the input, and running the first functions"""
    global complex_words
    parser = argparse.ArgumentParser()

    # Directory of evaluation data (BenchLS/ Lexmturk/ NNSeval)
    # #ToDo Fill in the exact datasets
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data directory.")


    # Location for caching

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
    device = "cpu"

    ### Opening/ Loading Files ###

    # Evaluation file:
    evaluation_file_name = args.eval_dir.split('/')[-1][:-4]

    # Loading the stemmer
    ps = PorterStemmer()

    # Number of candidates for substitution
    num_selection = args.num_selections

    ### Start initialization of the model ###

    ### Loading in the Model & Tokenizer ###
    print("Loading the model and tokenizer")

    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    model = AutoModelForMaskedLM.from_pretrained("GroNLP/bert-base-dutch-cased")

    embedding_path = "../models/wikipedia-320.txt"
    word_count_path = "../datasets/dutch_frequencies.txt"

    word_count = getWordCount(word_count_path)

    model.to(device)

    ### Loading in Embeddings ###
    print("Loading embeddings ...")
    # embedding_path = args.word_embeddings
    embedding_vocab, embedding_vectors = getWordmap(embedding_path) # the vocabulary of the embedding model in a list, and the corresponding emb values in another
    print("Done loading embeddings")

    ### Toward Generating the Substitutions:###
    candidates_list = []
    substitution_words = []
    bre_i = 0
    window_context = 11

    ### Retrieve the sentences, complex words and annotated labels ###
    if evaluation_file_name == 'lex.mturk' or evaluation_file_name == 'dutch_sentences':  # Specifically for these files (with header etc)
        eval_sents, complex_words, annotated_subs = read_eval_dataset_lexmturk(args.eval_dir)
    else:
        eval_sents, complex_words, annotated_subs = read_eval_index_dataset(args.eval_dir)

    ### Running the evaluation ###
    logger.info("***** Running evaluation *****")

    # Pytorch model in evaluation mode:
    model.eval()

    eval_size = len(eval_sents)

    with open("../results/dutch_simplifications_with_final.txt", "w",encoding="UTF-8") as outfile:

        ### Loop over the evaluation sentences: ###
        for i in range(200):
            print( f"_______________________\nSENTENCE{i} \n")
            sentence = eval_sents[i]
            print(sentence)

            if len(sentence)>220:
                print("sentence too long")
                continue

            outfile.write(str(sentence)+"\t")

            # Making a mapping between BERT's subword tokenized sent and nltk tokenized sent
            bert_sent, nltk_sent, bert_token_positions = convert_sentence_to_token(sentence, args.max_seq_length, tokenizer)

            assert len(nltk_sent) == len(bert_token_positions)

            complex_word = complex_words[i]
            print("complex word", complex_word, "\n")

            if complex_word in nltk_sent:
                mask_index = nltk_sent.index(complex_words[i])
            else:
                nltk_lemmas = lemmatizer([sentence])[0]
                print("lemmas", nltk_lemmas)
                print(complex_word)
                complex_lemma = lemmatizer([complex_word])[0][0]
                print("clemma", complex_lemma)

                if complex_lemma in nltk_lemmas:
                    mask_index = nltk_lemmas.index(complex_lemma)  # the location of the complex word:
                else:
                    print("word not in sentence")
                    continue

            outfile.write(nltk_sent[mask_index]+"\t")

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

            print(list(zip(pre_tokens, pre_probs)))

            # A hard cut on the selection, leaving maximum num_selection candidates
            candidate_words = substitution_generation(complex_words[i], pre_tokens, pre_probs, ps, num_selection)
            print("candidate words", candidate_words)


            for cand in candidate_words:
                outfile.write(cand + "\t")

            # candidates_list.append(candidate_words)

            predicted_word = substitution_ranking(complex_words[i], mask_context, candidate_words, embedding_vocab,
                                                  embedding_vectors, word_count, tokenizer, model, annotated_subs[i])
            substitution_words.append(predicted_word)

            outfile.write(predicted_word+"\n")

            print("predicted word: ", predicted_word)


if __name__ == "__main__":
    main()
