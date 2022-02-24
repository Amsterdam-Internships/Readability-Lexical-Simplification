export BERT_DIR=pytorch_pretrained_BERT/
export Result_DIR=results/

python.exe LSBert.py \
  --do_eval \
  --do_lower_case \
  --num_selections 10 \
  --eval_dir datasets/BenchLSshort.txt \
  --bert_model bert-pre-trained-readability \
  --max_seq_length 250 \
  --word_embeddings models/crawl-300d-2M-subword.vec\
  --word_frequency datasets/frequency_merge_wiki_child.txt \
  --output_SR_file $Result_DIR/aaa\
  --no_cuda
