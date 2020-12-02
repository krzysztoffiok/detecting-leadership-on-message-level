start=`date +%s`

# define models to be trained: transformer models and other models
declare -a transformer_list=('albert-base-v2' 'roberta-large' 'bert-large-uncased' 'xlm-roberta-large')

## models that will not be fine-tuned, instead LSTM over embeddings will be deployed. To change LSTM parameters
## change model_train.py
declare -a other_model_list=('fasttext' 'roberta_lstm')
#
## define pretrained SentenceTransformer models to be tested, or other models that will not be trained
## instead token embeddings provided by them will be pooled (averaged)
declare -a pretrained_model_list=('roberta')

declare -a baseline_model_list=('LIWC' 'SEANCE' 'TF')

# declare dependent variable and number of folds
dependent_variable="leadership"
input_file="5k_final_leadership_values.xlsx"
folds=5 ### CHANGE FOLDS
let "embed_folds=$folds - 1"

# split the input data
python3 input_data_splitter.py --folds=$folds --filepath=$input_file --binarize

# train selected models
for i in "${transformer_list[@]}"; do python3 ./model_train.py --filepath=$input_file \
 --dependent_variable=$dependent_variable --folds=$folds \
 --test_run="$i" --fine_tune; done

for i in "${other_model_list[@]}"; do python3 ./model_train.py --filepath=$input_file \
 --dependent_variable=$dependent_variable --folds=$folds \
 --test_run="$i" --epochs=1000; done ### CHANGE EPOCHS

# embed data with trained models
for i in "${transformer_list[@]}"; do for j in $(seq 0 $embed_folds); do python3 embed_data.py \
--dependent_variable=$dependent_variable --fold=$j --test_run="$i"  --filepath=$input_file --binarize; done; done
for i in "${other_model_list[@]}"; do for j in $(seq 0 $embed_folds); do python3 embed_data.py \
 --dependent_variable=$dependent_variable --fold=$j --test_run="$i"  --filepath=$input_file --binarize; done; done

# embed data with pre-trained models
for i in "${pretrained_model_list[@]}"; do python3 embed_data.py \
 --dependent_variable=$dependent_variable --test_run="$i" --pool=True  --filepath=$input_file --binarize; done

# carry out the final ML predictions
for i in "${transformer_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
 --dependent_variable=$dependent_variable --folds=$folds --filepath=$input_file --estimators=250 --binarize;done
for i in "${other_model_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
 --dependent_variable=$dependent_variable --folds=$folds --filepath=$input_file --estimators=250 --binarize;done
for i in "${pretrained_model_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
 --dependent_variable=$dependent_variable --folds=$folds --pooled --filepath=$input_file --estimators=250 --binarize;done
for i in "${baseline_model_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
--dependent_variable=$dependent_variable --folds=$folds --filepath=$input_file --estimators=250 --binarize;done

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./data/partial_results/binary_experiment_total_time.txt
echo $runtime > $destdir
