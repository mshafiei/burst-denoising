python2 kpn_train.py 
# python2 shuffle_test.py --dataset_dir ./data/challenge2018
exp_params="--dataset_dir ./data/challenge2018 \
--logdir ./logger/retrain \
--train_log_dir ./logger/retrain \
--expname retrain"

name=msh-burst-retrain7
scriptFn="kpn_train.py $exp_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"