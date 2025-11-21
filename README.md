Test rankability command

1. Resnet: check data path and use below command line(to test linear regression score: use run_age_exp_linear instead of run_age_exp)

python ./trainSpeakerNet.py \
--run_age_exp \ 
--initial_model pretrained/baseline_v2_smproto.model \
--train_list datasets/manifests/vox2_age_train_set_sep.txt \
--test_list datasets/manifests/vox2_age_test_set_sep.txt \
--train_path /mnt/datasets/voxcelebs/voxceleb2/dev/wav \
--test_path /mnt/datasets/voxcelebs/voxceleb2/test/wav \
--batch_size 100 \
--save_path exps/exp1_age_rank \
--model ResNetSE34V2 \
--log_input True \
--encoder_type ASP \
--n_mels 64 \
--trainfunc softmaxproto

2. ECAPA_TDNN(to test linear regression score: use run_age_exp_linear instead of train_age_predictor)

python ./trainSpeakerNet.py \
--train_age_predictor \
--initial_model pretrained/baseline_ecapatdnn_de.model \
--train_list datasets/manifests/vox2_age_train_set_sep.txt \
--test_list datasets/manifests/vox2_age_test_set_sep.txt \
--train_path /mnt/datasets/voxcelebs/voxceleb2/dev/wav \
--test_path /mnt/datasets/voxcelebs/voxceleb2/test/wav \
--batch_size 100 \
--save_path exps/exp1_age_rank \
--model ECAPA_TDNN \
--log_input True \
--trainfunc softmaxproto \
--n_mels 80 \
--nOut 256 \
--C 1024 \
--encoder_type ECA \
--spk_dims 256 \
--env_dims 256 \
--spk_dims 256 \
--env_dims 256

3. RedimNet

python ./trainSpeakerNet.py \
--run_age_exp \
--initial_model redimnet_base \
--train_list datasets/manifests/vox2_age_train_set_sep.txt \
--test_list datasets/manifests/vox2_age_test_set_sep.txt \
--train_path /mnt/datasets/voxcelebs/voxceleb2/dev/wav \
--test_path /mnt/datasets/voxcelebs/voxceleb2/test/wav \
--batch_size 32 \
--save_path exps/exp1_redimnet_age_rank \
--model ReDimNet \
--log_input False \
--trainfunc softmaxproto \
--n_mels 72 \
--nOut 192 \
--C 16
