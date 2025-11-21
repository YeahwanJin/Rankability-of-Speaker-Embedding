#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime

import utils
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.stats import spearmanr 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

warnings.simplefilter("ignore")

from torch.utils.tensorboard import SummaryWriter


## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',         type=int,   default=200,    help='Input audio length  to the network for training')
parser.add_argument('--eval_frames',        type=int,   default=400,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',         type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk',    type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread',  type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',            type=bool,  default=True,   help='Augment input')
parser.add_argument('--seed',               type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,       help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,     help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",      help='Loss function')
parser.add_argument('--adv_alpha',      type=float, default=0.5,     help='Alpha value for disentanglement')
parser.add_argument('--disc_iteration', type=int,   default=5,       help='Iterations of environment phase')

## Optimizer
parser.add_argument('--optimizer',          type=str,   default="adam",   help='sgd or adam')
parser.add_argument('--scheduler',          type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',                 type=float, default=0.001,    help='Learning rate')
parser.add_argument("--lr_decay",           type=float, default=0.97,     help='Learning rate decay every [lr_decay_interval] epochs')
parser.add_argument("--lr_decay_interval",  type=int,   default=1,        help='Learning rate decay interval')
parser.add_argument("--lr_decay_start",     type=int,   default=0,        help='Learning rate decay start epoch')
parser.add_argument('--weight_decay',       type=float, default=5e-5,     help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.3,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=3,      help='Number of utterances per speaker per batch. In our work, we should fix with the value 3')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')
parser.add_argument('--log_dir',        type=str,   default="logs", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="datasets/manifests/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="datasets/manifests/veri_test_cleaned.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="datasets/voxceleb2", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="datasets/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="datasets/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="datasets/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=64,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--C',              type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--spk_dims',       type=int,   default=256,    help='Embedding size of speaker feature')
parser.add_argument('--env_dims',       type=int,   default=256,    help='Embedding size of environment feature')
parser.add_argument('--disc_dims',      type=int,   default=256,    help='input  dims of Environment discriminator')
parser.add_argument('--disc_out_dims',  type=int,   default=128,    help='Output dims of Environment discriminator')
parser.add_argument('--dropout',        type=float, default=0.3,    help='Dropout ration of noise code')
parser.add_argument('--sinc_stride',    type=int,   default=10,     help='Stride size of the first analytic filterbank layer of RawNet3')


## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--run_age_exp',    dest='run_age_exp', action='store_true', help='Run age ranking experiment')
parser.add_argument('--run_age_exp_linear', dest='run_age_exp_linear', action='store_true', help='Run age ranking with Linear Regression (SRCC, PCC, MAE)')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--train_age_predictor', dest='train_age_predictor', action='store_true', help='Train a Linear Regression model to PREDICT age from embeddings')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

# trainSpeakerNet.py (main_worker 함수 위)

# 헬퍼 함수: 데이터셋에서 임베딩과 나이를 추출 (DE 모델 버전)
def extract_embeddings_with_age(model, data_loader, gpu):
    model.eval() # 모델을 평가 모드로 설정
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        print("Extracting speaker embeddings...")
        for i, (feats, ages) in enumerate(data_loader):
            feats = feats.cuda(gpu) # 데이터를 GPU로
            
            # ⭐️ 중요: DE 버전의 SpeakerNet.forward는 (spk_code, env_code)를 반환
            # 우리는 스피커 임베딩(spk_code)만 필요합니다.
            spk_code, _ = model(feats, label=None) 
            
            all_embeddings.append(spk_code.cpu().numpy())
            all_labels.append(ages.cpu().numpy())
            
            if i % 50 == 0:
                print(f"Processed {i*data_loader.batch_size} files...", end='\r')

    print("\nExtraction complete.")
    all_embeddings = numpy.concatenate(all_embeddings, axis=0)
    all_labels = numpy.concatenate(all_labels, axis=0)
    
    return all_embeddings, all_labels

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    writer = None 

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100]

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")
        if not args.eval:
            log_dir = os.path.join(args.log_dir, os.path.basename(args.save_path))
            if log_dir[-1] == '/':
                log_dir = log_dir[:-1]

            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))

    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()


    ## Evaluation code - must run on single GPU
    if args.eval == True:
        print('Test list',args.test_list)
        
        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))
            
        return
    
    if args.run_age_exp == True:
        
        if args.gpu != 0:
            return

        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Running Age Ranking Experiment...")

        ## 1. 모델 로드 확인
        if args.initial_model == "" and not modelfiles:
            print("Error: No model found. Use --initial_model to specify a trained model.")
            return
        
        current_model_path = args.initial_model or modelfiles[-1]
        print(f"Using model: {current_model_path}")
        
        ## 2. Train-set(young/old) 임베딩 추출
        print("Loading train list:", args.train_list)
        age_train_dataset = AgeDatasetLoader(args.train_list, **vars(args))
        age_train_loader = torch.utils.data.DataLoader(
            age_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            pin_memory=False,
            drop_last=False
        )
        
        train_embeds, train_ages = extract_embeddings_with_age(s, age_train_loader, args.gpu)
        
        ## 3. embed_young, embed_old 계산
        young_embeds = train_embeds[train_ages < 20]
        old_embeds = train_embeds[train_ages > 77]
        
        if len(young_embeds) == 0 or len(old_embeds) == 0:
            print("Error: Not enough data for young (<30) or old (>60) speakers.")
            print(f"Found {len(young_embeds)} young samples, {len(old_embeds)} old samples.")
            return

        embed_young = numpy.mean(young_embeds, axis=0)
        embed_old = numpy.mean(old_embeds, axis=0)
        
        print(f"Calculated embed_young from {len(young_embeds)} samples.")
        print(f"Calculated embed_old from {len(old_embeds)} samples.")

        ## 4. rank_axis 계산 (L2-normalize)
        rank_axis = embed_old - embed_young
        rank_axis = rank_axis / numpy.linalg.norm(rank_axis)
        
        ## 5. Test-set 임베딩 추출
        print("Loading test list:", args.test_list)
        age_test_dataset = AgeDatasetLoader(args.test_list, **vars(args))
        age_test_loader = torch.utils.data.DataLoader(
            age_test_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            pin_memory=False,
            drop_last=False
        )
        
        test_embeds, test_ages = extract_embeddings_with_age(s, age_test_loader, args.gpu)

        ## 6. Rank Score 계산 (Dot product)
        print("Calculating rank scores...")
        rank_scores = numpy.dot(test_embeds, rank_axis)

        #거리 차이 분석 로직
        df = pd.DataFrame({
            'age': test_ages.flatten(),
            'score': rank_scores.flatten() 
        })
        df = df.sort_values(by='age').reset_index(drop=True)
        
        ## 7. SRCC (Spearman's Rank Correlation Coefficient) 계산
        correlation, p_value = spearmanr(rank_scores, test_ages)
        
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "--- Age Ranking Experiment Results ---")
        print(f"Spearman's Rank Correlation Coefficient (SRCC): {correlation:.6f}")
        print(f"P-value: {p_value:.6f}")

        scores_file_path = os.path.join(args.result_save_path, "age_rank_scores.txt")
        print(f"Saving individual scores to {scores_file_path}")
        
        # pandas DataFrame을 탭(\t)으로 구분된 .txt 파일로 저장합니다.
        df.to_csv(
            scores_file_path, 
            sep='\t',         # 탭(tab)으로 열 구분
            index=False,      # 0, 1, 2... 같은 인덱스 번호는 저장 안 함
            float_format='%.6f' # 소수점 6자리까지만 저장
        )
        
        # 결과 저장
        result_file_path = os.path.join(args.result_save_path, "age_rank_srcc.txt")
        print(f"Saving results to {result_file_path}")
        with open(result_file_path, "w") as f:
            f.write(f"SRCC: {correlation}\n")
            f.write(f"P-value: {p_value}\n")
            f.write(f"Train List: {args.train_list}\n")
            f.write(f"Test List: {args.test_list}\n")
            f.write(f"Model: {current_model_path}\n")
            f.write(f"Young samples: {len(young_embeds)}, Old samples: {len(old_embeds)}\n")

        return # 실험 완료 후 종료
    
    if args.run_age_exp_linear == True:
        
        # 이 실험은 단일 GPU(gpu=0)에서만 실행하는 것이 좋습니다.
        if args.gpu != 0:
            return

        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Running Age Ranking + Linear Regression Experiment...")

        ## 1. 모델 로드 확인
        if args.initial_model == "" and not modelfiles:
            print("Error: No model found. Use --initial_model to specify a trained model.")
            return
        
        current_model_path = args.initial_model or modelfiles[-1]
        print(f"Using model: {current_model_path}")
        
        ## 2. Train-set(young/old) 임베딩 추출
        print("Loading train list:", args.train_list)
        age_train_dataset = AgeDatasetLoader(args.train_list, **vars(args))
        age_train_loader = torch.utils.data.DataLoader(
            age_train_dataset, batch_size=args.batch_size, 
            num_workers=args.nDataLoaderThread, pin_memory=False, drop_last=False
        )
        train_embeds, train_ages = extract_embeddings_with_age(s, age_train_loader, args.gpu)
        
        ## 3. embed_young, embed_old 계산 (사용자 설정: <20, >77)
        young_embeds = train_embeds[train_ages < 24]
        old_embeds = train_embeds[train_ages > 68]
        
        if len(young_embeds) == 0 or len(old_embeds) == 0:
            print(f"Error: Not enough data for young (<20) or old (>77) speakers.")
            print(f"Found {len(young_embeds)} young samples, {len(old_embeds)} old samples.")
            return

        embed_young = numpy.mean(young_embeds, axis=0)
        embed_old = numpy.mean(old_embeds, axis=0)
        print(f"Calculated embed_young from {len(young_embeds)} samples.")
        print(f"Calculated embed_old from {len(old_embeds)} samples.")

        ## 4. rank_axis 계산 (L2-normalize)
        rank_axis = embed_old - embed_young
        rank_axis = rank_axis / numpy.linalg.norm(rank_axis)
        
        ## 5. Test-set 임베딩 추출
        print("Loading test list:", args.test_list)
        age_test_dataset = AgeDatasetLoader(args.test_list, **vars(args))
        age_test_loader = torch.utils.data.DataLoader(
            age_test_dataset, batch_size=args.batch_size, 
            num_workers=args.nDataLoaderThread, pin_memory=False, drop_last=False
        )
        test_embeds, test_ages = extract_embeddings_with_age(s, age_test_loader, args.gpu)

        ## 6. Rank Score 계산 (Dot product)
        print("Calculating rank scores...")
        rank_scores = numpy.dot(test_embeds, rank_axis)

        # --- 7. 선형 회귀 및 상관관계 분석 ---
        print("Running Linear Regression & Correlation Analysis...")

        # 7-A. 선형 회귀 (Linear Regression)
        m, b = numpy.polyfit(rank_scores, test_ages, 1)
        predicted_ages = (m * rank_scores) + b
        
        # 7-B. MAE (Mean Absolute Error) & RMSE
        mae = numpy.mean(numpy.abs(predicted_ages - test_ages))
        rmse = numpy.sqrt(numpy.mean((predicted_ages - test_ages)**2))
        
        # 7-C. PCC (Pearson Correlation Coefficient)
        pcc, p_value_pcc = spearmanr(rank_scores, test_ages)

        # 7-D. SRCC (Spearman's Rank Correlation)
        srcc, p_value_srcc = spearmanr(rank_scores, test_ages)

        # 7-E. 결과 출력
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "--- Age Ranking Experiment Results ---")
        print("\n--- Correlation Metrics ---")
        print(f"Spearman's Rank (SRCC): {srcc:.6f} (p={p_value_srcc:.6f})")
        print(f"Pearson Linear (PCC): {pcc:.6f} (p={p_value_pcc:.6f})")
        print("\n--- Linear Regression Prediction ---")
        print(f"Fitted Line: predicted_age = {m:.4f} * score + {b:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f} years")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} years")

        # --- 8. 결과 저장 ---
        
        # 8-A. 개별 점수 저장 (predicted_age 포함)
        # 파일 이름을 'linear'로 구분
        scores_file_path = os.path.join(args.result_save_path, "age_rank_scores_linear.txt") 
        print(f"\nSaving individual scores to {scores_file_path}")
        
        df = pd.DataFrame({
            'age': test_ages.flatten(),
            'score': rank_scores.flatten(),
            'predicted_age': predicted_ages.flatten()
        })
        df = df.sort_values(by='age').reset_index(drop=True)
        df.to_csv(scores_file_path, sep='\t', index=False, float_format='%.6f')

        # 8-B. 요약 결과 저장 (모든 지표 포함)
        # 파일 이름을 'linear'로 구분
        result_file_path = os.path.join(args.result_save_path, "age_rank_summary_linear.txt") 
        print(f"Saving summary results to {result_file_path}")
        with open(result_file_path, "w") as f:
            f.write(f"SRCC: {srcc}\n")
            f.write(f"SRCC_p_value: {p_value_srcc}\n")
            f.write(f"PCC: {pcc}\n")
            f.write(f"PCC_p_value: {p_value_pcc}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"LinReg_Slope: {m}\n")
            f.write(f"LinReg_Intercept: {b}\n")
            f.write(f"Train List: {args.train_list}\n")
            f.write(f"Test List: {args.test_list}\n")
            f.write(f"Model: {current_model_path}\n")
            f.write(f"Young samples: {len(young_embeds)}, Old samples: {len(old_embeds)}\n")

        return
    if args.train_age_predictor == True:
        
        # 이 실험은 단일 GPU(gpu=0)에서만 실행하는 것이 좋습니다.
        if args.gpu != 0:
            return

        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Training Age PREDICTION Model (Linear Regression)...")

        ## 1. 모델 로드 확인
        if args.initial_model == "" and not modelfiles:
            print("Error: No model found. Use --initial_model to specify a trained model.")
            return
        
        current_model_path = args.initial_model or modelfiles[-1]
        print(f"Using model: {current_model_path}")

        ## 2. Train-set에서 임베딩(X_train)과 나이(y_train) 추출
        print("Loading train list for training predictor:", args.train_list)
        age_train_dataset = AgeDatasetLoader(args.train_list, **vars(args))
        age_train_loader = torch.utils.data.DataLoader(
            age_train_dataset, batch_size=args.batch_size, 
            num_workers=args.nDataLoaderThread, pin_memory=False, drop_last=False
        )
        # X_train: (N_train, 512), y_train: (N_train,)
        X_train, y_train = extract_embeddings_with_age(s, age_train_loader, args.gpu)

        ## 3. Test-set에서 임베딩(X_test)과 나이(y_test) 추출
        print("Loading test list for evaluating predictor:", args.test_list)
        age_test_dataset = AgeDatasetLoader(args.test_list, **vars(args))
        age_test_loader = torch.utils.data.DataLoader(
            age_test_dataset, batch_size=args.batch_size, 
            num_workers=args.nDataLoaderThread, pin_memory=False, drop_last=False
        )
        # X_test: (N_test, 512), y_test: (N_test,)
        X_test, y_test = extract_embeddings_with_age(s, age_test_loader, args.gpu)

        if len(X_train) == 0 or len(X_test) == 0:
            print("Error: Train or Test data is empty.")
            return
            
        print(f"Loaded {len(X_train)} train samples and {len(X_test)} test samples.")

        ## 4. 데이터 스케일링 (중요)
        # 선형 회귀는 피처 스케일링을 하면 더 안정적으로 수렴합니다.
        print("Standardizing features (embeddings)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # ⭐️ Train 스케일러로 Test 변환

        ## 5. 선형 회귀 모델 훈련
        print("Training LinearRegression model...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        ## 6. 테스트셋으로 나이 예측
        print("Evaluating model on test set...")
        y_pred = model.predict(X_test_scaled)
        
        ## 7. 성능 평가
        # 7-A. MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 7-B. RMSE (Root Mean Squared Error)
        rmse = numpy.sqrt(mean_squared_error(y_test, y_pred))
        
        # 7-C. PCC (Pearson Correlation) - 예측값과 실제값의 상관관계
        pcc, p_value_pcc = spearmanr(y_test, y_pred)
        
        # 7-D. SRCC (Spearman Correlation) - 예측값과 실제값의 순위 상관관계
        srcc, p_value_srcc = spearmanr(y_test, y_pred)

        ## 8. 결과 출력
        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "--- Age PREDICTION Model Results ---")
        print("\n--- Prediction Error (Test Set) ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f} years")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} years")

        print("\n--- Prediction Correlation (Test Set) ---")
        print(f"Spearman's Rank (SRCC): {srcc:.6f} (p={p_value_srcc:.6f})")
        print(f"Pearson Linear (PCC): {pcc:.6f} (p={p_value_pcc:.6f})")

        ## 9. 결과 저장
        # 9-A. 개별 예측 결과 저장
        scores_file_path = os.path.join(args.result_save_path, "age_prediction_scores.txt")
        print(f"\nSaving individual predictions to {scores_file_path}")
        
        df = pd.DataFrame({
            'real_age': y_test.flatten(),
            'predicted_age': y_pred.flatten()
        })
        df = df.sort_values(by='real_age').reset_index(drop=True)
        df.to_csv(scores_file_path, sep='\t', index=False, float_format='%.6f')

        # 9-B. 요약 결과 저장
        result_file_path = os.path.join(args.result_save_path, "age_prediction_summary.txt") 
        print(f"Saving summary results to {result_file_path}")
        with open(result_file_path, "w") as f:
            f.write(f"Task: Age Prediction from Embeddings\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"PCC: {pcc}\n")
            f.write(f"PCC_p_value: {p_value_pcc}\n")
            f.write(f"SRCC: {srcc}\n")
            f.write(f"SRCC_p_value: {p_value_srcc}\n")
            f.write(f"Train List: {args.train_list}\n")
            f.write(f"Test List: {args.test_list}\n")
            f.write(f"Model: {current_model_path}\n")
            f.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

        print("Data Augmentaiton:",   args.augment)
        print("Embed   dims:",        args.nOut)
        print("Latent  dims:",        args.spk_dims + args.env_dims)
        print("Speaker dims:",        args.spk_dims)
        print("ENV     dims:",        args.env_dims)
        print("Adv_loss alpha:",      args.adv_alpha)
        print("Discriminator iteration:",  args.disc_iteration)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        res = trainer.train_network(train_loader, verbose=(args.gpu == 0), adv_alpha=args.adv_alpha, num_D_steps=args.disc_iteration, lr_schedule=it+1 > args.lr_decay_start)
        loss, loss_spk, loss_neg, loss_recons, traineer = res

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, NEGL {:.3f}, RECONSL {:.3f}, LR {:f} \n".format(
                            it, traineer, loss, loss_neg, loss_recons, max(clr)))

        if it % args.test_interval == 0:
            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:
                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()
                
        # If you need to use tensorboard, uncomment this block
        """if args.gpu == 0: 
            loss_scalars = {
                'loss/total_loss'     : loss,
                'loss/spk_loss'       : loss_spk,
                'loss/neg_loss'       : loss_neg,
                'loss/recon_loss'     : loss_recons
            }
            
            EER_scalars = {
                'EER/train' : traineer,
                'EER/test'  : result[1]
            }
            
            
            utils.summarize(writer, it, loss_scalars, EER_scalars, multi_scalars={})"""

    if args.gpu == 0:
        scorefile.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()