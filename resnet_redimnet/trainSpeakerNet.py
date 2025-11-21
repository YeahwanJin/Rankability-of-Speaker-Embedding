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
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.stats import spearmanr
import pandas as pd # ⭐️ 이 줄을 추가하세요
from scipy.stats import pearsonr # ⭐️ 이 줄을 추가하세요
from huggingface_hub import hf_hub_download
#to match with titan cuda version
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="datasets/manifests/vox2_train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="datasets/manifests/vox1-my_list1.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="/mnt/datasets/voxcelebs/voxceleb2/dev/wav", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="/mnt/datasets/voxcelebs/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="/mnt/datasets/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/mnt/datasets/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

#for age ranking
parser.add_argument('--task',           type=str,   default='train_speaker', help='Task to run: train_speaker or train_age')
parser.add_argument('--run_age_exp',    dest='run_age_exp', action='store_true', help='Run age ranking experiment')
parser.add_argument('--run_age_exp_linear', dest='run_age_exp_linear', action='store_true', help='Run age ranking with Linear Regression (SRCC, PCC, MAE)')
#for ecapa tdnn
parser.add_argument('--C',              type=int,   default=1024,   help='Channel size for ECAPA_TDNN model')
parser.add_argument('--spk_dims',       type=int,   default=256,    help='Embedding size of speaker feature')
parser.add_argument('--env_dims',       type=int,   default=256,    help='Embedding size of environment feature')
parser.add_argument('--disc_dims',      type=int,   default=256,    help='input  dims of Environment discriminator')
parser.add_argument('--disc_out_dims',  type=int,   default=128,    help='Output dims of Environment discriminator')
# ===============================
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

#function for extracting embedding for age data
def extract_embeddings_with_age(model, data_loader, gpu):
    model.eval() # 모델을 평가 모드로 설정
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        print("Extracting embeddings...")
        for i, (feats, ages) in enumerate(data_loader):
            feats = feats.cuda(gpu) # 데이터를 GPU로
            
            # model(feats, label=None)을 호출하면
            # SpeakerNet.forward(feats, label=None)가 호출되어
            # 임베딩이 반환됩니다.
            embeds = model(feats, label=None) 
                
            all_embeddings.append(embeds.cpu().numpy())
            all_labels.append(ages.cpu().numpy())
            
            if i % 50 == 0:
                print(f"Processed {i*data_loader.batch_size} files...", end='\r')

    print("\nExtraction complete.")
    all_embeddings = numpy.concatenate(all_embeddings, axis=0)
    all_labels = numpy.concatenate(all_labels, axis=0)
    
    return all_embeddings, all_labels


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

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

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model == "redimnet_base"):
        print("Model loaded!")

    elif(args.initial_model != ""):
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

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        
        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))

        return
    
    ## =============================================
    ## ===== AGE RANKING EXPERIMENT (신규 추가) =====
    ## =============================================
    if args.run_age_exp == True:
        
        # 이 실험은 분산 학습(DDP) 모드에서는 권장되지 않으며,
        # 단일 GPU(gpu=0)에서만 실행하는 것이 좋습니다.
        if args.gpu != 0:
            return

        print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Running Age Ranking Experiment...")

        ## 1. 모델 로드 확인 (이미 main_worker 초반에 로드됨)
        if args.initial_model == "" and not modelfiles:
            print("Error: No model found. Use --initial_model to specify a trained model.")
            return
        
        current_model_path = args.initial_model or modelfiles[-1]
        print(f"Using model: {current_model_path}")
        
        ## 2. Train-set(young/old) 임베딩 추출
        print("Loading train list:", args.train_list)
        # AgeDatasetLoader는 train_path 인자를 데이터 루트 경로로 사용합니다.
        age_train_dataset = AgeDatasetLoader(args.train_list, **vars(args))
        age_train_loader = torch.utils.data.DataLoader(
            age_train_dataset,
            batch_size=args.batch_size, # 배치 사이즈는 조절 가능
            num_workers=args.nDataLoaderThread,
            pin_memory=False,
            drop_last=False
        )
        
        train_embeds, train_ages = extract_embeddings_with_age(s, age_train_loader, args.gpu)
        
        ## 3. embed_young, embed_old 계산
        young_embeds = train_embeds[train_ages < 24]
        old_embeds = train_embeds[train_ages > 68]
        
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
        # AgeDatasetLoader는 test_path가 아닌 train_path를 사용하므로
        # args.train_path가 test_list의 파일 경로도 포함해야 합니다.
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

        #거리 차이 분석 로직 끝
        
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

        return 
    
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
        young_embeds = train_embeds[train_ages < 20]
        old_embeds = train_embeds[train_ages > 77]
        
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
        pcc, p_value_pcc = pearsonr(rank_scores, test_ages)

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

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))

        if it % args.test_interval == 0:

            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()

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