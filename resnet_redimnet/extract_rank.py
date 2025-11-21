#!/usr/bin/python
#-*- coding: utf-8 -*-

# 1. train.py에서 모든 import를 복사
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
from DatasetLoader import * # loadWAV, test_dataset_loader 등 필요
import torch.distributed as dist
import torch.multiprocessing as mp

# 2. ⭐️ 새로 추가된 import
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## 3. ⭐️ "나이 랭킹 리스트"를 읽는 새 Dataset 클래스 정의
## ===== ===== ===== ===== ===== ===== ===== =====
class ExtractionDataset(Dataset):
    def __init__(self, list_file, test_path, eval_frames):
        self.test_path = test_path
        self.eval_frames = eval_frames
        self.data_list = []
        
        # 'vox2_age_test_list.txt' (경로 나이) 파일을 읽음
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.data_list.append((parts[0], float(parts[1]))) # (path, age)

    def __getitem__(self, index):
        file_path, age = self.data_list[index]
        
        # loadWAV (eval모드, 1개 청크)
        audio = loadWAV(os.path.join(self.test_path, file_path), 
                        self.eval_frames, 
                        evalmode=True, 
                        num_eval=1)
        
        # [1, N] -> [N]
        return torch.FloatTensor(audio).squeeze(0), age, file_path

    def __len__(self):
        return len(self.data_list)

## ===== ===== ===== ===== ===== ===== ===== =====
## 4. train.py와 동일한 argparse (수정 없음)
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')
# ... (train.py의 모든 parser.add_argument를 여기에 복사) ...
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')
parser.add_argument('--train_list',     type=str,   default="datasets/manifests/vox2_train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="datasets/manifests/vox1-my_list1.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="/mnt/datasets/voxcelebs/voxceleb2/dev/wav", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="/mnt/datasets/voxcelebs/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="/mnt/datasets/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/mnt/datasets/simulated_rirs", help='Absolute path to the test set')
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

# 5. ⭐️ 새로 추가된 Argument
parser.add_argument('--output_file',    type=str,   default="embeddings.pt", help='Output file to save embeddings')

args = parser.parse_args()

## ===== ===== ===== ===== ===== ===== ===== =====
## 6. YAML 파싱 (train.py와 동일)
## ===== ===== ===== ===== ===== ===== ===== =====
def find_option_type(key, parser): # (train.py와 동일)
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None: # (train.py와 동일)
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

## ===== ===== ===== ===== ===== ===== ===== =====
## 7. ⭐️ 수정된 main_worker
## (학습/EER 평가 로직이 ➡️ 임베딩 추출 로직으로 변경)
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models (train.py와 동일)
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

    ## Load model weights (train.py와 동일)
    ## ⭐️ (주의) ModelTrainer를 로드용으로만 사용
    trainer = ModelTrainer(s, **vars(args))

    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
    
    print("Model loaded, starting embedding extraction...")

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## ⭐️ 8. 핵심 추출 로직
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    # 모델을 평가 모드로 설정
    s.eval()
    
    # 위에서 정의한 3번 ExtractionDataset 사용
    # ⭐️ (주의) --test_list 인자에 'vox2_age_test_list.txt'를 전달해야 함
    dataset = ExtractionDataset(list_file=args.test_list, 
                                test_path=args.test_path, 
                                eval_frames=args.eval_frames)
    
    loader = DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.nDataLoaderThread, 
                        drop_last=False)
    
    all_embeddings = []
    all_ages = []
    all_paths = []

    # (분산 학습일 경우 GPU 0번에서만 tqdm 표시)
    disable_tqdm = args.distributed and args.gpu != 0
    
    with torch.no_grad(): # ⭐️ 그라디언트 계산 비활성화
        for (audio, ages, paths) in tqdm(loader, desc="Extracting Embeddings", disable=disable_tqdm):
            
            audio = audio.cuda(args.gpu)
            
            # ⭐️ s(audio) 호출 -> SpeakerNet.forward(audio) 실행
            # (SpeakerNet.py의 forward가 라벨 없이 호출 시 임베딩을 반환한다고 가정)
            embedding = s(audio) 
            
            all_embeddings.append(embedding.detach().cpu())
            all_ages.extend(ages.tolist()) # batch의 나이 리스트 추가
            all_paths.extend(paths)      # batch의 경로 리스트 추가

    # (분산 학습일 경우 GPU 0번에서만 결과 취합 및 저장)
    if not args.distributed or args.gpu == 0:
        
        # ⭐️ 모든 배치의 결과를 하나의 텐서/리스트로 합침
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        
        # ⭐️ 결과를 딕셔너리로 저장
        results = {
            'embeddings': embeddings_tensor,
            'ages': all_ages,
            'paths': all_paths
        }
        
        # ⭐️ --output_file 인자로 받은 경로에 저장
        torch.save(results, args.output_file)
        
        print(f"\nSuccessfully extracted {len(all_paths)} embeddings.")
        print(f"Results saved to {args.output_file}")
    
    return

## ===== ===== ===== ===== ===== ===== ===== =====
## 9. Main function (train.py와 동일)
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
        mp.spawn(main_worker, n_gpus=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()