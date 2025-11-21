import random
import sys
from collections import defaultdict

# --- 1. 경로 설정 ---
# ❗️입력 파일은 'vox2_age_test_list.txt'가 맞습니다.
input_list_path = 'datasets/manifests/vox2_age_test_list.txt' 

# ❗️새로 저장할 '샘플링된' 리스트 파일 이름
output_list_path = 'vox2_age_2_per_spk.txt'
# ---------------------

print(f"Reading full list from: {input_list_path}...")

# ⬇️⬇️⬇️ 2번 섹션 (수정됨) ⬇️⬇️⬇️

# 2. 모든 (파일 경로, 나이) 쌍을 스피커 ID별로 그룹화
speaker_files = defaultdict(list)

try:
    with open(input_list_path, 'r') as f_in:
        for line in f_in:
            parts = line.strip().split()
            
            # 파일 형식은 '[Path] [Age]' (2열)
            if len(parts) == 2:
                file_path = parts[0] # ⬅️ 이것이 실제 파일 경로
                age = parts[1]       # ⬅️ 이것이 나이
                
                try:
                    # ⬅️ 실제 ID (id00012)를 파일 경로에서 추출
                    speaker_id = file_path.split('/')[0] 
                except Exception:
                    print(f"Skipping malformed path: {file_path}")
                    continue 
                    
                # ⬅️ (수정) 실제 ID를 key로, (경로, 나이) 튜플을 value로 저장
                speaker_files[speaker_id].append( (file_path, age) )
                
except FileNotFoundError:
    print(f"Error: Input file not found at {input_list_path}")
    print("Please check the 'input_list_path' variable in the script.")
    sys.exit()

print(f"Found {len(speaker_files)} unique speaker IDs.")

# ⬆️⬆️⬆️ 2번 섹션 (수정됨) ⬆️⬆️⬆️


# ⬇️⬇️⬇️ 3번 섹션 (수정됨) ⬇️⬇️⬇️

# 3. 각 스피커 ID 그룹에서 랜덤으로 2개씩 샘플링
total_files_written = 0
total_speakers_written = 0

with open(output_list_path, 'w') as f_out:
    # 딕셔너리의 (key, value) 쌍을 순회
    # (수정) all_paths -> all_samples 로 변수명 변경
    for speaker_id, all_samples in speaker_files.items():
        
        # ❗️(중요) 해당 스피커가 2개 이상의 파일을 가질 때만 처리
        if len(all_samples) >= 2:
            # 리스트 'all_samples'에서 랜덤으로 2개의 (경로, 나이) 튜플을 뽑음
            selected_samples = random.sample(all_samples, 2)
            
            total_speakers_written += 1
            
            # (수정) 
            # 선택된 2개의 샘플을 "Path Age" 형식으로 씀
            for (path, age) in selected_samples:
                f_out.write(f"{path} {age}\n")
                total_files_written += 1

print("-" * 30)
print(f"Successfully created new list at: {output_list_path}")
print(f"Wrote {total_files_written} total files.")
print(f"Included {total_speakers_written} speakers (who had >= 2 files).")