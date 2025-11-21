import pandas as pd

# 1. 나이 CSV 파일을 읽어 '나이 지도' (dictionary) 생성
# 'VoxCeleb_ID'가 'id02022' 형식이므로 'id'를 제거하고 사용

age_csv_path = 'datasets/manifests/age-train.txt'
age_df = pd.read_csv(age_csv_path)

# (중요) ID를 기준으로 고유한 나이 맵 생성
# 예: {'id02022': 44.0, 'id03530': 67.0, ...}
age_map = age_df.set_index('VoxCeleb_ID')['speaker_age'].to_dict()
print(f"Loaded {len(age_map)} speaker ages.")

# 2. 기존 pair 리스트에서 '고유한' 파일 경로 모두 추출
pair_list_path = 'datasets/manifests/vox2_train_list.txt'
unique_files = set() # set을 사용해 중복 자동 제거

with open(pair_list_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            # parts[1] (id10001/Y8h.../00001.wav)
            # parts[2] (id10001/1zc.../00001.wav)
            unique_files.add(parts[1])

print(f"Found {len(unique_files)} unique test files.")

# 3. 새로운 'vox1_age_test_list.txt' 파일 생성
new_list_path = 'vox2_age_test_list.txt' # 저장할 새 파일 이름
files_written = 0
files_skipped = 0

with open(new_list_path, 'w') as f_out:
    for file_path in sorted(list(unique_files)): # 정렬해서 쓰기
        
        # 파일 경로에서 VoxCeleb_ID 추출 (예: 'id10001')
        try:
            vox_id = file_path.split('/')[0]
        except Exception:
            print(f"Skipping malformed path: {file_path}")
            continue
            
        # '나이 지도'에서 나이 검색
        age = age_map.get(vox_id)
        
        if age is not None:
            # 나이를 찾았으면 새 파일에 "파일경로 나이" 형식으로 쓰기
            f_out.write(f"{file_path} {age}\n")
            files_written += 1
        else:
            # 나이 정보가 없는 ID는 건너뛰기
            files_skipped += 1

print(f"Successfully wrote {files_written} files to {new_list_path}.")
print(f"Skipped {files_skipped} files (age not found).")