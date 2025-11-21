import sys

# --- 1. 경로 설정 ---
# ❗️입력 파일: '스피커당 2개' 리스트
input_list_path = 'vox2_age_2_per_spk.txt' 

# ❗️출력 파일 1: 학습용 리스트 (Speaker-Disjoint)
output_train_path = 'vox2_age_train_set_sep.txt'
# ❗️출력 파일 2: 테스트용 리스트 (Speaker-Disjoint)
output_test_path = 'vox2_age_test_set_sep.txt'
# ---------------------

print(f"Reading list from: {input_list_path}...")

lines_written_train = 0
lines_written_test = 0
speaker_count = 0  # 라인 카운트 대신 '스피커 카운트'를 사용

try:
    with open(input_list_path, 'r') as f_in, \
         open(output_train_path, 'w') as f_train_out, \
         open(output_test_path, 'w') as f_test_out:
        
        # 2. 파일을 2줄씩 (스피커 1명 단위로) 읽어들입니다.
        # zip(f_in, f_in)은이터레이터를 두 개 묶어 (line1, line2) 페어를 만듭니다.
        for line1, line2 in zip(f_in, f_in):
            line1 = line1.strip()
            line2 = line2.strip()

            if not line1 or not line2: # 빈 줄이 섞여있으면 건너뛰기
                continue
            
            # 3. 스피커 번호(speaker_count)가 홀수/짝수인지로 분리
            if speaker_count % 2 == 0:
                # 짝수 번째 스피커 (0, 2, 4, ...) -> Train Set
                f_train_out.write(line1 + '\n')
                f_train_out.write(line2 + '\n')
                lines_written_train += 2
            else:
                # 홀수 번째 스피커 (1, 3, 5, ...) -> Test Set
                f_test_out.write(line1 + '\n')
                f_test_out.write(line2 + '\n')
                lines_written_test += 2
            
            speaker_count += 1 # 다음 스피커로 카운트 증가

except FileNotFoundError:
    print(f"Error: Input file not found at {input_list_path}")
    print("Please check the 'input_list_path' variable in the script.")
    sys.exit()

print("-" * 30)
print(f"Successfully split the file by SPEAKER (Disjoint).")
print(f"Total speakers processed: {speaker_count}")
print(f"Wrote {lines_written_train} lines ({lines_written_train // 2} speakers) to {output_train_path}")
print(f"Wrote {lines_written_test} lines ({lines_written_test // 2} speakers) to {output_test_path}")