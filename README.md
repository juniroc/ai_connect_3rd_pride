지난 2021년 9월에 진행되었던\
`판교 ai_camp 대회`\
자연어처리 분야의 3등을 기록한 학습 소스코드


### library
    - import library needed
    - 필요 라이브러리 import

### dataset
    - preprocessed data path
    - 전처리된 데이터 셋이 저장된 path
    1. korean dataset_origin (question / answer) 한글 질의/응답 데이터
    2. english dataset (by google api translator) 구글 API를 이용해 영어로 번역한 데이터셋 
    3. korean dataset (remove stopword)한국어 불용어 제거버전

### config
    - config_setting
    1. bert_model : 'bert-base-multilingual-cased' 
    2. batch_size : 32
    3. num_train_epochs : 5
    4. learning_rate : 5e-5 
    5. warmup_proportion = 0.1
    6. max_seq_length = 128
    7. output_dir = './output'
    8. device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### model_class
    - get config from init, and train_model
    - 위에서 지정한 config를 토대로 사전학습 버트 모델 가져오고, 이를 이용해 학습
    - data path, model_name, batch_size, epochs, learning_rate, max_seq_length, optimizer_config, saved_path, device(gpu) 
    - 데이터 경로, 모델 이름, 배치사이즈, 에폭수, 러닝레이트, 문장 최대길이, 옵티마이저 파라미터, 모델저장위치, gpu(있을경우 'cuda', 없을 경우 'cpu' 처리

1. get_dataexamples
    - get data from data path config
    - 데이터 경로로 부터 데이터 셋을 가져온다.

2. get_model_tokenizer
    - get tokenizer and model
    - 토크나이저와 모델을 가져온다.
    
3. get_optimizer
    - get optimizer (BertAdam)
    - 옵티마이저를 가져온다. (BertAdam)
    
4. get_data
    - get input_ids, mask, segment_ids, lbael_ids (tokeninzed data)
    - 토큰화 된 데이터 셋을 가져온다.
    - split question / answer by encoding, label encoding, masking 
    - 질의/응답을 구분하는 인코딩, 라벨을 인코딩, 단어의 마스킹처리를 진행한다.

5. save_model
    - save model and config to configured path (bin, json)
    - 모델과 초기값을 지정한 경로에 저장 (bin 파일과 json 형식으로 저장)
    
6. train_run
    - training model by methods upon
    - 위에서 지정한 메소드를 이용해 학습한다.
    - and if validation loss doesn't lower than before, it stop early (earlystoping method) 
    - 만약 val loss가 이전보다 2번 이상 낮아지지 않으면 학습을 종료한다.

### EarlyStopping
    - earlyStopping during training when validation loss doesn't lower than before 
    - 이전 val loss 보다 낮을 경우 학습 종료


### load_model
    - load pretrained model for inference
    - 인퍼런스를 위해 위에서 생성된 모델들을 불러온다.


### inference_testset
    - get inference result from each data(from data path) and return logit
    - 데이터 경로에서 testset 데이터를 불러와 인퍼런스 진행/로짓 출력
    

### average_ensemble
    - get average_ensemble result using loaded model
    - 로드한 모델에서 얻은 인퍼런스 결과 로짓들의 평균을 구해 라벨과 맵핑 시켜 결과 도출 
    - save result to csv file
    - 위에서 도출한 결과를 제출 양식에 맞춰 csv 로 저장
