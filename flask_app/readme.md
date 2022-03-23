## Set Current Working Directory

1. ```flask_app```으로 경로를 설정해주세요.

```cmd
cd flask_app

C:\Desktop\YOUR_DIR_NAME\KLUE-STS_API\flask_app>
```

## Requirements

2. ```requirement.txt``` 를 설치해 주세요.

```cmd
pip install -r requirements.txt
```
## Download Model Checkpoint

3. 다 설치가 되면 모델 체크포인트 파일을 다운받아 주세요.
4. ```flask_app/project/view/main.py```에 들어가서 체크포인트 파일 이름(경로)을 고쳐주세요.

```python
ckpt = torch.load((f'{YOUR_DIR_NAME}/{MODEL_CHECKPOINT_NAME}'), map_location = device)
```

## Execute App

5. 이제 아래 명령어를 cmd 창에 입력해주시면 됩니다.

```cmd
FLASK_APP=project flask run
```

6. 창이 뜨면 url 뒤에 ```/predict```를 입력해주면 됩니다.
