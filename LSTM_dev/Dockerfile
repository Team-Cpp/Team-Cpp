FROM python:3.7

COPY requirements.txt ./
RUN pip install -r requirements.txt
ADD inputs/* ./
ADD output/* ./
ADD predict_lstm.py ./

CMD [ "python", "./predict_lstm.py"]