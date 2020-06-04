FROM python:3.7

COPY requirements.txt ./LSTM_dev/
RUN pip install -r ./LSTM_dev/requirements.txt
ADD inputs/* ./LSTM_dev/
ADD outputs/* ./LSTM_dev/
ADD env.sh ./
ADD predict_lstm.py ./LSTM_dev/

ENV DF_ROOT="./"
CMD [ "python", "./predict_lstm.py", "--getData", "--trainModel", "--predict", "--testProfit", "--daysSince", "366", "--daysUntil", "1"]