FROM python:3.14-slim

WORKDIR /app

RUN pip install pipenv
RUN pipenv install --system --deploy
RUN pip install flask gunicorn
RUN pip install flask gunicorn scikit-learn

COPY predict.py model_C=1.0.bin ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]