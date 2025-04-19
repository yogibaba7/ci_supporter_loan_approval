# base

FROM python:3.11

# workdir
WORKDIR /app

# copy
COPY model_serving/ /app/
COPY models/encoder.pkl /app/models/encoder.pkl
COPY models/imputer.pkl /app/models/imputer.pkl
COPY models/scaler.pkl /app/models/scaler.pkl

# run command
RUN pip install -r requirements.txt

# expose port
EXPOSE 8000

# CMD
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"] 