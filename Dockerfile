FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt ./


RUN pip install -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 4000

CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]
