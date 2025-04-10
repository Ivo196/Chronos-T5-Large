FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /workspace/app/
COPY data/ /workspace/data/

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]