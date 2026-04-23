FROM python:3.11-slim

WORKDIR /workspace

# 必要なシステムパッケージ
RUN apt-get update && apt-get install -y \
    git curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter起動ポート
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]