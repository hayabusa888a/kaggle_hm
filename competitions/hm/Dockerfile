FROM rapidsai/base:26.04-cuda12.8-py3.12

WORKDIR /workspace

# 追加パッケージ
RUN pip install --no-cache-dir \
    jupyterlab \
    kaggle \
    python-dotenv \
    scikit-learn \
    matplotlib \
    seaborn \
    lightgbm \
    xgboost \
    catboost \
    implicit \
    polars \
    pyarrow \
    scipy \
    tqdm \
    cudf-polars-cu12

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
