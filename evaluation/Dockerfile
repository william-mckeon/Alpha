FROM python:3.12.14-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY evaluation/requirements-eval.lock /tmp/requirements-eval.lock
RUN pip install --no-cache-dir -r /tmp/requirements-eval.lock
COPY evaluation /app/evaluation
COPY scripts/eval_foundations.py /app/scripts/eval_foundations.py
CMD ["python", "scripts/eval_foundations.py", "validate"]
