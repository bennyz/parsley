FROM python:3 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target=/app/lib -f https://download.pytorch.org/whl/cpu/torch_stable.html

FROM python:3 AS runtime
WORKDIR /app
COPY --from=builder /app/lib /app/lib
ENV PYTHONPATH=/app/lib
COPY . .
RUN chmod -R g+rwX /app
EXPOSE 8080
CMD ["python", "app/handler.py"]
