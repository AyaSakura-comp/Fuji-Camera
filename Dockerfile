# Web/pool server only — NO GPU, NO torch. Generation runs on the host via
# gen_service.py (reached over HTTP at FUJI_GEN_URL). Tiny, portable image.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pillow python-multipart

COPY server.py .
COPY static ./static

ENV FUJI_GEN_URL=http://host.docker.internal:7863
EXPOSE 8090
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
