FROM python:3.14-slim

WORKDIR /app

# Install dependencies first so Docker can cache this step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (.dockerignore keeps secrets and junk out)
COPY . .

EXPOSE 5000

CMD ["python", "main.py", "--host", "0.0.0.0", "--no-browser"]