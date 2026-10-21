# Execution and measurement environment: every test, profile and analyzer run happens here.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip g++ openjdk-21-jdk-headless nodejs ruby golang-go \
      linux-tools-common linux-tools-generic util-linux git ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /harness
COPY pyproject.toml configs/analyzers/env_requirements.txt ./
RUN python3.12 -m venv /venv && /venv/bin/pip install --no-cache-dir -r env_requirements.txt
COPY . .
RUN /venv/bin/pip install --no-cache-dir -e .
ENV PATH=/venv/bin:$PATH
ENTRYPOINT ["harness"]
