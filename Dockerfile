FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1


# deps first so they stay cached when only the code changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-dev


COPY . .


EXPOSE 5555

CMD [".venv/bin/python", "web/server.py", "--port", "5555"]
