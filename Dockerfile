# Use a clean, official lightweight Python base image
FROM python:3.12-slim

# Step 1: Copy the fast uv binary directly from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Step 2: Establish the working directory inside the container
WORKDIR /app

# Step 3: Optimize configurations for running uv inside a container
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Step 4: Copy ONLY your dependency configuration files first
# This allows Docker to cache your installed libraries so you don't re-download 
# them every time you make minor edits to your application code files.
COPY pyproject.toml uv.lock ./

# Step 5: Pre-install all project dependencies using your lockfile
# --frozen ensures uv strictly respects uv.lock without trying to update it.
# --no-dev keeps development tools out of the production build image.
RUN uv sync --frozen --no-dev --no-install-project

# Step 6: Copy the rest of your application source code into the container
COPY . .

# Step 7: Final sync to register the source code package structure
RUN uv sync --frozen --no-dev

# Step 8: Define how to run the container application by default
# (You can modify this later to run your web API, like: ["uv", "run", "fastapi", "run"])
CMD ["uv", "run", "pytest"]
