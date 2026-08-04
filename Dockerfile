# One image serves both the API and the built React UI.
#
# Stage 1 builds the frontend to web/dist; stage 2 is the Python app that serves
# that build and runs the nightly job. Cloud-agnostic: runs on any container host.

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build


FROM python:3.12-slim AS app
WORKDIR /app

# System deps kept minimal; psycopg[binary] needs no libpq at build time.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY frostlib/ ./frostlib/
COPY service/ ./service/
COPY prepare.py train.py predict.py alembic.ini ./
COPY --from=web /web/dist ./web/dist

# The model artifact is fitted at deploy time (it is not committed); a real
# deployment runs `python3 predict.py --fit` once against the prepared data.
EXPOSE 8000
CMD ["uvicorn", "service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
