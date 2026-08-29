#!/usr/bin/env bash
set -euxo pipefail
cd ~/driftline
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519' git pull

python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e .

docker compose up -d
echo "waiting for services to become healthy..."
sleep 5
docker compose ps
