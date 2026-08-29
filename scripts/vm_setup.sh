#!/usr/bin/env bash
set -euxo pipefail

sudo apt-get -qq update
sudo apt-get -y -qq install python3-pip python3-venv git jq unzip

# kubectl
KVER=$(curl -Ls https://dl.k8s.io/release/stable.txt)
curl -Lo /tmp/kubectl "https://dl.k8s.io/release/${KVER}/bin/linux/amd64/kubectl"
sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl

# k3d
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | sudo bash

# helm
curl -s https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | sudo bash

echo "=== versions ==="
python3 --version
kubectl version --client
k3d version
helm version
git --version
docker --version
