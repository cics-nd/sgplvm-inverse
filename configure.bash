#!/bin/bash
# One-stop shop for configuring this repo

# Get a virtual env ready:
VENV_DIR="venv"
if [ -d ${VENV_DIR} ]; then
    rm -rf ${VENV_DIR}
fi
# TODO ensure python>=3.5, not sure it will work with earlier versions.
virtualenv ${VENV_DIR} -p python3
source ${VENV_DIR}/bin/activate

# Get the required packages from GitHub:
echo "Clone and install required packages..."
PACKAGE_DIR="other_packages"
if [ -d ${PACKAGE_DIR} ]; then
    rm -rf ${PACKAGE_DIR}
fi
mkdir ${PACKAGE_DIR}

REQUIRED_REPO_URLS=("https://github.com/sdatkinson/GPflow.git" \
"https://github.com/cics-nd/structured-gpflow.git")
REQUIRED_REPO_NAMES=("GPflow" "structured-gpflow")
cd ${PACKAGE_DIR}
for (( I=0; I<${#REQUIRED_REPO_URLS[@]}; I++ )); do
    REPO_URL=${REQUIRED_REPO_URLS[${I}]}
    REPO_DIR=${REQUIRED_REPO_NAMES[${I}]}
    git clone ${REPO_URL} ${REPO_DIR}
    cd ${REPO_DIR}
    python setup.py install
    cd ..
done
cd ..

deactivate

echo "--done"
