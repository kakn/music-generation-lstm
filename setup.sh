#!/bin/bash

# Function to install Python 3.9 locally
install_python39() {
    echo "Installing Python 3.9 locally..."
    # Specify the version and the installation directory
    PYTHON_VERSION=3.9.0
    INSTALL_DIR=$HOME/localpython

    # Download Python 3.9
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

    # Extract and install
    tar -xzf Python-$PYTHON_VERSION.tgz
    cd Python-$PYTHON_VERSION
    ./configure --prefix=$INSTALL_DIR
    make
    make install
    cd ..

    # Cleanup
    rm -rf Python-$PYTHON_VERSION
    rm Python-$PYTHON_VERSION.tgz

    # Add local python bin to PATH
    export PATH=$INSTALL_DIR/bin:$PATH
}

# Check if Python 3.9 is installed
if ! command -v python3.9 &> /dev/null
then
    install_python39
fi

echo "Creating a Python virtual environment with Python 3.9..."
python3.9 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Downloading and unzipping the dataset..."
wget http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz -O clean_midi.tar.gz
mkdir -p datasets
tar -xzf clean_midi.tar.gz -C datasets
rm clean_midi.tar.gz
