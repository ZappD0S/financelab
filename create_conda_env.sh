set -e

source /opt/anaconda/bin/activate
conda create -y -n financelab python --file conda_requirements.txt -c pytorch
# conda create -n financelab python --file conda_requirements.txt
conda activate financelab
# conda install ~/Downloads/pytorch-build/linux-64/pytorch-1.7.0-custom.tar.bz2
pip install pyro-ppl


