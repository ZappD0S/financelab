set -e

source /opt/anaconda/bin/activate
conda create -y -n financelab python --file conda_requirements.txt -c pytorch
conda activate financelab
pip install pyro-ppl


