#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -j y
#$ -N drift_null
#$ -cwd
#$ -t 1-48
#$ -tc 24

source /share/apps/source_files/python/python-3.9.5.source
source ~/myenv/bin/activate

export DP_SEED_BASE=0
export DP_NULL_KERNELS=1
export DP_STOP=550
export DP_OUT="$(pwd)/out"

python3 /home/ajafree/lobSimulations/HawkesRLTrading/drift_probe.py
