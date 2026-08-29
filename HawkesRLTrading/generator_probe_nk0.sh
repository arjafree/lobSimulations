#$ -S /bin/bash
#$ -l h_rt=24:00:00
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -j y
#$ -N genprobe_nk0
#$ -cwd
#$ -t 1-48
#$ -tc 24

source /share/apps/source_files/python/python-3.9.5.source
source ~/myenv/bin/activate

export GP_SEED_BASE=0
export GP_T=550
export GP_NULL_KERNELS=0
export GP_OUT="$(pwd)/out"

python3 /home/ajafree/lobSimulations/HawkesRLTrading/generator_probe.py
