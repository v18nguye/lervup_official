"""
This file generates .slurm files to fine-tune models

Usage:

    python3 gen_slurm_split.py -f it_mobi.slurm -p allcpu -n node09 -e 1 -s IT -d MOBI -pft ORG -fe 1 -N 200
"""
import os
import argparse


def argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="bank_mobi.slurm", help="saved slurm file name")
    parser.add_argument("-dir", "--directory", default= '/home/users/apopescu/van-khoa/saved_models', help= 'saved directory')
    parser.add_argument("-p", "--partition", required= True, help= "partition of node")
    parser.add_argument("-n", "--node", default="", required=True, help="node where to train models")
    parser.add_argument("-e", "--exclusive", required=True, help="1 to turn on the exclusive mode")
    parser.add_argument("-s", "--situation", required=True, help="IT, ACCOM, BANK, WAIT")
    parser.add_argument("-d", "--detector", required=True, help="MOBI, RCNN")
    parser.add_argument("-pft", required=True, help="POOLINGx2, POOLING, ORG")
    parser.add_argument("-fe", required=True, help="1 to turn on the Focal Exposure")
    parser.add_argument("-N", required=True, help="Number of training profiles: "
                                                  "-1: ALL user profiles"
                                                  "N: N profiles (N < 400)")

    return parser

def main():
    """

    Returns
    -------

    """
    SEEDs = [10, 100, 1000, 100000, 1000000]
    MODEs = ['OBJECT'] # FE mode
    EPSs = [0.05, 0.1, 0.15, 0.2]
    KEEPs = [0.8, 0.85, 0.9, 0.95, 1.0]
    FEATURE_TYPEs = ['VOTE', 'ORG']
    args = argument_parser().parse_args()
    slurm_file_1 = args.file.replace('.slurm','_1.slurm')
    writer_1 = open(slurm_file_1,'w')
    writer_1.write('#!/bin/bash\n# SBATCH -N 1\n#SBATCH -n 72\n')
    writer_1.write('#SBATCH --partition '+args.partition+'\n')


    if args.exclusive == '1':
        writer_1.write('#SBATCH --exclusive\n')
    else:
        writer_1.write('#SBATCH -w '+args.node+'\n')
    short_name= args.file.replace('.slurm','')
    writer_1.write('#SBATCH -J '+short_name+"\n")

    writer_1.write("#SBATCH --output=outputs_slurm/%j"+args.file.replace('slurm','out')+"\n")
    writer_1.write("#SBATCH --error=outputs_slurm/%j"+args.file.replace('slurm', 'errs')+"\n")
    writer_1.write("export OMP_NUM_THREADS=1\n")    
    writer_1.write('mkdir -p outputs_slurm\n')

    slurm_file_2 = args.file.replace('.slurm','_2.slurm')
    writer_2 = open(slurm_file_2,'w')
    writer_2.write('#!/bin/bash\n# SBATCH -N 1\n#SBATCH -n 72\n')
    writer_2.write('#SBATCH --partition '+args.partition+'\n')


    if args.exclusive == '1':
        writer_2.write('#SBATCH --exclusive\n')
    else:
        writer_2.write('#SBATCH -w '+args.node+'\n')
    short_name= args.file.replace('.slurm','')
    writer_2.write('#SBATCH -J '+short_name+"\n")

    writer_2.write("#SBATCH --output=outputs_slurm/%j"+args.file.replace('slurm','out')+"\n")
    writer_2.write("#SBATCH --error=outputs_slurm/%j"+args.file.replace('slurm', 'errs')+"\n")
    writer_2.write("export OMP_NUM_THREADS=1\n")    
    writer_2.write('mkdir -p outputs_slurm\n')

    common_part = 'python3 ft_official.py --config_file ../configs/'
    save_model = args.file.split('.slurm')[0]
    out_dir = os.path.join(args.directory, args.file.split('.slurm')[0])
    counter = 0

    for seed in SEEDs:
        for mode in MODEs:
            for eps in EPSs:
                for keep in KEEPs:
                    for feature in FEATURE_TYPEs:
                        if args.detector == 'MOBI':
                            model_part = common_part+'rf_kmeans_ft_mobi_cv5.yaml --model_name '+save_model+'_'+str(counter)+'.pkl '+'--situation '+args.situation+' --fe '+args.fe+' --N '+str(args.N)
                        elif args.detector == 'RCNN':
                            model_part = common_part+'rf_kmeans_ft_rcnn_cv5.yaml --model_name '+save_model+'_'+str(counter)+'.pkl '+'--situation '+args.situation+' --fe '+args.fe+' --N '+str(args.N)
                        param_part = model_part+' --opts'+' MODEL.SEED '+str(seed)+' FE.MODE '+mode+' USER_SELECTOR.EPS '+str(eps)+' SOLVER.PFT '+args.pft+' USER_SELECTOR.KEEP '+str(keep)+' SOLVER.FEATURE_TYPE '+str(feature)+' OUTPUT.DIR '+out_dir
                        if counter < 20:
                            writer_1.write(param_part+' &\n')
                        else:
                            writer_2.write(param_part+' &\n')
                        counter += 1

    writer_1.write('wait\n')
    writer_2.write('wait\n')    
    writer_1.close()
    writer_2.close()

if __name__ == '__main__':
    main()
