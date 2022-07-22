while read one;
do
    echo ${one##*/}
    tar zxvf '/mnt/disks/ds/datasets/kinetics/kinetics-400/rars/val/'${one##*/} -C '/mnt/disks/ds/datasets/kinetics/kinetics-400/flat/val/'
done < $1

