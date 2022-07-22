while read one;
do
    echo $one
    wget $one -P '/mnt/disks/ds/datasets/kinetics/kinetics-400/rars/val/'
done < $1