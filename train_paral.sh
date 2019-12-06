#!/bin/ bash
#Run as bash trainparal.sh

echo hola

for i in 0 #1 2 3 4
do
  for j in 1 2 3 4
    do
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.80 --n-epochs 40 --n-steps 170 
    python plotting.py --out-file 'paral' --stage 'test' --name-file '_80_percent' --name-model 'UNet11' --fold-out $i --fold-in $j --count 94

    done
done

