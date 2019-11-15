#!/bin/ bash
#Run as bash trainparal.sh

echo hola

for i in 0 #1 2 3 4
do
  for j in 0 1 2 3 4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.06 --n-epochs 30
    python train_seq.py --fold-out $i  --fold-in $j --percent 0.06 --n-epochs 10 --n-steps 15 
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.06 --n-epochs 10 --n-steps 15 
   
    done
done

for i in 0 #1 2 3 4
do
  for j in 0  2  4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.70 --n-epochs 30
    python train_seq.py --fold-out $i  --fold-in $j --percent 0.70 --n-epochs 10 --n-steps 50 
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.70 --n-epochs 10 --n-steps 50 
   
    done
done

