#!/bin/ bash
#Run as bash trainparal.sh

echo hola

for i in 0 #1 2 3 4
do
  for j in 0 1 2 3 4
    do
    python train_seq.py --fold-out $i  --fold-in $j --percent 0.08 --n-epochs 30 --n-steps 34
    python train_seq.py --fold-out $i  --fold-in $j --percent 0.20 --n-epochs 15 --n-steps 68 
    done
done

for i in 0 
do
  for j in 1  3  
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.40 --n-epochs 30
    python train_seq.py --fold-out $i  --fold-in $j --percent 0.40 --n-epochs 30 --n-steps 136 
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.40 --n-epochs 30 --n-steps 136 
   
    done
done

for i in 0 
do
  for j in  1  3  4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.80 --n-epochs 30
    python train_seq.py --fold-out $i  --fold-in $j --percent 0.80 --n-epochs 30 --n-steps 170 
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.80 --n-epochs 30 --n-steps 170 
   
    done
done