#!/bin/ bash
#Run as bash trainparal.sh

echo hola

for i in 0 #1 2 3 4
do
  for j in 0 #1 #2 3 4
    do
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.08 --n-epochs 40 --n-steps 34 
    
    
    done
done

