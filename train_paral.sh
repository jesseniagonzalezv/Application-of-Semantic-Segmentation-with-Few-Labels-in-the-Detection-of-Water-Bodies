#!/bin/ bash
#Run as bash trainparal.sh

echo hola

for i in 0 #1 2 3 4
do
  for j in 0 1 #2 3 4
    do
    python train_paral.py --fold-out $i  --fold-in $j --percent 0.06 --n-epochs 4 --n-steps 15 
    
    
    done
done

