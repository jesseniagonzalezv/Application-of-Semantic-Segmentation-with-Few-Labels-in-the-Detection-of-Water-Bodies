#!/bin/ bash
#Run as bash trainHR.sh

echo hola

for i in 0 #2 4
do
  for j in 3 4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.06 
    done
done



