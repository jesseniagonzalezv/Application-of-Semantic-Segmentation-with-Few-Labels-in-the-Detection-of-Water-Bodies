#!/bin/ bash
#Run as bash trainHR.sh

echo hola

for i in 0 1 2 3 4
do
  for j in 0 1 2 3 4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.25
    done
done

for i in 0 1 2 3 4
do
  for j in 0 1 2 3 4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.45
    done
done

for i in 0 1 2 3 4
do
  for j in 0 1 2 3 4
    do
    python train_HR.py --fold-out $i  --fold-in $j --percent 0.70
    done
done