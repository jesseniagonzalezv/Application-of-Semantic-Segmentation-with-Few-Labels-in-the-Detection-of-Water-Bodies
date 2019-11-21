#!/bin/ bash
#Run as bash dice_plot.sh

echo dice

    python dice_comparation.py --percent 8 --fold-out 0 --fold-in 0,1,2,3,4
    #python dice_comparation.py --percent 20 --fold-out 0 --fold-in 0,1,2,3,4   
    #python dice_comparation.py --percent 40 --fold-out 0 --fold-in 1,3
    #python dice_comparation.py --percent 80 --fold-out 0 --fold-in 1,3,4

