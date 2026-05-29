set datafile separator comma
set terminal pngcairo size 1200,800 enhanced font "Arial,12"
set output "predictions.png"

set title "Function Approximation"
set xlabel "x"
set ylabel "y"
set grid
set key outside

plot "predictions.csv" using 1:2 with lines linewidth 3 title "true", \
     "predictions.csv" using 1:3 with lines linewidth 3 title "predicted"
