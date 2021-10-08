
set linetype  1 lc rgb "#9400d3"
set linetype  2 lc rgb "#009e73"
set linetype  3 lc rgb "#56b4e9"
set linetype  4 lc rgb "#e69f00"
set linetype  5 lc rgb "#f0e442"
set linetype  6 lc rgb "#0072b2"
set linetype  7 lc rgb "#000000" dt 3
set linetype  8 lc rgb "#000000" dt 2

#set terminal pngcairo
set terminal eps
set logscale
set xlabel "Total Samples"
set ylabel "MSE"
set format x "10^{%L}"
set format y "10^{%L}"

#set output "./ne_testcase_compare.png"
set output "./ne_testcase_compare.eps"
plot "./ne_testcase_nmc.txt" using 1:2 with lines title "NMC" lt 1, \
    "./ne_testcase_strong.txt" using 1:2 with lines title "GAM-based" lt 4, \
    "./ne_testcase_proposed.txt" using 1:2 with lines title "proposed" lt 3, \
    "./ne_testcase_proposed_reg.txt" using 1:2 with lines title "proposed with regression" lt 6, \
    [1000:100000] sqrt(1/x) * sqrt(0.1) with lines title "N^{-1/2}" lt 7, \
    [1000:100000] 1/x * 0.1 with lines title "N^{-1}" lt 8

#set output "./evsi_testcase_compare.png"
set output "./evsi_testcase_compare.eps"
plot "./evsi_testcase_nmc.txt" using 1:2 with lines title "NMC" lt 1, \
    "./evsi_testcase_strong.txt" using 1:2 with lines title "GAM-based" lt 4, \
    "./evsi_testcase_proposed.txt" using 1:2 with lines title "proposed" lt 3, \
    "./evsi_testcase_proposed_reg.txt" using 1:2 with lines title "proposed with regression" lt 6, \
    [1000:100000] sqrt(1/x) * sqrt(0.1) with lines title "N^{-1/2}" lt 7, \
    [1000:100000] 1/x * sqrt(0.1) with lines title "N^{-1}" lt 8

#set output "./evsi_medical_compare.png"
set output "./evsi_medical_compare.eps"
plot "./evsi_medical_mlmc.txt" using 1:2 with lines title "MLMC" lt 2, \
    "./evsi_medical_strong.txt" using 1:2 with lines title "GAM-based" lt 4, \
    "./evsi_medical_proposed.txt" using 1:2 with lines title "proposed" lt 3, \
    "./evsi_medical_proposed_reg.txt" using 1:2 with lines title "proposed with regression" lt 6, \
    [1000:100000] 1/x * 10000000 with lines title "N^{-1}" lt 8

