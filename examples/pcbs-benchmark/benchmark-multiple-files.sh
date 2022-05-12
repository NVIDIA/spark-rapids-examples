# File path for the files
FILE_PREFIX="/media/rjafri/data/data/mortgage_parquet/"
# List of files to benchmark
FILE_NAMES="
20m 
40m 
95m
196m
293m
"

#Remove results from previous run 
if [[ -f cache-perf.txt ]]; then
   echo "Removing old file"
   rm cache-perf.txt
fi

for i in $FILE_NAMES 
do
  echo "Running benchmark for " $i
  ./pcbs-benchmark.sh $FILE_PREFIX$i
done
