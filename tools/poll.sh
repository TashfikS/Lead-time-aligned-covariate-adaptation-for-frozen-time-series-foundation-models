#!/usr/bin/env bash
# Return when any kernel in tools/kernels.tsv reaches a terminal state not yet recorded in tools/reported.txt.
touch tools/reported.txt
for i in $(seq 1 ${1:-19}); do
  line=""; new=""
  while IFS=$'\t' read -r V K; do
    [ -z "$K" ] && continue
    s=$(bash tools/kg.sh $V kernels status $K 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | cut -d. -f2)
    line="$line\n$K=$s"
    case "$s" in COMPLETE|ERROR|CANCEL*) grep -qx "$K" tools/reported.txt || { new="$new $K"; echo "$K" >> tools/reported.txt; };; esac
  done < tools/kernels.tsv
  if [ -n "$new" ]; then echo -e "NEW:$new$line"; exit 0; fi
  sleep 90
done
echo -e "TIMEOUT$line"
