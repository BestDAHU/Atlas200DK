#!/bin/bash
source ~/.bashrc
cd $(dirname $0);
result=`pmupload 2>/dev/null  | grep pmupload`
currentdir=`pwd`
if [ ! -z "$result" ];then
  echo "[INFO]  pmupload exists."
  pmupload $currentdir/workspace_mind_studio_Atlas200DK
else
  export SLOG_PRINT_TO_STDOUT=0
  $currentdir/workspace_mind_studio_Atlas200DK
fi
./workspace_mind_studio_Atlas200DK ../data/