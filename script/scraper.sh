#!/bin/bash

for ((i=$2;i<$3;i++))
do
  echo $i
  python -m retrieval.scraper_for_knowledge_store -i /mnt/data/factcheck/averitec-data/data_store/urls/"$1"/"$1"_store/$i.tsv -o /mnt/data/factcheck/averitec-data/data_store/new/output_"$1" &
done

wait