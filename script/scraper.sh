#!/bin/bash

for ((i=$2;i<$3;i++))
do
  echo $i
  python -m src.retrieval.scraper_for_knowledge_store -i data_store/"$1"_store/$i.tsv -o data_store/output_"$1" &
done

wait