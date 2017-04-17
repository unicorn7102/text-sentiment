#!/bin/bash

ini=0
fin=0
for ((i=${ini}; i<=${fin}; ++i))
do
cat << EOF > calSentScore${i}.sh
cp sentimentscoring.py sentimentscoring${i}.py
python3 sentimentscoring${i}.py ./Lists/list${i}
rm -rf sentimentscoring${i}.py
EOF
sh calSentScore${i}.sh
#nohup sh calSentScore${i}.sh > ./Logs/score.list${i} &
done
echo "done... from ${ini} to ${fin}"
