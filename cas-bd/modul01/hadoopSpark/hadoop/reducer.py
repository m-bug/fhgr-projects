# reducer.py
import sys

# Befehl auf dem Hadoop-Cluster
#hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
#    -files mapper.py,reducer.py \
#    -mapper "python3 mapper.py" \
#    -reducer "python3 reducer.py" \
#    -input /user/hadoop/input.txt \
#    -output /user/hadoop/output_dir

current_word = None
current_count = 0

for line in sys.stdin:
    word, count = line.strip().split('\t', 1)
    count = int(count)
    
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = count

if current_word == word:
    print(f"{current_word}\t{current_count}")