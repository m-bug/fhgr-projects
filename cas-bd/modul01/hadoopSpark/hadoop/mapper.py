# mapper.py
import sys

# Befehl auf dem Hadoop-Cluster
#hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
#    -files mapper.py,reducer.py \
#    -mapper "python3 mapper.py" \
#    -reducer "python3 reducer.py" \
#    -input /user/hadoop/input.txt \
#    -output /user/hadoop/output_dir

for line in sys.stdin:
    words = line.strip().split()
    for word in words:
        # Gibt für jedes Wort eine 1 aus (wird auf die Festplatte geschrieben)
        print(f"{word}\t1")