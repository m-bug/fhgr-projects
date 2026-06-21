# mapper.py
import sys

for line in sys.stdin:
    words = line.strip().split()
    for word in words:
        # Gibt für jedes Wort eine 1 aus (wird auf die Festplatte geschrieben)
        print(f"{word}\t1")