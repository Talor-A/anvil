# decisions served
wc -l data/training/pauper-red-vs-islands/iter-000/mu.jsonl

# games completed so far
for f in data/runs/pauper-red-vs-islands-i000-*/workers/inv-*/games.jsonl; do
 echo "$f: $(wc -l < "$f")"; done
