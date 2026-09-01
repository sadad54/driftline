#!/usr/bin/env bash
cd ~/driftline
python3 -c "
import json
d = json.load(open('serving/sample_requests.json'))
json.dump(d[0], open('/tmp/one_req.json', 'w'))
"
curl -s -X POST http://localhost:8000/score -H 'Content-Type: application/json' -d @/tmp/one_req.json
echo
