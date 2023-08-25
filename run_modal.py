# Trying to call the deployed function from another python session

import modal
import json
import sys

rss = sys.argv[1]
out = sys.argv[2]

f = modal.Function.lookup("corise-podcast-project-whisper", "process_podcast")
output = f.remote(rss, '/content/podcast/')

import json
with open(out, "w") as outfile:
  json.dump(output, outfile)