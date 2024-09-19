#!/bin/bash

# Loop through topics 1 to 11
for topic in {1..11}
do
  # Run the command for each topic and capture the output
  output=$(allorad q emissions latest-network-inference $topic --node https://allora-rpc.testnet.allora.network:443 | grep -f allora.txt)
  
  # Check if the output is not empty
  if [ -n "$output" ]; then
    # Print the topic number and the output
    echo "Topic $topic has return value:"
    echo "$output"
    echo ""
  fi
done
