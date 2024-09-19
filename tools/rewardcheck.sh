#!/bin/bash

# Define the file containing wallet IDs
FILE="allora.txt"
total_points_sum=0

checkPoints() {
  local walletID="$1"
  response=$(curl -s -H "accept: application/json, text/plain, */*" \
                    -H "accept-language: en-US,en;q=0.9" \
                    -H "cache-control: no-cache" \
                    -H "origin: https://app.allora.network" \
                    -H "pragma: no-cache" \
                    -H "priority: u=1, i" \
                    -H "referer: https://app.allora.network/" \
                    -H "sec-ch-ua: \"Not)A;Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"" \
                    -H "sec-ch-ua-mobile: ?0" \
                    -H "sec-ch-ua-platform: \"Windows\"" \
                    -H "sec-fetch-dest: empty" \
                    -H "sec-fetch-mode: cors" \
                    -H "sec-fetch-site: cross-site" \
                    -H "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36" \
                    -H "x-api-key: UP-XXXXXXXXX" \
                    "https://api.upshot.xyz/v2/allora/points/$walletID")

  #echo "API response: $response"  # Debug print
  
  # Validate JSON format
  echo "$response" | jq . > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "Error: Invalid JSON response"
    return
  fi
  
  status=$(echo "$response" | jq -r '.status')
  if [ "$status" = "true" ]; then
    cosmos_address=$(echo "$response" | jq -r '.data.cosmos_address')
    total_points=$(echo "$response" | jq -r '.data.allora_leaderboard_stats.total_points // "0"')
    rank=$(echo "$response" | jq -r '.data.allora_leaderboard_stats.rank // "N/A"')
    echo "Allo Address: $cosmos_address"
    echo "Total Points: $total_points"
    echo "Rank: $rank"
    
    echo "Campaign Points:"
    echo "$response" | jq -r '.data.campaign_points[] | "\(.campaign_slug): \(.points)"'
  else
    echo "Error: $(echo "$response" | jq -r '.apiResponseMessage // "Unknown error"')"
  fi
}

signin() {
  local wallet=$1
  local url="https://api.upshot.xyz/v2/allora/users/connect"
  local data="{\"allora_address\":\"$wallet\",\"evm_address\":null}"

  response=$(curl -s -X POST "$url" \
                   -H 'accept: application/json, text/plain, */*' \
                   -H 'accept-language: en-US,en;q=0.9' \
                   -H 'cache-control: no-cache' \
                   -H 'content-type: application/json' \
                   -H 'origin: https://app.allora.network' \
                   -H 'pragma: no-cache' \
                   -H 'priority: u=1, i' \
                   -H 'referer: https://app.allora.network/' \
                   -H 'sec-ch-ua: "Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"' \
                   -H 'sec-ch-ua-mobile: ?0' \
                   -H 'sec-ch-ua-platform: "Windows"' \
                   -H 'sec-fetch-dest: empty' \
                   -H 'sec-fetch-mode: cors' \
                   -H 'sec-fetch-site: cross-site' \
                   -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36' \
                   -H 'x-api-key: UP-XXXXXXXXXXX' \
                   --data "$data")

#  echo "signin response: $response"  # Debugging line

walletID=$(echo "$response" | jq -r '.data.id')
points=$(checkPoints "$walletID")
echo "$points"
echo
total_point=$(echo "$points" | awk '/Total Points:/ {print $3}' | grep -oE '[0-9]+([.][0-9]+)?')
total_point_sum=$(awk -v a="$total_point_sum" -v b="$total_point" 'BEGIN {printf "%.3f", a + b}')
}

# Read wallet IDs from file and process each
while IFS= read -r wallet; do
  signin "$wallet"
done < "$FILE"

echo "Total sum of points: $total_point_sum"
