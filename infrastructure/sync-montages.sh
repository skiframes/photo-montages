#!/bin/bash
# Sync photo montage sessions from edge device to S3
# Usage: ./sync-montages.sh <session_dir>
#    or: ./sync-montages.sh --auto   (scan output dir for new/updated sessions)

set -e

# Ensure aws CLI is on PATH (pip install location)
export PATH="$PATH:$HOME/.local/bin"

# Configuration
BUCKET_NAME="avillachlab-netm"
REGION="us-east-1"
MEDIA_DISTRIBUTION_ID="E1NKIYZ9037N7Q"
OUTPUT_DIR="/home/paul/skiframes/photo-montages/output"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --auto mode: scan output dir for sessions needing sync
if [ "$1" = "--auto" ]; then
    SYNCED_ANY=false
    for dir in "$OUTPUT_DIR"/*/; do
        [ -d "$dir" ] || continue
        [ -f "$dir/manifest.json" ] || continue

        if [ ! -f "$dir/.synced" ]; then
            # Never synced — sync now
            "$0" "$dir" && SYNCED_ANY=true
        elif [ "$dir/manifest.json" -nt "$dir/.synced" ]; then
            # Manifest updated since last sync — re-sync (incremental)
            "$0" "$dir" && SYNCED_ANY=true
        fi
    done
    exit 0
fi

# Single session sync mode
SESSION_DIR="$1"

if [ -z "$SESSION_DIR" ]; then
    echo -e "${RED}Usage: $0 <session_directory>${NC}"
    echo "   or: $0 --auto"
    exit 1
fi

# Remove trailing slash
SESSION_DIR="${SESSION_DIR%/}"

if [ ! -d "$SESSION_DIR" ]; then
    echo -e "${RED}Directory not found: $SESSION_DIR${NC}"
    exit 1
fi

MANIFEST_PATH="$SESSION_DIR/manifest.json"
if [ ! -f "$MANIFEST_PATH" ]; then
    echo -e "${RED}No manifest.json found in: $SESSION_DIR${NC}"
    exit 1
fi

DIR_NAME=$(basename "$SESSION_DIR")

# Derive event_id from session directory name
# Format: 2026-02-07_0921_u10_training → event_id: 2026-02-07_u10_training (drop HHMM)
EVENT_ID=$(python3 -c "
import re
d = '$DIR_NAME'
# Match: YYYY-MM-DD_HHMM_group_type
m = re.match(r'^(\d{4}-\d{2}-\d{2})_\d{4}_(.+)$', d)
if m:
    print(f'{m.group(1)}_{m.group(2)}')
else:
    # Fallback: use as-is
    print(d)
" 2>/dev/null)

echo -e "${YELLOW}Syncing montage session to S3...${NC}"
echo "Source: $SESSION_DIR"
echo "Event ID: $EVENT_ID"
echo "Destination: s3://$BUCKET_NAME/events/$EVENT_ID/"

# Sync all image files to S3
aws s3 sync "$SESSION_DIR" "s3://$BUCKET_NAME/events/$EVENT_ID/" \
    --exclude ".DS_Store" \
    --exclude "*.log" \
    --exclude "*.txt" \
    --exclude ".synced" \
    --region "$REGION"

# Upload manifest with proper content type and short cache
aws s3 cp "$MANIFEST_PATH" "s3://$BUCKET_NAME/events/$EVENT_ID/manifest.json" \
    --content-type "application/json" \
    --cache-control "max-age=60" \
    --region "$REGION"

# Extract event info from manifest for index update
EVENT_NAME=$(python3 -c "
import json
try:
    with open('$MANIFEST_PATH') as f:
        m = json.load(f)
    group = m.get('group', '')
    event_type = m.get('event_type', 'training')
    discipline = m.get('discipline', 'freeski')
    event_date = m.get('event_date', '$EVENT_ID'.split('_')[0])
    # Map discipline codes to display names
    disc_names = {'sl_youth': 'SL', 'sl_adult': 'SL', 'gs_panel': 'GS', 'sg_panel': 'SG', 'freeski': 'Free Ski'}
    disc_label = disc_names.get(discipline, discipline)
    # Build readable name: 'U14 SL Training - 2026-02-07' or 'U14 Free Ski Training - 2026-02-07'
    parts = []
    if group:
        parts.append(group.upper())
    if disc_label:
        parts.append(disc_label)
    parts.append(event_type.title())
    parts.append(event_date)
    print(' - '.join(parts) if len(parts) > 1 else parts[0])
except:
    print('$EVENT_ID')
" 2>/dev/null)

EVENT_DATE=$(python3 -c "
import json
try:
    with open('$MANIFEST_PATH') as f:
        m = json.load(f)
    print(m.get('event_date') or '$EVENT_ID'.split('_')[0])
except:
    print('$EVENT_ID'.split('_')[0])
" 2>/dev/null)

EVENT_TYPE=$(python3 -c "
import json
try:
    with open('$MANIFEST_PATH') as f:
        m = json.load(f)
    print(m.get('event_type', 'training'))
except:
    print('training')
" 2>/dev/null)

MONTAGE_COUNT=$(python3 -c "
import json
try:
    with open('$MANIFEST_PATH') as f:
        m = json.load(f)
    print(len(m.get('runs', [])))
except:
    print(0)
" 2>/dev/null)

GROUP=$(python3 -c "
import json
try:
    with open('$MANIFEST_PATH') as f:
        m = json.load(f)
    print(m.get('group', ''))
except:
    print('')
" 2>/dev/null)

DISCIPLINE=$(python3 -c "
import json
try:
    with open('$MANIFEST_PATH') as f:
        m = json.load(f)
    print(m.get('discipline', 'freeski'))
except:
    print('freeski')
" 2>/dev/null)

# Update root index.json
echo -e "${YELLOW}Updating root index.json...${NC}"

INDEX_PATH="/tmp/skiframes_index.json"
aws s3 cp "s3://$BUCKET_NAME/index.json" "$INDEX_PATH" --region "$REGION" 2>/dev/null || echo '{"events":[],"last_updated":""}' > "$INDEX_PATH"

python3 << EOF
import json
from datetime import datetime

with open('$INDEX_PATH', 'r') as f:
    index = json.load(f)

event_id = '$EVENT_ID'
event_name = '$EVENT_NAME'
event_date = '$EVENT_DATE'
event_type = '$EVENT_TYPE'
montage_count = int('$MONTAGE_COUNT' or 0)
group = '$GROUP'
discipline = '$DISCIPLINE'

# Convert old string format to object format if needed
if index['events'] and isinstance(index['events'][0], str):
    index['events'] = [{'event_id': e, 'event_name': e, 'event_date': e.split('_')[0]} for e in index['events']]

# Check if event already exists
existing = next((e for e in index['events'] if e.get('event_id') == event_id), None)

if existing:
    # Update existing event
    existing['event_name'] = event_name
    existing['event_date'] = event_date
    existing['event_type'] = event_type
    existing['discipline'] = discipline
    existing['montage_count'] = montage_count
    if group:
        existing['categories'] = [group.upper()]
    print(f"Updated {event_id} in index (montages: {montage_count}, discipline: {discipline})")
else:
    # Add new event
    new_event = {
        'event_id': event_id,
        'event_name': event_name,
        'event_date': event_date,
        'event_type': event_type,
        'discipline': discipline,
        'location': 'Ragged Mountain, NH',
        'montage_count': montage_count,
        'video_count': 0,
        'teams': []
    }
    if group:
        new_event['categories'] = [group.upper()]
    index['events'].append(new_event)
    print(f"Added {event_id} to index (montages: {montage_count}, discipline: {discipline})")

# Sort by date descending
index['events'].sort(key=lambda x: x.get('event_date', ''), reverse=True)
index['last_updated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

with open('$INDEX_PATH', 'w') as f:
    json.dump(index, f, indent=2)
EOF

# Upload updated index
aws s3 cp "$INDEX_PATH" "s3://$BUCKET_NAME/index.json" \
    --content-type "application/json" \
    --cache-control "max-age=60" \
    --region "$REGION"

# Invalidate CloudFront cache for this event and index
if [ -n "$MEDIA_DISTRIBUTION_ID" ]; then
    echo -e "${YELLOW}Invalidating CloudFront cache...${NC}"
    aws cloudfront create-invalidation \
        --distribution-id "$MEDIA_DISTRIBUTION_ID" \
        --paths "/index.json" "/events/$EVENT_ID/*" \
        --region "$REGION" \
        --output text > /dev/null 2>&1 || echo -e "${RED}CloudFront invalidation failed (non-fatal)${NC}"
fi

# Mark as synced
touch "$SESSION_DIR/.synced"

echo -e "${GREEN}Sync complete!${NC}"
echo "Event: $EVENT_NAME"
echo "Montages: $MONTAGE_COUNT"
echo "Media URL: https://media.skiframes.com/events/$EVENT_ID/"
echo "View at: https://skiframes.com/event.html?event=$EVENT_ID"
