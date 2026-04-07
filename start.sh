#!/bin/bash
cd "$(dirname "$0")"
python3 -m http.server &
sleep 1
open "http://localhost:8000"
wait
