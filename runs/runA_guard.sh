#!/bin/zsh
# Stop run A at 19:00 if still training (best.pt + last.pt are saved every epoch), so evaluation finishes before 23:00.
while pgrep -f "out runA" >/dev/null; do
  if [ "$(date +%H%M)" -ge 1900 ]; then
    done_ep=$(grep -c '"epoch"' runs/runA.log)
    pkill -INT -f "python3 -m corners.train --n-kp 16"; sleep 5; pkill -f "out runA"
    echo "$(date +%H:%M) guard: stopped run A after ${done_ep} of 10 epochs" >> runs/runA.log
    exit 0
  fi
  sleep 60
done
