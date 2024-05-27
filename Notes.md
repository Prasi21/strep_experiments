## Memory management
Check nvidia-smi to see the gpu usage
kill -9 {PID}

nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9