[ -z "$REPLICA" ] && REPLICA=12
[ -z "$PREEMPTIBLE" ] && PREEMPTIBLE=no
[ -z "$CPU" ] && CPU=4
[ -z "$MEMORY" ] && MEMORY=64000
rlaunch -P$REPLICA --preemptible $PREEMPTIBLE --cpu=$CPU --memory=$MEMORY -- python3 sender.py --oup_pipe hza.pairwise.sample --dataset CASIA --sampler sample_xn2 --n 2 --num-worker 4
