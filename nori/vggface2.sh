[ -z "$REPLICA" ] && REPLICA=12
[ -z "$PREEMPTIBLE" ] && PREEMPTIBLE=no
[ -z "$CPU" ] && CPU=4
[ -z "$MEMORY" ] && MEMORY=64000
rlaunch -P$REPLICA --preemptible $PREEMPTIBLE --cpu=$CPU --memory=$MEMORY -- python3 sender.py --oup_pipe vggface2.hza.x1 --num-worker 4 --dataset vggface2