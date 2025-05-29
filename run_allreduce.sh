export WORLD_SIZE=2
export RANK=0
export MASTER_ADDR=192.168.3.41
export MASTER_PORT=12345

torchrun --nproc_per_node gpu \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT benchmark/kernels/all_reduce/benchmark_mscclpp.py
