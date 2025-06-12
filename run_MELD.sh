export CUDA_VISIBLE_DEVICES=3
# export CUBLAS_WORKSPACE_CONFIG=:16:8

python -u train_MELD.py --base-model 'GRU' --dropout 0.4 --lr 0.0001 --batch-size 16 \
	--graph_type='hyper' --epochs=15 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' \
	--modals='avl' --Dataset='MELD' --norm BN --proto_k=1