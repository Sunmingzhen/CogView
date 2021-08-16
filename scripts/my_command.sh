# 第一个实验，在webvid数据集上，根据输入文本预测4个帧: cogview_webvid-2021-08-13-12:11:20
# Webvid数据集，原始处理分布
deepspeed --master_port 29502 --num_nodes 1 --num_gpus 4 pretrain_gpt2.py \
--batch-size 48 \
--experiment-name cogview_webvid \
--img-tokenizer-num-tokens 8192 \
--dataset-type WebvidTokensDataset \
--train-data /raid/datasets/video_datasets/webvid/ \
--model-parallel-size 1 \
--num-layers 12 \
--hidden-size 512 \
--num-attention-heads 16 \
--save /home/user/mzsun/codes/CogView/experiments/ \
--train-iters 40000 \
--distributed-backend nccl \
--lr-decay-style cosine \
--warmup .1 \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--max-position-embeddings 1089 \
--max-memory-length 0 \
--txt-loss-scale 5 \
--fp16 \
--deepspeed --deepspeed_config /home/user/mzsun/codes/CogView/scripts/ds_config.json

# 续第一个实验，在webvid数据集上，根据输入文本预测4个帧
# Webvid数据集，原始处理分布
deepspeed --master_port 29502 --num_nodes 1 --num_gpus 4 pretrain_gpt2.py \
--load /raid/users/mzsun/codes/CogView/experiments/cogview_webvid-2021-08-13-12:11:20/ --fast-load \
--batch-size 48 \
--experiment-name cogview_webvid \
--img-tokenizer-num-tokens 8192 \
--dataset-type WebvidTokensDataset \
--train-data /raid/datasets/video_datasets/webvid/ \
--model-parallel-size 1 \
--num-layers 12 \
--hidden-size 512 \
--num-attention-heads 16 \
--save /home/user/mzsun/codes/CogView/experiments/ \
--train-iters 100000 \
--distributed-backend nccl \
--lr-decay-style cosine \
--warmup .1 \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--max-position-embeddings 1089 \
--max-memory-length 0 \
--txt-loss-scale 5 \
--fp16 \
--deepspeed --deepspeed_config /home/user/mzsun/codes/CogView/scripts/ds_config.json


# 第二个实验，在webvid数据集上，根据输入文本预测4个帧: cogview_webvid-2021-08-15-13:48:38
# 降低文本预测损失函数，提升图像预测损失函数
deepspeed --master_port 29502 --num_nodes 1 --num_gpus 4 pretrain_gpt2.py \
--load /raid/users/mzsun/codes/CogView/experiments/cogview_webvid-2021-08-15-13:48:38/ --fast-load \
--batch-size 48 \
--experiment-name cogview_webvid \
--img-tokenizer-num-tokens 8192 \
--dataset-type WebvidTokensDataset \
--train-data /raid/datasets/video_datasets/webvid/ \
--model-parallel-size 1 \
--num-layers 12 \
--hidden-size 512 \
--num-attention-heads 16 \
--save /home/user/mzsun/codes/CogView/experiments/ \
--train-iters 40000 \
--distributed-backend nccl \
--lr-decay-style cosine \
--warmup .1 \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--max-position-embeddings 1089 \
--max-memory-length 0 \
--img-loss-scale 5 \
--txt-loss-scale 1 \
--fp16 \
--deepspeed --deepspeed_config /home/user/mzsun/codes/CogView/scripts/ds_config.json


# 第三个实验，在webvid数据集上，根据输入文本预测4个帧
# 输入64x64维度的图像
deepspeed --include localhost:1 --master_port 29503 pretrain_gpt2.py \
--batch-size 48 \
--image-size 64 \
--experiment-name cogview_webvid_img64_ds3 \
--img-tokenizer-num-tokens 8192 \
--dataset-type WebvidFramesDataset \
--model-parallel-size 1 \
--num-layers 12 \
--hidden-size 512 \
--num-attention-heads 16 \
--train-iters 40000 \
--distributed-backend nccl \
--lr-decay-style cosine \
--warmup .1 --fp16 \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--max-position-embeddings 1089 \
--max-memory-length 0 \
--img-loss-scale 5 \
--txt-loss-scale 1 \
--train-data /raid/datasets/video_datasets/webvid/ \
--save /home/user/mzsun/codes/CogView/experiments/ \
--img-tokenizer-path /home/user/mzsun/codes/Video_VQVAE/pretrained/OPENAI/ \
--deepspeed --deepspeed_config /home/user/mzsun/codes/CogView/scripts/ds_config.json



# 生成效果展示, 在webvid数据集上
python generate_samples.py \
--deepspeed \
--model-parallel-size 1 \
--num-layers 12 \
--hidden-size 512 \
--load /raid/users/mzsun/codes/CogView/experiments/cogview_webvid-2021-08-13-12:11:20/ \
--num-attention-heads 16 \
--max-position-embeddings 1089 \
--fp16 \
--temperature 1. \
--top_k 200 \
--top_p 0 \
--img-tokenizer-path /home/user/mzsun/codes/Video_VQVAE/pretrained/OPENAI/ \
--query-window 64 \
--key-window-times 4 \
--num-pivot 256 \
--is-sparse 0 \
--max-position-embeddings-finetune 1089 \
--generation-task predict4frames \
--input-source /home/user/mzsun/codes/CogView/generate_inputs.txt \
--output-path /raid/users/mzsun/codes/CogView/experiments/cogview_webvid-2021-08-13-12:11:20/samples_predict4frames \
--batch-size 4 \
--max-inference-batch-size 4 \
--device 0