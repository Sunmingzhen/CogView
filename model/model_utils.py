import torch
USE_TORCH_DDP = True
import os
import mpu
import sys
sys.path.append('..')
from data_utils import get_tokenizer

def get_masks_and_position_ids(data,
                            loss_mask=None,
                            attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        # single direction, [PAD]s are at the end of the seq, so doesn't matter.
        attention_mask = torch.ones((1, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
        attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    if args is not None and args.finetune and args.max_position_embeddings < args.max_position_embeddings_finetune:
        # for each sample, find [ROI2] and split
        # ([ROI1] text... [BOI1] img... [EOI1] [ROI2]<pos_id==1089> ...)
        start_token = get_tokenizer()['[ROI2]']
        tmp = torch.nonzero(data == start_token, as_tuple=False)
        start_token_poses = [100000] * batch_size
        for x, y in tmp:
            start_token_poses[x] = min(start_token_poses[x], y)
        assert 100000 not in start_token_poses, 'Some samples do not have [ROI2]!'
        position_ids = torch.zeros(batch_size, seq_length, dtype=torch.long,
                                    device=data.device)
        for i in range(batch_size):
            sep = start_token_poses[i]
            torch.arange(start=0, end=sep, out=position_ids[i, :sep],
                dtype=torch.long, device=data.device)
            second_pos = 0 # reuse
            torch.arange(start=second_pos, end=second_pos + seq_length - sep,
                out=position_ids[i, sep:],
                dtype=torch.long, device=data.device)
        position_ids[position_ids >= args.max_position_embeddings] = args.max_position_embeddings - 1
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '1234')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method = init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    # if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_activation_checkpointing:
    #     set_deepspeed_activation_checkpointing(args)

def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    reduced_losses = lm_loss.view(1)

    # Reduce losses for logging
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / args.world_size

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        if not USE_TORCH_DDP:
            timers('allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('allreduce').stop()

    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced