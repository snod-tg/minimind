import time
import argparse
import os
import random
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')


def init_ep(args):
    if not args.use_ep:
        return 0, 1
    if 'model' not in args.load_from:
        raise ValueError('--use_ep currently only supports --load_from model')
    if int(os.environ.get('RANK', -1)) == -1:
        raise RuntimeError('--use_ep must be launched with torchrun')

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if backend == 'nccl':
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        args.device = f'cuda:{local_rank}'
    else:
        args.device = 'cpu'

    # EP 必然建立在 MoE 层之上，用户不需要再额外传 --use_moe 1。
    args.use_moe = 1
    return rank, world_size


def broadcast_object(value, rank):
    if not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            use_ep=bool(args.use_ep),
            ep_size=dist.get_world_size() if args.use_ep else None,
            num_experts=args.num_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        if args.use_ep:
            # 完整 MoE checkpoint 含有全部 Expert。先加载到 CPU，再只保留
            # 当前 rank 的共享参数和本地 Expert，避免把远端 Expert 放上 GPU。
            full_state_dict = torch.load(ckp, map_location='cpu')
            local_keys = set(model.state_dict().keys())
            local_state_dict = {
                key: value
                for key, value in full_state_dict.items()
                if key in local_keys
            }
            missing_keys = local_keys - set(local_state_dict.keys())
            if missing_keys:
                raise RuntimeError(
                    f'checkpoint缺少当前EP rank需要的参数: {sorted(missing_keys)[:5]}'
                )
            model.load_state_dict(local_state_dict, strict=True)
            del full_state_dict, local_state_dict
        else:
            model.load_state_dict(
                torch.load(ckp, map_location=args.device),
                strict=True,
            )
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_ep', action='store_true', help="启用Expert Parallel；会自动启用MoE，必须使用torchrun启动")
    parser.add_argument('--num_experts', default=4, type=int, help="全局Expert数量")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="每个Token选择的Expert数量（Top-K）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    rank, world_size = init_ep(args)
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n')) if rank == 0 else None
    input_mode = broadcast_object(input_mode, rank)
    streamer = (
        TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if rank == 0 else None
    )

    prompt_index = 0
    while True:
        if rank == 0:
            if input_mode == 0:
                prompt = prompts[prompt_index] if prompt_index < len(prompts) else ''
                prompt_index += 1
            else:
                prompt = input('💬: ')
        else:
            prompt = None
        prompt = broadcast_object(prompt, rank)
        if not prompt:
            break

        seed = random.randint(0, 31415926) if rank == 0 else None
        seed = broadcast_object(seed, rank)
        setup_seed(seed)
        if input_mode == 0 and rank == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        if rank == 0: print('🧠: ', end='')
        st = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
            )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if rank == 0:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
