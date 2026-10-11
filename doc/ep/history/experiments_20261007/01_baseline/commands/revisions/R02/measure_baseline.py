"""Measure the unchanged EP model; profiling and replay capture run separately."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import textwrap
import time
import types

import torch
import torch.distributed as dist
from transformers import AutoTokenizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    ordered = sorted(values)
    return dict(samples=len(values), median_ms=statistics.median(values),
                mean_ms=statistics.mean(values), min_ms=ordered[0],
                max_ms=ordered[-1], p95_ms=ordered[int(.95 * (len(values)-1))])


class Probe:
    def __init__(self, output, rank):
        self.output, self.rank = output, rank
        self.profile = False
        self.capture = False
        self.save_replay = False
        self.rows = []
        self.replays = []

    def region(self, name, layer):
        if self.profile:
            return torch.profiler.record_function(f"baseline/L{layer}/{name}")
        return contextlib.nullcontext()

    def collect(self, layer, x, counts, y, module):
        if not self.capture:
            return
        self.rows.append(dict(layer=layer, rank=self.rank, counts=counts,
                              total_m=sum(counts), input_shape=list(x.shape),
                              input_dtype=str(x.dtype), input_stride=list(x.stride()),
                              output_dtype=str(y.dtype), output_shape=list(y.shape),
                              local_expert_ids=list(module.local_expert_ids),
                              weight_dtype=str(next(module.experts.parameters()).dtype)))
        if self.save_replay and layer in (0, 3, 7):
            # These copies are made only in a separate capture pass.
            self.replays.append(dict(layer=layer, rank=self.rank, counts=counts,
                                     local_expert_ids=list(module.local_expert_ids),
                                     x=x.detach().cpu(), y=y.detach().cpu()))


def instrument_ep_forward(model_module):
    """Wrap original AST statements without rewriting their operations/order."""
    original = model_module.ExpertParallelMOEFeedForward.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    # Source snapshot starts at line 337; offsets below identify original statements.
    boundaries = [(0, "router"), (29, "rank_pack"), (40, "count_exchange"),
                  (46, "split_to_cpu"), (49, "dispatch"), (70, "expert_sort"),
                  (75, "expert_counts_to_cpu"), (80, "experts"),
                  (97, "inverse_expert"), (104, "combine"), (112, "restore")]
    groups = {label: [] for _, label in boundaries}
    for node in fn.body:
        offset = node.lineno - 1
        label = next(label for start, label in reversed(boundaries) if offset >= start)
        groups[label].append(node)
    assert isinstance(groups["split_to_cpu"][0], ast.Assign)
    assert groups["split_to_cpu"][0].targets[0].id == "send_splits"
    assert groups["expert_counts_to_cpu"][0].targets[0].id == "local_expert_counts"
    assert groups["experts"][0].targets[0].id == "expert_outputs"
    assert groups["combine"][0].targets[0].id == "returned_sorted_output"
    fn.body = []
    for _, label in boundaries:
        expression = ast.parse(
            f"probe.region('{label}', self._baseline_layer)", mode="eval").body
        fn.body.append(ast.With(items=[ast.withitem(expression, None)], body=groups[label]))
        if label == "experts":
            fn.body += ast.parse(
                "probe.collect(self._baseline_layer, expert_sorted_x, "
                "local_expert_counts, expert_sorted_output, self)").body
    fn.decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<baseline_ep_annotations>", "exec"), namespace)
    return original, namespace["forward"], namespace


def instrument_ff(module, probe, layer):
    def forward(self, x):
        with probe.region("gate_linear", layer):
            gate = self.gate_proj(x)
        with probe.region("activation", layer):
            activated = self.act_fn(gate)
        with probe.region("up_linear", layer):
            up = self.up_proj(x)
        with probe.region("activation", layer):
            hidden = activated * up
        with probe.region("down_linear", layer):
            return self.down_proj(hidden)
    return types.MethodType(forward, module)


def measured_calls(fn, warmups, steps, device):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize(device)
    wall, gpu = [], []
    for _ in range(steps):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        wall.append((time.perf_counter()-t)*1000)
        gpu.append(start.elapsed_time(end))
    return dict(wall=summary(wall), cuda_interval=summary(gpu),
                wall_samples_ms=wall, cuda_samples_ms=gpu)


def profiler_summary(profiler, repetitions):
    rows = {}
    for event in profiler.events():
        if event.name.startswith("baseline/L"):
            name = event.name.split("/", 2)[-1]
            row = rows.setdefault(name, dict(calls=0, cpu_us=0., device_us=0.))
            row["calls"] += 1
            row["cpu_us"] += event.cpu_time_total
            row["device_us"] += event.device_time_total
    for row in rows.values():
        row["cpu_ms_per_forward"] = row.pop("cpu_us") / repetitions / 1000
        row["device_ms_per_forward"] = row.pop("device_us") / repetitions / 1000
    return rows


def generate_inputs(repo, rank, batch, length, variation):
    tokenizer = AutoTokenizer.from_pretrained(str(repo / "model"), local_files_only=True)
    pool = []
    used_rows = 0
    with (repo / "dataset/pretrain_t2t_mini.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if index >= 512:
                break
            tokens = tokenizer(json.loads(line)["text"], add_special_tokens=False).input_ids
            pool += [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
            used_rows += 1
    assert len(pool) > 10000
    offset = (rank * 19391 + variation * 8191) % len(pool)
    values = [pool[(offset+i) % len(pool)] for i in range(batch * length)]
    return torch.tensor(values, dtype=torch.long).reshape(batch, length), dict(
        data_source=str(repo / "dataset/pretrain_t2t_mini.jsonl"),
        source_rows=used_rows, token_pool_size=len(pool), start_offset=offset,
        construction="pack consecutive corpus tokens with BOS/EOS; no padding",
        input_sha256=hashlib.sha256(bytes(str(values), "utf-8")).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(Path(__file__).with_name("model_minimind_snapshot.py")))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--workloads", default="prefill_small,prefill_medium,prefill_large,decode_single,decode_batch")
    parser.add_argument("--topks", default="1,2")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=device)
    assert dist.get_world_size() == 2
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo)
    spec = importlib.util.spec_from_file_location("baseline_minimind", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.MiniMindConfig(hidden_size=768, num_hidden_layers=8,
                                   use_moe=True, use_ep=True, ep_size=2,
                                   num_experts=4, num_experts_per_tok=1,
                                   moe_intermediate_size=2432, dropout=0.,
                                   ep_comm_dtype="native", max_position_embeddings=4096)
    model = module.MiniMindForCausalLM(config)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key: weights[key] for key in model.state_dict()}
    model.load_state_dict(state, strict=True)
    del weights, state
    model = model.to(device).eval()
    # FP32 parameters with BF16 autocast match the project's training precision policy.
    dtype = torch.bfloat16
    ep_original, ep_annotated, namespace = instrument_ep_forward(module)
    probe = Probe(root, rank)
    namespace["probe"] = probe
    # exec copied the globals mapping; update the function's actual globals as well.
    ep_annotated.__globals__["probe"] = probe
    ff_original = module.FeedForward.forward
    for i, layer in enumerate(model.model.layers):
        layer.mlp._baseline_layer = i

    def annotate(enabled):
        module.ExpertParallelMOEFeedForward.forward = ep_annotated if enabled else ep_original
        for i, layer in enumerate(model.model.layers):
            for expert in layer.mlp.experts.values():
                expert.forward = instrument_ff(expert, probe, i) if enabled else types.MethodType(ff_original, expert)

    info = dict(rank=rank, device=str(device), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu=torch.cuda.get_device_name(device), gpu_properties=str(torch.cuda.get_device_properties(device)),
                python=sys.version, executable=sys.executable, torch=torch.__version__,
                cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                nccl=torch.cuda.nccl.version(), platform=platform.platform(),
                source_sha256=sha256(args.source), checkpoint=args.checkpoint,
                checkpoint_sha256=sha256(args.checkpoint), config=config.to_dict(),
                parameter_dtype=str(next(model.parameters()).dtype), autocast_dtype=str(dtype),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                torch_threads=torch.get_num_threads(), warmups=args.warmups, steps=args.steps,
                profile_steps=args.profile_steps, timing="unannotated original forward; barrier outside wall timing; synchronize at end",
                logits_to_keep=0, cache_policy="prefill builds cache; decode uses a fixed 128-token KV snapshot per call",
                caveat="Top-2 is a routing experiment on the Top-1 checkpoint, not a trained Top-2 model")
    (root / f"environment_rank{rank}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    all_results = []
    workloads = dict(prefill_small=(1, 128, 0), prefill_medium=(8, 340, 0),
                     prefill_large=(32, 340, 0), decode_single=(1, 1, 128),
                     decode_batch=(16, 1, 128))
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for topk in map(int, args.topks.split(",")):
            config.num_experts_per_tok = topk
            for name in args.workloads.split(","):
                batch, length, context = workloads[name]
                case = f"{name}_top{topk}"
                case_root = root / case
                case_root.mkdir(exist_ok=True)
                annotate(False)
                cpu_input, input_info = generate_inputs(repo, rank, batch, context or length, topk)
                inputs = cpu_input.to(device)
                cache = None
                if context:
                    initial = model(inputs, use_cache=True)
                    cache = initial.past_key_values
                    inputs = initial.logits[:, -1].argmax(-1, keepdim=True)
                    del initial

                def run():
                    return model(inputs, past_key_values=cache, use_cache=True)

                e2e = measured_calls(run, args.warmups, args.steps, device)
                reference = run()
                ref_logits = reference.logits.detach().clone()
                ref_hidden = reference.hidden_states.detach().clone()
                annotate(True)
                probe.profile = False
                probe.capture = True
                probe.save_replay = name in ("prefill_small", "prefill_medium", "decode_batch")
                probe.rows, probe.replays = [], []
                captured = run()
                equal = torch.equal(ref_logits, captured.logits) and torch.equal(ref_hidden, captured.hidden_states)
                max_error = (ref_logits.float()-captured.logits.float()).abs().max().item()
                if not equal:
                    raise RuntimeError(f"Annotation changed outputs: {case}, {max_error}")
                probe.capture = False
                torch.save(dict(input_ids=inputs.cpu(), context_input_ids=cpu_input if context else None,
                                checkpoint=args.checkpoint, checkpoint_sha256=info["checkpoint_sha256"],
                                replays=probe.replays, counts=probe.rows),
                           case_root / f"capture_rank{rank}.pt")
                (case_root / f"counts_rank{rank}.json").write_text(json.dumps(probe.rows, indent=2))

                # Timings below exercise complete original expert MLPs on captured inputs.
                # Replay CPU tensors are transferred before timing; weights remain FP32.
                annotate(False)
                micro = []
                for replay in probe.replays:
                    layer = replay["layer"]
                    mlp = model.model.layers[layer].mlp
                    x = replay["x"].to(device)
                    counts = replay["counts"]

                    def expert_mlp():
                        values, offset = [], 0
                        for expert_id, count in zip(mlp.local_expert_ids, counts):
                            if count:
                                values.append(mlp.experts[str(expert_id)](x[offset:offset+count]))
                            offset += count
                        return torch.cat(values) if values else x.clone()

                    out = expert_mlp()
                    expected = replay["y"].to(device)
                    assert torch.equal(out, expected)
                    micro.append(dict(layer=layer, counts=counts,
                                      timing=measured_calls(expert_mlp, 3, max(10, args.steps), device)))
                    del x, out, expected

                annotate(True)
                probe.profile = True
                for _ in range(2):
                    run()
                torch.cuda.synchronize(device)
                dist.barrier()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True) as prof:
                    for _ in range(args.profile_steps):
                        run()
                    torch.cuda.synchronize(device)
                prof.export_chrome_trace(str(case_root / f"trace_rank{rank}.json"))
                stages = profiler_summary(prof, args.profile_steps)
                kernels = sorted([dict(name=e.key, calls=e.count, self_device_us=e.self_device_time_total)
                                  for e in prof.key_averages() if e.self_device_time_total > 0],
                                 key=lambda x: x["self_device_us"], reverse=True)
                probe.profile = False
                record = dict(case=case, rank=rank, local_batch=batch, seq_len=length,
                              context_len=context, topk=topk, input=input_info, e2e=e2e,
                              annotation_equal=equal, max_logits_error=max_error,
                              counts=probe.rows, micro_mlp=micro, profiler_stages=stages,
                              profiler_kernels=kernels,
                              stage_caveat="diagnostic inclusive ranges from instrumented profiler; overlapping/nested times are not an end-to-end decomposition")
                (case_root / f"result_rank{rank}.json").write_text(json.dumps(record, indent=2))
                all_results.append(record)
                print(json.dumps(dict(case=case, rank=rank,
                                      median_wall_ms=e2e["wall"]["median_ms"], annotation_equal=equal)), flush=True)
                del reference, captured, ref_logits, ref_hidden, cache
                torch.cuda.empty_cache()
    (root / f"summary_rank{rank}.json").write_text(json.dumps(all_results, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
