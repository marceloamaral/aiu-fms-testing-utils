import os
import time
import platform
import numpy as np
import pandas as pd
import torch
from torch.profiler import profile, schedule, ProfilerActivity
from torch_sendnn import torch_sendnn  # noqa: F401

def replace_names(table, prof):
    """Make profiler table more readable by renaming AIU/CUDA events."""
    table = table.replace("CUDA", 
                          "AIU")
    table = table.replace("Torch-Compiled Region: 0/0", 
                          "                      TTFT")
    table = table.replace("Torch-Compiled Region: 0/1", 
                          "                       ITL")
    table = table.replace("aiuLaunchSuperNode", 
                          "        aiuRuntime")
    table = table.replace("aiuScheduleWait",
                          "    aiuHardware")

    # for evt in prof.key_averages():
    #     if "_sendnn_super_node_op" in evt.key:
    #         table = table.replace(evt.key, "aiuSendnnRoundTrip")
    #         break

    return table


def isolate_prefill_events(prof, root_event_prefix="Torch-Compiled Region: 0/0"):
    """
    Identify prefilling events as those that happen before the first major sub-block
    (first token) inside the first iteration.
    """
    events = list(prof.events())

    def get_start(e):
        tr = getattr(e, "time_range", None)
        return getattr(tr, "start", 0) if tr else 0

    def get_end(e):
        tr = getattr(e, "time_range", None)
        return getattr(tr, "end", 0) if tr else 0

    # Sort by start time
    events.sort(key=get_start)

    prefill_events, decode_events = [], []

    # Identify iteration-level events
    iteration_events = [e for e in events if e.name.startswith(root_event_prefix)]
    if not iteration_events:
        print("No iteration markers found; cannot isolate prefill.")
        return prefill_events, events  # fallback

    # First iteration as reference
    iter_end = get_end(iteration_events[0])

    # eps = 1.005 # 0.5%
    # split_time = iter_end * eps
    split_time = iter_end 
    prefill_events = [e for e in events if get_end(e) <= split_time]
    decode_events = [e for e in events if get_start(e) > split_time]
    return prefill_events, decode_events


def create_row(events):
    events_by_name = {}
    for e in events:
        events_by_name.setdefault(e.name, []).append(e)

    n2m = 1000 # nanoseconds to miliseconds
    rows = []
    for name, events in events_by_name.items():
        total_cpu = sum(e.cpu_time/n2m for e in events)
        self_cpu = sum(e.self_cpu_time_total/n2m for e in events)
        self_aiu = sum(e.self_device_time_total/n2m for e in events)
        total_aiu = sum(e.device_time_total/n2m for e in events)

        cpu_mem = sum(e.cpu_memory_usage for e in events)
        self_cpu_mem = sum(e.self_cpu_memory_usage for e in events)
        aiu_mem = sum(e.device_memory_usage for e in events)
        self_aiu_mem = sum(e.self_cuda_memory_usage for e in events)

        n_calls = sum(e.count for e in events)
        cpu_times = [e.cpu_time/n2m for e in events]
        aiu_times = [e.device_time/n2m for e in events]

        rows.append({
            "Name": name,
            "Self CPU": self_cpu,
            "CPU total": total_cpu,
            "Self AIU": self_aiu,
            "AIU total": total_aiu,
            "CPU time median": np.median(cpu_times),
            "CPU time max": np.max(cpu_times),
            "AIU time median": np.median(aiu_times),
            "AIU time max": np.max(aiu_times),
            "CPU Mem": cpu_mem,
            "Self CPU Mem": self_cpu_mem,
            "AIU Mem": aiu_mem,
            "Self AIU Mem": self_aiu_mem,
            "# of Calls": n_calls,
        })

    # Compute percentages
    total_self_cpu = max(1, sum(r["Self CPU"] for r in rows))
    total_cpu = max(1, sum(r["CPU total"] for r in rows))
    total_self_aiu = max(1, sum(r["Self AIU"] for r in rows))
    total_aiu = max(1, sum(r["AIU total"] for r in rows))
    for r in rows:
        r["Self CPU %"] = r["Self CPU"] / total_self_cpu * 100
        r["CPU total %"] = r["CPU total"] / total_cpu * 100
        r["Self AIU %"] = r["Self AIU"] / total_self_aiu * 100
        r["AIU total %"] = r["AIU total"] / total_aiu * 100

    return rows


def summarize_events(prof, isolate_prefill=True):
    """Aggregate raw events for more detailed analysis."""
    prefill_events = None
    decode_events = None
    prefill_rows = None
    decode_rows = None
    
    if isolate_prefill:
        prefill_events, decode_events = isolate_prefill_events(prof, root_event_prefix="Torch-Compiled Region: 0/0")
    else:
        decode_events = list(prof.events())
    
    if isolate_prefill:
        prefill_rows = create_row(prefill_events)
    decode_rows = create_row(decode_events)

    return prefill_rows, decode_rows


def create_table(rows, table_prefix="", sort_field="CPU time max", filter_field=False, row_limit=30):
    """Return a string table (as text) of aggregated profiler data."""
    df = pd.DataFrame(rows)
    
    # filter columns
    if filter_field:
        if "CPU" in sort_field:
            cols = [col for col in df.columns if "AIU" not in col]
            df = df[cols]
        elif "AIU" in sort_field:
            cols = [col for col in df.columns if "CPU" not in col]
            df = df[cols]
    
    if sort_field in df.columns:
        df = df.sort_values(by=sort_field, ascending=False)
    df = df.head(row_limit)

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].round(3)

    output = []
    output.append("=" * 100)
    output.append(f"{table_prefix} Table sorted by: {sort_field}")
    output.append("=" * 100)
    output.append(df.to_string(index=False))
    output.append("=" * 100 + "\n")

    return "\n".join(output)

def save_and_print_table(cpu_file, aiu_file, rows, prof, table_prefix, print_table=False):
        cpu_table = create_table(rows, table_prefix=table_prefix, sort_field="CPU time median", row_limit=100)
        aiu_table = create_table(rows, table_prefix=table_prefix, sort_field="AIU time median", row_limit=100)

        with open(cpu_file, "w") as f:
            f.write(replace_names(cpu_table, prof))
        with open(aiu_file, "w") as f:
            f.write(replace_names(aiu_table, prof))

        if print_table:
            cpu_table = create_table(rows, table_prefix=table_prefix, sort_field="CPU time median", filter_field=True, row_limit=20)
            print(replace_names(cpu_table, prof))

            aiu_table = create_table(rows, table_prefix=table_prefix, sort_field="AIU time median", filter_field=True, row_limit=10)
            print(replace_names(aiu_table, prof))

def make_trace_handler(name_prefix, torch_profiling_dir, rank, print_summary=True):
    """Return a custom handler function for profiler traces."""
    def handler(prof):
        dir_path = f"{torch_profiling_dir}/{name_prefix}"
        os.makedirs(dir_path, exist_ok=True)
        host_name = platform.node()

        prof.export_chrome_trace(f"{dir_path}/{host_name}-rank{rank}_{int(time.time())}.json")

        cpu_file_prefill = f"{dir_path}/profile_table_summary_cpu_prefill-rank{rank}-{prof.step_num}.txt"
        aiu_file_prefill = f"{dir_path}/profile_table_summary_aiu_prefill-rank{rank}-{prof.step_num}.txt"
        cpu_file_decode = f"{dir_path}/profile_table_summary_cpu_decode-rank{rank}-{prof.step_num}.txt"
        aiu_file_decode = f"{dir_path}/profile_table_summary_aiu_decode-rank{rank}-{prof.step_num}.txt"

        prefill_rows, decode_rows = summarize_events(prof)

        print_table = str(rank) == "0" and print_summary
        save_and_print_table(cpu_file_prefill, aiu_file_prefill, prefill_rows, prof, table_prefix=f"RANK-{rank} Prefill (first token)", print_table=print_table)
        save_and_print_table(cpu_file_decode, aiu_file_decode, decode_rows, prof, table_prefix=f"RANK-{rank} Decode (subsequent tokens)", print_table=print_table)

    return handler


class aiu_profile:
    """Context manager to profile a code block using AIU backend."""

    def __init__(self, name_prefix, iters=1):
        self.name_prefix = name_prefix
        self.rank = os.getenv("LOCAL_RANK", "0")

        self.is_prof_enabled = os.getenv("ENABLE_TORCH_PROFILER", "0") in ("1", "true", "True", "yes", "on")
        self.torch_profiling_dir = os.getenv("TORCH_PROFILER_DIR", "./torch_traces")
        self.print_summary = os.getenv("TORCH_PROFILER_PRINT_SUMMARY", "0") in ("1", "true", "True", "yes", "on")

        # Profiler feature flags
        self.is_prof_with_stack_enabled = os.getenv("TORCH_PROFILER_WITH_STACK", "1").lower() in ("1", "true", "yes", "on")
        self.is_prof_with_memory_enabled = os.getenv("TORCH_PROFILER_WITH_MEMORY", "0").lower() in ("1", "true", "yes", "on")
        self.is_prof_with_tensor_shape_enabled = os.getenv("TORCH_PROFILER_WITH_TENSODR_SHAPE", "0").lower() in ("1", "true", "yes", "on")

        # Repeat steps
        self.repeat_steps = int(os.getenv("TORCH_PROFILER_SCHEDULE_REPEAT_STEPS", "1"))

        num_steps = iters

        # ---------------- skip_first ----------------
        self.is_skip_first_enabled = os.getenv("TORCH_PROFILER_SCHEDULE_SKIP_FIRST_STEP_ENABLED", "1").lower() in ("1", "true", "yes", "on")
        self.skip_first = 1 if self.is_skip_first_enabled and num_steps > 1 else 0
        if self.skip_first > 0:
            num_steps -= self.skip_first

        # ---------------- wait_steps ----------------
        self.wait_steps = max(1, num_steps // 10) # ~10% of remaning steps
        wait_enabled = os.getenv("TORCH_PROFILER_SCHEDULE_WAIT_STEPS_ENABLED", "1").lower() in ("1", "true", "yes", "on")
        if not wait_enabled or (num_steps - self.wait_steps) <= 0:
            self.wait_steps = 0
        else:
            num_steps -= self.wait_steps

        # ---------------- warmup_steps ----------------
        self.warmup_steps = max(1, num_steps // 10) # ~10% of remaning steps
        warmup_enabled = os.getenv("TORCH_PROFILER_SCHEDULE_WARMUP_STEPS_ENABLED", "1").lower() in ("1", "true", "yes", "on")
        if not warmup_enabled or (num_steps - self.warmup_steps) <= 0:
            self.warmup_steps = 0
        else:
            num_steps -= self.warmup_steps

        # ---------------- sanity check ----------------
        if num_steps == 0:
            if warmup_enabled and self.warmup_steps > 0:
                self.warmup_steps -= 1
                num_steps += 1
            elif wait_enabled and self.wait_steps > 0:
                self.wait_steps -= 1
                num_steps += 1
            elif self.is_skip_first_enabled and self.skip_first > 0:
                self.skip_first -= 1
                num_steps += 1

        # ---------------- active_steps ----------------
        self.active_steps = max(1, num_steps)

    def __enter__(self):
        # Rename PrivateUse1 backend safely if it was renamed yet
        try:
            torch.utils.rename_privateuse1_backend("aiu")
        except RuntimeError:
            pass

        # Register AIU backend safely if it was registered yet
        try:
            torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
        except RuntimeError as e:
            if "already been registered" not in str(e):
                raise

        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
            schedule=schedule(
                skip_first=self.skip_first,
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=self.repeat_steps
            ),
            record_shapes=self.is_prof_with_tensor_shape_enabled,
            with_stack=self.is_prof_with_stack_enabled,
            profile_memory=self.is_prof_with_memory_enabled,
            on_trace_ready=make_trace_handler(self.name_prefix, self.torch_profiling_dir, self.rank, self.print_summary),
        )
        if self.is_prof_enabled:
            self.prof.__enter__()
        return self.prof

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_prof_enabled:
            self.prof.__exit__(exc_type, exc_value, traceback)
