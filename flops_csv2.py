#!/usr/bin/env python3
import time
import torch
import timm
import numpy as np
import gc
from jtop import jtop
import threading
import queue
from ptflops import get_model_complexity_info  
import torchinfo  

LOOPS = 500
USE_FP16 = True
IMAGE_SIZE = 224
MAX_BS_CAP = 8
PROBLEM_MODELS = ['edgenext_small', 'convit_tiny','levit_384','tnt_s_patch16_224','edgenext_x_smal']

def get_gpu_free_mem(jetson):
    """Lấy free memory GPU (bytes) từ jtop, dùng 'RAM' cho Jetson Nano và tính available mem."""
    try:
        ram = jetson.memory['RAM']
        # Tính available mem: free + buffers + cached (KB)
        free_kb = ram.get('free', 0)
        buffers_kb = ram.get('buffers', 0)
        cached_kb = ram.get('cached', 0)
        available_kb = free_kb + buffers_kb + cached_kb
        return int(available_kb * 1024) # Chuyển sang bytes
    except KeyError:
        print("Lỗi lấy RAM từ jtop. Giả sử free_mem 2GB.")
        return 2 * 1024 * 1024 * 1024 # Fallback 2GB bytes

def find_max_bs(model, device, jetson, use_fp16):
    bs = 1
    max_bs = 1
    step = 1
    dtype_size = 2 if use_fp16 else 4 # bytes per float (fp16=2, fp32=4)
    while True:
        free_mem = get_gpu_free_mem(jetson)
        input_mem_est = bs * 3 * IMAGE_SIZE * IMAGE_SIZE * dtype_size
        overhead_est = input_mem_est * 4 # Tăng lên 4x cho conservative hơn
        if input_mem_est + overhead_est > free_mem * 0.7:
            print(f"Bỏ qua bs={bs} vì ước tính vượt bộ nhớ (free: {free_mem / 1e6:.0f} MB, est: {(input_mem_est + overhead_est)/1e6:.0f} MB)")
            break
        if bs > MAX_BS_CAP:
            print(f"Đạt giới hạn cap bs={MAX_BS_CAP}")
            break
        
        print(f"Thử batch size: {bs} (free mem: {free_mem / 1e6:.0f} MB)")
        try:
            x = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            if use_fp16:
                x = x.half()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            max_bs = bs
            bs += step
            step *= 2
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM at bs={bs}")
                break
            else:
                raise e
    
    # Backtrack binary fine
    low = max_bs // 2
    high = max_bs
    while low <= high:
        mid = (low + high) // 2
        free_mem = get_gpu_free_mem(jetson)
        input_mem_est = mid * 3 * IMAGE_SIZE * IMAGE_SIZE * dtype_size
        overhead_est = input_mem_est * 3 # Tăng conservative
        if input_mem_est + overhead_est > free_mem * 0.8:
            print(f"Bỏ qua backtrack mid={mid} vì ước tính vượt bộ nhớ")
            high = mid - 1
            continue
        
        print(f"Backtrack thử batch size: {mid} (free mem: {free_mem / 1e6:.0f} MB)")
        try:
            x = torch.randn(mid, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            if use_fp16:
                x = x.half()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            max_bs = mid
            low = mid + 1
        except RuntimeError:
            high = mid - 1
    return max_bs

def run_benchmark(model_name, device, jetson):
    print(f"--- Configuration for {model_name.replace('resnet', 'resnet ')} ---")
    print(f"Model: {model_name.replace('resnet', 'resnet ')}, Device: {device}")
    print(f"Loops: {LOOPS}\n")
    
    use_fp16 = USE_FP16 and model_name not in PROBLEM_MODELS
    
    model = timm.create_model(model_name, pretrained=True).eval()  # Tạo model trên CPU trước
    
    # Tính params_ptflops, macs_ptflops dùng ptflops với as_strings=False để lấy số đầy đủ
    macs_pt_num, params_pt_num = get_model_complexity_info(model, (3, IMAGE_SIZE, IMAGE_SIZE),
                                                           as_strings=False,
                                                           print_per_layer_stat=False)
    
    # Tính params_tinfo, macs_tinfo dùng torchinfo
    stats = torchinfo.summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), verbose=0)
    params_tinfo = stats.total_params
    macs_tinfo = stats.total_mult_adds  #macs
    
    # Tính activations dùng hook (tổng numel output mỗi layer)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    total_activations = [0]
    
    def count_activations(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                num_elements = sum(o.numel() for o in output if isinstance(o, torch.Tensor))
            else:
                num_elements = output.numel()
            total_activations[0] += num_elements
        return hook
    
    hooks = []
    try:
        model = model.to('cpu')  # Đảm bảo model trên CPU để tránh lỗi device mismatch
        dummy_input = dummy_input.to('cpu')
        for name, layer in model.named_modules():
            hooks.append(layer.register_forward_hook(count_activations(name)))
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"Lỗi tính activations: {e}")
    finally:
        for h in hooks:
            h.remove()
    
    activations = total_activations[0]
    
    # In các chỉ số mới (tương tự GitHub), với số đầy đủ cho ptflops
    print(f"Params (torchinfo): {params_tinfo}")
    print(f"MACs (torchinfo): {macs_tinfo}")
    print(f"Params (ptflops): {params_pt_num}")
    print(f"MACs (ptflops): {macs_pt_num}")
    print(f"Activations: {activations}\n")
    
    # Chuyển model sang device và half nếu cần (sau tính toán)
    model = model.to(device)
    if use_fp16:
        model = model.half()
    
    # Tính max_bs
    print("Finding max batch size...")
    max_bs = find_max_bs(model, device, jetson, use_fp16)
    print(f"Max BS found: {max_bs}")
    
    # Gen bs list: 1 đến max_bs với step hợp lý
    if max_bs > 8:
        step = max(1, max_bs // 8)
    else:
        step = 1
    bs_list = list(range(1, max_bs + 1, step))
    if bs_list[-1] != max_bs:
        bs_list.append(max_bs)
    print(f"BS list: {bs_list}")
    
    # Idle power: dùng 'power' cho tức thời, lấy nhiều mẫu
    idle_readings = {'TOT': [], 'CPU': [], 'GPU': []}
    for _ in range(20):
        power_stats = jetson.power
        idle_readings['TOT'].append(power_stats['tot']['power']) # mW tức thời
        idle_readings['CPU'].append(power_stats['rail']['POM_5V_CPU']['power'])
        idle_readings['GPU'].append(power_stats['rail']['POM_5V_GPU']['power'])
        time.sleep(0.1)
    idle_avg = {k: np.mean(v) for k, v in idle_readings.items()}
    
    energies = []
    throughputs = []
    latencies = [] # avg ns per batch
    delta_gpus = [] # Để lưu delta per bs
    total_bs = len(bs_list)
    for idx, bs in enumerate(bs_list, start=1):
        print(f"\nĐang xử lý batch size thứ {idx}/{total_bs}: {bs}")
        print(f"--- BS: {bs} ---")
        try:
            # Warm-up
            x = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            if use_fp16:
                x = x.half()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
                torch.cuda.synchronize()
            # Chuẩn bị queue và event cho thread
            power_queue = queue.Queue()
            stop_event = threading.Event()
            def power_sampler():
                while not stop_event.is_set():
                    power_stats = jetson.power
                    jtop_stats = jetson.stats
                    # jtop lưu GPU util ở key 'GPU' (int/float %), không phải 'GR3D' (tegrastats format)
                    gpu_util_raw = jtop_stats.get('GPU', 0)
                    if isinstance(gpu_util_raw, dict):
                        gpu_util = float(gpu_util_raw.get('val', 0))
                    else:
                        gpu_util = float(gpu_util_raw) if gpu_util_raw else 0.0
                    power_queue.put({
                        'TOT': power_stats['tot']['power'],
                        'CPU': power_stats['rail']['POM_5V_CPU']['power'],
                        'GPU': power_stats['rail']['POM_5V_GPU']['power'],
                        'GPU_util': gpu_util
                    })
                    time.sleep(0.1)  # 10Hz khớp với jtop refresh rate (~1Hz), tránh đọc giá trị cached lặp lại
            # Bắt đầu sampler thread
            sampler_thread = threading.Thread(target=power_sampler)
            sampler_thread.start()
            # Chạy inference loop
            timestamps = []
            start_time = time.time()
            with torch.no_grad():
                for _ in range(LOOPS):
                    loop_start = time.perf_counter()
                    _ = model(x)
                    torch.cuda.synchronize()
                    loop_end = time.perf_counter()
                    timestamps.append((loop_end - loop_start) * 1e9) # ns
            end_time = time.time()
            duration_s = end_time - start_time
            # Dừng sampler và lấy dữ liệu
            stop_event.set()
            sampler_thread.join()
            active_readings = {'TOT': [], 'CPU': [], 'GPU': [], 'GPU_util': []}
            while not power_queue.empty():
                reading = power_queue.get()
                for k in active_readings:
                    active_readings[k].append(reading[k])
            if not active_readings['GPU']:
                print("Không có mẫu power, dùng fallback 0")
                active_avg = idle_avg.copy()
                avg_gpu_util = 0
            else:
                active_avg = {k: np.mean(v) for k, v in active_readings.items() if k != 'GPU_util'}
                avg_gpu_util = np.mean(active_readings['GPU_util'])
            delta = {k: max(0, active_avg[k] - idle_avg[k]) for k in active_avg} # Tránh delta âm
            delta_gpus.append(delta['GPU'])
            # Energy: ưu tiên delta power nếu đủ lớn (> 100mW để vượt noise đo lường jtop)
            # Nếu delta quá nhỏ (jtop refresh chậm, readings bị cache), dùng tổng công suất active
            if delta['TOT'] >= 100:
                energy_power_mw = delta['TOT']
                energy_source = 'delta'
            else:
                energy_power_mw = active_avg['TOT']
                energy_source = 'active_total'
            gpu_energy_j = (energy_power_mw / 1000.0) * duration_s / (bs * LOOPS)
            gpu_energy_uj = gpu_energy_j * 1e6
            energies.append(gpu_energy_uj)
            avg_latency_ns = np.mean(timestamps)
            avg_latency_ns_per_image = avg_latency_ns / bs
            latencies.append(avg_latency_ns_per_image)
            throughput = (bs * LOOPS) / duration_s
            throughputs.append(throughput)
            print(f"Duration: {duration_s:.3f}s")
            print(f"Delta Power GPU: {delta['GPU']:.1f} mW")
            print(f"Energy GPU per image: {gpu_energy_uj:.8f} uJ [{energy_source}]")
            print(f"Avg Latency: {avg_latency_ns_per_image:.8f} ns/image")
            print(f"Throughput: {throughput:.8f} images/s")
            # Debug thêm
            print(f"Debug: Delta TOT: {delta['TOT']:.1f} mW, Delta CPU: {delta['CPU']:.1f} mW")
            print(f"Debug: Average GPU util: {avg_gpu_util:.1f}%")
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"Killed at BS={bs} due to OOM")
                break
            else:
                raise e
    
    # Tính optimal_bs: min energy
    if energies:
        min_energy_idx = np.argmin(energies)
        optimal_bs = bs_list[min_energy_idx]
        print(f"\n--- Summary for {model_name.replace('resnet', 'resnet ')} ---")
        print(f"Params (torchinfo): {params_tinfo}")
        print(f"MACs (torchinfo): {macs_tinfo}")
        print(f"Params (ptflops): {params_pt_num}")
        print(f"MACs (ptflops): {macs_pt_num}")
        print(f"Activations: {activations}")
        print(f"Optimal BS (min GPU energy): {optimal_bs}")
        print(f"Throughput at optimal: {throughputs[min_energy_idx]:.8f} images/s")
        print(f"Latency at optimal: {latencies[min_energy_idx]:.8f} ns/image")
        print(f"Energy at optimal: {energies[min_energy_idx]:.8f} uJ")
    else:
        print("No valid BS found")
    
    # Clean up model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Chạy đúng 1 model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = [ 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1',
        'mobilenetv2_100', 'mobilenetv3_small_100', 'convnext_tiny', 'mnasnet_small',
        'resnext50_32x4d', 'densenet121', 'inception_v3', 'inception_v4',
        'vit_tiny_patch16_224', 'deit_small_patch16_224', 'swin_tiny_patch4_window7_224',
        'gmlp_ti16_224', 'mixer_s32_224', 'edgenext_small', 'levit_128', 'levit_192',
        'convit_tiny', 'resmlp_12_224'
        'legacy_seresnet50'] + ['resnet26', 'resnet26d', 'resnet50d', 'resnet101', 'resnet152', 'resnest50d',
         'resnext101_32x8d', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_lite0', 
         'efficientnet_lite1', 'efficientnetv2_rw_t', 'efficientnetv2_rw_s', 'mobilenetv2_050', 'mobilenetv2_075', 
         'mobilenetv2_120d', 'mobilenetv3_large_100', 'mobilenetv3_small_075', 'mnasnet_050', 'mnasnet_075'
         'densenet169','densenet201','inception_resnet_v2','regnetx_002','regnetx_004','regnety_002',
         'regnety_004', 'cspresnet50','cspdarknet53']
    if args.model:
        model_names = [args.model]

    with jtop() as jetson:
        time.sleep(1.0)

        for i, model_name in enumerate(model_names):
            run_benchmark(model_name, device, jetson)
            
            if i < len(model_names) - 1:
                print(f"\nPausing for 5 minutes to cool down before next model...")
                time.sleep(300)
            
            # Extra clean giữa các mô hình
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
