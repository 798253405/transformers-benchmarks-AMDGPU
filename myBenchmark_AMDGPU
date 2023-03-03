import torch

print('Pytorch version\t:', torch.__version__)
print('CUDA version\t:', torch.version.cuda)
print('GPU\t\t:',torch.cuda.get_device_name())

import inspect
from collections import defaultdict
import pandas as pd
from torch.utils import benchmark 

pd.options.display.precision = 3

def var_dict(*args):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return dict([(name, val) for name, val in callers_local_vars if val is arg][0] 
                for arg in args)

def walltime(stmt, arg_dict, duration=3):
    return benchmark.Timer(stmt=stmt, globals=arg_dict).blocked_autorange(
        min_run_time=duration).median

matmul_tflops = defaultdict(lambda: {})
for n in [128, 512, 2048, 8192]:
    for dtype in (torch.float32, torch.float16):
        a = torch.randn(n, n, dtype=dtype).cuda()
        b = torch.randn(n, n, dtype=dtype).cuda()   
        t = walltime('a @ b', var_dict(a, b))
        matmul_tflops[f'n={n}'][dtype] = 2*n**3 / t / 1e12
        del a, b
        
print(pd.DataFrame(matmul_tflops), ' Matrix Multiplication ')
vector = defaultdict(lambda: {})
for n in [1024*64, 1024*256, 1024*1024, 1024*1024*4]:
    a = torch.randn(n).cuda()
    t = walltime('a * 1.2', var_dict(a))
    vector[n]['TFLOPS'] = n / t / 1e12
    vector[n]['GB/s'] = 8 * n / t / 1e9
    
print(pd.DataFrame(vector),' Vector Multiplication ')

#%%
from transformers import AutoConfig, BertLayer
config = AutoConfig.from_pretrained("bert-large-uncased")
layer = BertLayer(config).half().cuda()

#%%
def layer_benchmark(layer, hidden_size, seq_lens, batch_sizes, cross_attention=False):
    h = hidden_size
    results = defaultdict(lambda: {})    
    encoder_state = 'encoder_hidden_states=X' if cross_attention else ''
    for s in seq_lens:
        for b in batch_sizes:            
            ffn = 16*b*s*h*h / 1e12  # TFLOPS for the Feed-Forward Network
            atten = (4*b*h*s*s + 8*b*s*h*h) / 1e12  # TFLOPS for attention            
            forward = ffn + (2 if cross_attention else 1) * atten
            
            X = torch.randn(b, s, h).half().cuda()
            results[f'batch={b}'][f'fwd seq_len={s}'] = forward / walltime(
                f'layer(X, {encoder_state})', var_dict(layer, X))
            results[f'batch={b}'][f'fwd+bwd seq_len={s}'] = 3 * forward / walltime(
                f'layer(X, {encoder_state})[0].sum().backward()', var_dict(layer, X))            
    return pd.DataFrame(results)

print(layer_benchmark(layer, config.hidden_size, [128, 512], [2, 4, 8, 16, 32, 64, 128]), ' bert benchmark')
print('bert done')
#%%
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
config = AutoConfig.from_pretrained("gpt2-medium")
layer = GPT2Block(config, layer_idx=0).half().cuda()
print(layer_benchmark(layer, config.n_embd, [512, 1024], [2, 4, 8, 16, 32, 64]),' result gpt2 ')
print('layer bench mark done')

#%%t5
from transformers.models.t5.modeling_t5 import T5Block

config = AutoConfig.from_pretrained("t5-large")
config.use_cache = False
config.is_decoder = False
config.is_encoder_decoder = False

encoder = T5Block(config).half().cuda()
print(layer_benchmark(encoder, config.d_model, [512], [2, 4, 8, 16, 32, 64, 128]),'t5 encoder')

config.is_decoder = True
decoder = T5Block(config).half().cuda()
print(layer_benchmark(decoder, config.d_model, [512], [2, 4, 8, 16, 32, 64, 128], cross_attention=True),'t5 decoder')
