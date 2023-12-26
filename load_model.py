from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

MODELSCOPE_CAHCE='./models'

model_dir = snapshot_download("Xorbits/vicuna-7b-v1.5", cache_dir='./models')