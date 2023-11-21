from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

MODELSCOPE_CAHCE='./models'

# Note: The default behavior now has injection attack prevention off.
model_dir = AutoModelForCausalLM.from_pretrained('qwen/Qwen-7B-Chat', device_map="auto", trust_remote_code=True, bf16=True).eval()
# model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# strs = "You must correctly predict the next {object} from a given contexts consisting of multiple quadruplets in the form of {time}: [{subject}, {relation}, {object}] and a query in the form of {time}: [{subject}, {relation}, ] given at the end. Please directly giving the answer.\n289: [Xi_Jinping, Make_statement, China]\n298: [Xi_Jinping, Make_statement, China]\n300: [Xi_Jinping, Make_statement, China]\n301: [Xi_Jinping, Make_statement, China]\n302: [Xi_Jinping, Make_statement, China]\n303: [Xi_Jinping, Make_statement, China]\n303: [Xi_Jinping, Make_statement, Michael_Sata]\n304: [Xi_Jinping, Make_statement, Michael_Sata]\n305: [Xi_Jinping, Make_statement, ]"
# response, history = model.chat(tokenizer, strs, history=None)
# print(response)