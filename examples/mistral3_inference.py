import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText  # noqa: E402

ckpt = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
model_name = ckpt
config = AutoConfig.from_pretrained(ckpt)
config.text_config.num_hidden_layers = 2
config.vision_config.num_hidden_layers = 2
config._attn_implementation = "eager"

model = QEFFAutoModelForImageTextToText.from_pretrained(model_name, config=config, kv_offload=True)

model.compile(num_devices=4, prefill_seq_len=3072, img_size=896, ctx_len=4096)

processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

img1_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"

messages_1 = [
    {
        "role": "user",
        "content": [{"type": "image", "url": img1_url}, {"type": "text", "text": "What animal is on the candy?"}],
    }
]

inputs = processor.apply_chat_template(
    messages_1, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)

inputs.pop("image_sizes")
streamer = TextStreamer(tokenizer)
output = model.generate(inputs=inputs, device_ids=[4, 5, 6, 7], generation_len=50)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
