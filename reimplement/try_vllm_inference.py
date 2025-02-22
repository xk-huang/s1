import dotenv
dotenv.load_dotenv()

from vllm import LLM, SamplingParams
from typing import Optional, Sequence
import time
from transformers import AutoTokenizer, HfArgumentParser
import os

def _qwen_forward(
    prompts: Sequence[str],
    model_name: str,
    tokenizer_path: str,
    max_length: int = 32768,
    temperature: float = 0.05,
) -> Optional[Sequence[str]]:
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_length)
    if "7B" in model_name:
        tensor_parallel_size = 1
    else:
        tensor_parallel_size = 2
    model = None
    while model is None:
        try:
            model = LLM(model=model_name,
                        tokenizer=tokenizer_path,
                        tensor_parallel_size=tensor_parallel_size)
        except Exception as e:
            print(f"Error loading model: {e}")
            time.sleep(10)
    outputs = model.generate(prompts=prompts,
                             sampling_params=sampling_params)
    result = []
    for output in outputs:
        result.append(output.outputs[0].text)
    return result
    
def main():
    model_name = "Qwen/Qwen2.5-32B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(os.getenv("HF_HOME"))

    prompt = "A 66-year-old male presents to his primary care physician to discuss his increasing shortness of breathover the last 3 months. He notes that this is particularly obvious when he is mowing his lawn or climbing the stairs in his home. His past medical history is significant for hypertension that is well-controlled with lisinopril. His vital signs are as follows: T 37.6 C, HR 88, BP 136/58, RR 18, SpO2 97% RA. Physical examination is significant for an early diastolic blowing, decrescendo murmur heard best at the left sternal border, a midsystolic murmur heard best at the right upper sternal border, and a late diastolic rumbling murmur heard best at the apex on auscultation. In addition, an S3 heart sound is also present. Bounding pulses are palpated at the radial arteries bilaterally. Which of the following diagnoses is most likely in this patient? A. Mitral regurgitation B. Aortic regurgitation C. Aortic stenosis D. Mitral prolapse"
    dialog = [{"role" : "user", "content":prompt}]
    prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
    prompts = [prompt]
    results = _qwen_forward(prompts, model_name, model_name)

    # fmt: off
    import IPython; IPython.embed()
    # fmt: on


if __name__ == "__main__":
    main()