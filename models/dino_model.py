import torch
from transformers import AutoImageProcessor, AutoModel

class DINOv3Wrapper(torch.nn.Module):
    def __init__(self, model_name: str, token=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, token=token)

    def get_hidden_states(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        inputs = self.processor(images=x, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        all_hidden_states = outputs.hidden_states
        return all_hidden_states 
