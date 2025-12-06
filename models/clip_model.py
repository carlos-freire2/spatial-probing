import torch
from transformers import AutoProcessor, CLIPVisionModel

class CLIPWrapper(torch.nn.Module):
    def __init__(self, model_name: str = ""):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)

    def forward(self, pixel_values, output_hidden_states):
        return self.model(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
        
        