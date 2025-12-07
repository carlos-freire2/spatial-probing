import torch
from transformers import AutoImageProcessor, AutoModel

class DINOv3Wrapper(torch.nn.Module):
    def __init__(self, model_name: str, token=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, token=token)

    def forward(self, pixel_values, output_hidden_states):
        return self.model(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
    
    def get_hidden_states(self, x: list[torch.Tensor], device) -> list[torch.Tensor]:
        inputs = self.processor(images=x, return_tensors="pt")
                
        self.model = self.model.to(device)
        self.model.eval()
        inputs = inputs.to(device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
        all_hidden_states = outputs.hidden_states
        return all_hidden_states
