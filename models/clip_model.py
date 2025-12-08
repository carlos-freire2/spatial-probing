import torch
from transformers import AutoImageProcessor, CLIPModel
from PIL import Image

class CLIPWrapper(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        self.model = clip_model.vision_model  # only vision encoder

    def forward(self, pixel_values, output_hidden_states):
        return self.model(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
    
    def get_hidden_states(self, x: torch.Tensor, device) -> list[torch.Tensor]:
        x = [Image.fromarray(im.numpy()) for im in x]
        inputs = self.processor(images=x, return_tensors="pt")

        self.model = self.model.to(device)
        self.model.eval()
        inputs = inputs.to(device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states
        return all_hidden_states
        