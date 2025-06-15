import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_model import Classifier

class PreprocessModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # Change image format
        x = self.resize(x)
        x = x / 255.0
        x = self.normalize(x)
        return x

def convert_to_onnx() -> None:
    model: Classifier = Classifier()
    model.load_state_dict(torch.load("pytorch_model_weights.pth", map_location=torch.device("cpu")))
    model.eval()
    preprocess: PreprocessModule = PreprocessModule()
    class CombinedModel(nn.Module):
        def __init__(self, preprocess: PreprocessModule, model: Classifier) -> None:
            super().__init__()
            self.preprocess = preprocess
            self.model = model
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.preprocess(x)
            return self.model(x)
    combined_model: CombinedModel = CombinedModel(preprocess, model)
    dummy_input: torch.Tensor = torch.randn(1, 224, 224, 3)
    torch.onnx.export(
        combined_model,
        dummy_input,
        "image_classifier.onnx",
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        do_constant_folding=True
    )
    print("Model saved as image_classifier.onnx")

if __name__ == "__main__":
    convert_to_onnx()