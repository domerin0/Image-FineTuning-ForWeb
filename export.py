import onnx
import torch
import onnxruntime as ort
import numpy as np

batch_size = 1
image_size = 224
model_name = "resnet18.onnx"

model = torch.load('./best.pt')
model.eval()

x = torch.randn(batch_size, 3, image_size, image_size, requires_grad=False)
torch_out = model(x)

# Export the model
torch.onnx.export(model,
                  x,
                  model_name,
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}
                  })

onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(model_name)
outputs = ort_sess.run(None, {'input': x.numpy()})
print(outputs)
