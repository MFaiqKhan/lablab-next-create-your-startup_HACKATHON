import onnx
model = onnx.load("models\inceptionv3\\1\inceptionv3.onnx")

# Check the opset version
print(onnx.helper.printable_graph(model.graph))