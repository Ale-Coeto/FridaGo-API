import cv2
from openvino.runtime import Core

core = Core()
age_gender_model = core.read_model("models/age-gender-recognition-retail-0013/age-gender-recognition-retail-0013.xml")
age_gender_compiled_model = core.compile_model(age_gender_model, "CPU")

def get_input_size(model):
    input_tensor = model.input(0)  # Get the first input tensor
    input_shape = input_tensor.shape
    return (input_shape[3], input_shape[2])  # (width, height)

image = cv2.imread(".png")
input_size = get_input_size(age_gender_model)
image = cv2.resize(image, (input_size[0], input_size[1]))
input_tensor = image.transpose(2, 0, 1)
input_tensor = input_tensor.reshape(1, *input_tensor.shape)

age_gender_result = age_gender_compiled_model([input_tensor])[age_gender_compiled_model.output(0)]

# print(age_gender_result)
predicted_age = age_gender_result[0][0][0][0]
predicted_gender = age_gender_result[0][1][0][0]
# print(len(age_gender_result))

# print("Predicted age:", predicted_age)
# print("Predicted gender:", predicted_gender)

predicted_age = predicted_age * 100  # Multiply by 100 to get the actual age
predicted_gender = "Female" if predicted_gender > 0.5 else "Male"

print("Predicted age:", predicted_age)
print("Predicted gender:", predicted_gender)