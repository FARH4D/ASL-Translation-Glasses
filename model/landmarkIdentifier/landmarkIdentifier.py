import numpy as np
import tensorflow as tf

class landmarkIdentifier(object):
    def __init__(
        self
    ):
        self.interpreter = tf.lite.Interpreter(model_path='model/landmarkIdentifier/landmarkIdentifier.tflite', num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self,landmarkList):

        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmarkList], dtype=np.float32))
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        
        result_probs = np.squeeze(result)
        result_index = np.argmax(result_probs)
        confidence = result_probs[result_index]

        return result_index, confidence