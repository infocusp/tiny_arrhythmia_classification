# utils/model_selector.py
from models.model.cnn_attentaion_bilstm_improve import EnhancedCNNModel
from models.model.cnn1d_resent_18 import ResNet1DModel

def get_model(model_name, input_shape, num_classes):
    if model_name == "EnhancedCNNModel":
        return EnhancedCNNModel(input_shape, num_classes).get_model()
    elif model_name == "ResNet1DModel":
        return ResNet1DModel(input_shape, num_classes).get_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")
