import torch
import mlflow.pytorch

# Load the model from MLflow
run_id = '405f6853ea9649dcbd3292f61f37a118' # Replace <run_id> with your actual run ID
model_uri = f"runs:/{run_id}/model" 
print(model_uri)
model = mlflow.pytorch.load_model(model_uri)
model.eval()


def predict(X_new):
    with torch.no_grad():
        # X_new should be a torch tensor with the same shape as your training data
        output = model(X_new)
        predicted = output.argmax(dim=1)
        return predicted
    
