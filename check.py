import pickle

def print_model_info(model):
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {dir(model)}")
    
    if isinstance(model, dict):
        print("Model is a dictionary. Keys:")
        for key, value in model.items():
            print(f"- {key}: {type(value)}")
            # Print the contents of the dictionary
            if key == 'arima':
                print("  ARIMA Model Summary:")
                print(value.summary())  # Assuming value is an ARIMAResultsWrapper
            elif key == 'garch':
                print("  GARCH Model Summary:")
                print(value)  # Print GARCH model results
            else:
                print(f"  Contents: {value}")  # General contents for other types
    elif hasattr(model, 'head'):  # Assuming it's a DataFrame
        print("Model is a DataFrame. Preview:")
        print(model.head())  # Show the first few rows of the DataFrame
    else:
        print(f"Model is not a dictionary, but a {type(model)}")

def load_model(file_path):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    model_file_path = "./inference-data/models/price_model_arb_1d.pkl"  # Update with your actual file path
    model = load_model(model_file_path)
    print_model_info(model)