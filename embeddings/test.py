import pickle

def load_data_from_pickle(filename):
    """
    Load data from a pickle file.

    Args:
        filename (str): The path to the file from which data will be loaded.

    Returns:
        The loaded data.
    """
    # Open the file in read-binary mode
    with open(filename, 'rb') as file:
        # Use pickle.load to deserialize the data from the file
        data = pickle.load(file)
    
    return data

# Example usage
filename = 'embeddings_20.bin'  # Specify the path to your pickle file

# Load data from the pickle file
loaded_data = load_data_from_pickle(filename)

# Print the loaded data (you can also perform any other operation on the data)
print("Loaded data:")
print('Data')
for k in loaded_data:
    print(k, loaded_data[k].shape)
