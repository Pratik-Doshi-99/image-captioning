import pickle
import os

def load_pickle_files(directory, print_image_names=False):
    unique_images = []  # To store unique image file names
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} not found.")
        return
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Check if the file is a .bin file
        if os.path.isfile(filepath) and filename.endswith('.bin'):
            try:
                with open(filepath, 'rb') as file:
                    # Load the pickle object into memory
                    data = pickle.load(file)
                    # Update unique_images with the keys of the loaded dictionary
                    unique_images.extend(data.keys())
                    # Optionally print image file names
                    if print_image_names:
                        print(f"Image file names in {filename}: {list(data.keys())}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return unique_images

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


if __name__ =='__main__':
    # # Example usage
    # filename = 'embeddings_2.bin'  # Specify the path to your pickle file

    # # Load data from the pickle file
    # loaded_data = load_data_from_pickle(filename)

    # # Print the loaded data (you can also perform any other operation on the data)
    # print("Loaded data:")
    # print('Data')
    # for k in loaded_data:
    #     print(k, loaded_data[k].shape)

    
    files = load_pickle_files('.')
    print(f'\n\nTotal Embeddings: {len(files)}; Unique Embeddings: {len(set(files))}')
