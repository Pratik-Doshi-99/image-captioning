
## Preprocessing 

### Create Dictionary ( Done)
 * iterate through the captions and get all the words (using tokenizer )
 * create a dictionary and add those word as key and index as value 
 * add 4 extra values <unk> <pad> <end> <start> 

### Preprocess the Caption ( Done )
 * iterate through the captions, find the longest caption 
 * iterate again through all the captions, if a caption is less than the max length then add <pad> 

# Dataset Preprocessing 

- iterate through the folder and divide dataset into train, test and validation data 


