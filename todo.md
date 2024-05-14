
## Preprocessing of Captions 

### Create Dictionary ( Done)
 * iterate through the captions and get all the words (using tokenizer )
 * create a dictionary and add those word as key and index as value 
 * add 4 extra values <unk> <pad> <end> <start> 

### Preprocess the Caption ( Done )
 * iterate through the captions, find the longest caption 
 * iterate again through all the captions, if a caption is less than the max length then add <pad> 
 * Also start and end token for the captions need to be added before padding 

## Preprocessing of Image 

 * Resize the image to 224 * 244 , so the tensor will be of shape 3 * 244 * 244 
 * Normalize the image using mean = [0.485, 0.456, 0.406]
                             std = [0.229, 0.224, 0.225]

## Encoder 
 * Create embedding of the images 
 * Shape of the input will be N * 3 * 244 * 244 where N is the batch size 

## Dataset Preprocessing 
 * iterate through the folder and divide dataset into train, test and validation data 
 * Create Dataset Class that has embedding and coresponding image 
 * create train , test and validation loaders for these dataset 

## Decoder 
 * Flatten the emedding 
 * Input : combine each image embedding with the corresponding caption indices 
 * Output will be 





