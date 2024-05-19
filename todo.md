
## Preprocessing of Captions 

### Create Dictionary  [x] 
 * iterate through the captions and get all the words (using tokenizer ) [x]
 * create a dictionary and add those word as key and index as value [x]
 * add 4 extra values <unk> <pad> <end> <start> [x]

### Preprocess the Caption [x]
 * iterate through the captions, find the longest caption [x]
 * iterate again through all the captions, if a caption is less than the max length then add <pad> [x]
 * Also start and end token for the captions need to be added before padding [x]

## Preprocessing of Image [x]

 * Resize the image to 224 * 244 , so the tensor will be of shape 3 * 244 * 244 [x]
 * Normalize the image using mean = [0.485, 0.456, 0.406]
                             std = [0.229, 0.224, 0.225]  [x]

## Encoder
* Shape of the input will be B* 3 * 244 * 244 where B is the batch size 
 ### CNN Encoder [x]
    * Resnet Encoder 
    * Remove the last layer and replace with linear layer with an output dimension as embedd size ( 256 in our case)
    * create forward method 
    * output : B * 256 

## Dataset Preprocessing 
 * iterate through the folder and divide dataset into train, test and validation data [ ]
 * Create Dataset Class that has embedding and coresponding image [ ]
 * create train , test and validation loaders for these dataset [ ]

## Decoder 

### LSTM decoder [x]
 * The first layer would be an embedding layer , where thr caption will be passed 
 * Embeddding size will be set to 256 
 * So our tensor is B * S * 256 
 * LSTM Input : combine each image embedding( feature) with the corresponding caption indices ( i.e output from the embedding layer)
 * How to combine ? : 
 *

## Training  [ ]
* Create training loop [ ]

## Evaluation Metric [ ]
* For Metric we could use Blue Score ( which context size = 4 ) 
* 2 approach for this  ( Decide on these)
    * Use blue score against individual caption prediction 
    * Use blue score against all combined captions for a single image 

## Visualization [ ]
* Use Tensor Board (preferred) ? OR create your own method 


## currently our model uses 1 LSTM layer and then predict one word at a time and then keep on predicting word 
## we could use teacher forcing to pass the hidden state and the cells states 
## next would be to add the attention mechanism 




## IF the data is already split no need to split again since the loader and is created and then at end again it is split so it might override if the seeding is not done !!! Change that ! s

## See if the accuracy is a little better and check the whole flow if something we are doing is wrong 



