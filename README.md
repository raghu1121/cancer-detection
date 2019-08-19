# cancer-detection

A simple 1DCNN model for cancer detection based on genomic sequencing and mutations.
The results are 

                        precision    recall  

           Daisy          0.80      1.00      
           Control        1.00      0.95      

           accuracy        0.96      
   
   This was not finetuned very well, due to lack of time. 
   This can also be implemented with 2DCNN by reshaping the input_shape. A further adaptation of Resnet architecture 
   might improve the metrics. 
