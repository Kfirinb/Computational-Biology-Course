Computational Biology Third Assignment - Neural Networks
Presented by Kfir Inbal and CUCUMBER by Or S. Naim

To run our software, kindly use the following command
python main.py #InputFile #LearningSetSize #GradualLearningEnabler #HammingDistanceOptimizer #-1Or0Labels

#InputFile:
The name of the file containing the examples need to be learned. There should be 100 examples in total, 10 for each digit.

#LearningSetSize:
A natural number from 1 to 10, specifying the number of examples should be learned for each digit.

 #GradualLearningEnabler:
Either 0 or 1.
If 1 injected, the program will run for 10 epochs, Learning an additional digit each time.
If 0 injected, the program will run for a single epoch, learning all 10 digits upfront.

#HammingDistanceOptimizer:
Either 0 or 1.
If 1 injected, the model will try to solve examples after 1,001 failed attempts using a Hamming Distance based optimizer.
If 0 injected, the model will skip the optimizing stage.

#-1Or0Labels:
Either -1 or 0.
Specifying how you prefer white cells to be labeled.