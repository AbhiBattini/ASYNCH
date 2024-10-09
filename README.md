# ASYNCH
The architecture for the ASYNCH Lead Synthesis AI Model, which synthesizes the 12 leads of an ECG from Leads I, II, and V1

Highlights the way that the:
* Linear Recurrent Neural Networks were designed
* ASYNCH Model is a hybrid CNN-Transformer architecture; and uses alternating encoder-decoder networks in the Transformer portion
* Output layer is a Feed-Forward Neural Network instead of a Convolutional Layer for finer detail
